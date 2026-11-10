"""Busca los choques de cada evaluacion de CCDiff y propone escenas candidatas para BeamNG.

Para cada carpeta de evaluacion (<grupo>/<ejecucion>/scene_edit_eval/data.hdf5):
  1. Detecta choques entre agentes con cajas orientadas (centroide, rumbo y tamano
     de cada agente en cada paso de 0,1 s): primer instante de contacto por pareja.
  2. Marca si interviene algun agente CONTROLADO por CCDiff (los demas reproducen
     nuScenes). Los indices controlados se leen del log del job ("set guidance dim").
  3. Comprueba con el mapa de nuScenes si los dos coches estaban sobre la calzada
     (capa drivable_area) en el momento del choque.
  4. Escribe choques.csv (todos los choques), escenas.json (cada escena con choque:
     su PRIMER choque, si se acepta y por que no) y candidatas.json (solo las aceptadas).

El accidente de una escena es su PRIMER choque con un agente controlado: despues,
tbsim sigue sin fisica (los coches se atraviesan) y los choques siguientes no son
reales. Si ese primer choque no es valido, la escena entera se descarta.

Escena candidata = choque con algun agente controlado, que no exista ya en el primer
paso, con los dos coches sobre la calzada, con velocidad relativa >= --v_min m/s y
PLAUSIBLE hasta el impacto: no antes de T_MIN, sin coches pegados al inicio, sin giros,
aceleraciones o derrapes imposibles, entre turismos y sin agentes que aparezcan o
desaparezcan cerca antes del impacto (umbrales al principio del archivo;
el motivo de cada descarte queda en la columna "motivos" de choques.csv).

Se ejecuta dentro del contenedor (necesita h5py, numpy, shapely y nuscenes-devkit):
  python /workspace/scripts/analizar_choques.py /workspace/evaluaciones/ctrl_30esc \\
      --log /workspace/scripts/logs/ccdiff-eval_5483.out
(lo hace analizar.sh). Con --sin_mapa se salta la comprobacion de la calzada.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

DT = 0.1  # paso de simulacion de tbsim (s)


# ---------------------------------------------------------------- log del job
def leer_controlados(log_path):
    """Devuelve {ejecucion: {(escena, frame_inicio): {ctrl: [indices controlados], idx: n}}}.

    El log tiene una cabecera "==== <ckpt> w_collision=..." por ejecucion y, en cada
    escena, "set guidance dim:  [..]" seguido de "['scene-XXXX']" y "[frame]".
    El nombre de la ejecucion se toma de la linea "[evaluacion] salida: .../<nombre>".
    idx es el indice de la escena en eval_scenes (lo que espera --escenas_idx).
    """
    res, actual, pendiente, escena = {}, None, None, None
    indice = {}  # posicion de cada lote (1 escena) en eval_scenes, para --escenas_idx
    patron_lista = re.compile(r"\[([0-9,\s]*)\]")
    for linea in Path(log_path).read_text(errors="replace").splitlines():
        if linea.startswith("[evaluacion] salida:"):
            actual = Path(linea.split(":", 1)[1].strip()).name
            res.setdefault(actual, {})
            indice[actual] = 0
        elif linea.startswith("no valid scenes in this batch") and actual is not None:
            indice[actual] += 1
        elif linea.startswith("set guidance dim:"):
            m = patron_lista.search(linea)
            pendiente = [int(x) for x in m.group(1).split(",") if x.strip()] if m else None
        elif re.fullmatch(r"\['scene-\d+'\]", linea.strip()):
            escena = linea.strip()[2:-2]
        elif escena and re.fullmatch(r"\[\d+\]", linea.strip()):
            if actual is not None:
                res[actual][(escena, int(linea.strip()[1:-1]))] = dict(ctrl=pendiente, idx=indice[actual])
                indice[actual] += 1
            escena, pendiente = None, None
    return res


# ------------------------------------------------------------- geometria
def esquinas(c, yaw, largo, ancho):
    """Esquinas (4, 2) de la caja orientada."""
    d = np.array([[largo / 2, ancho / 2], [largo / 2, -ancho / 2],
                  [-largo / 2, -ancho / 2], [-largo / 2, ancho / 2]])
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return d @ R.T + c


def solapan(a, b):
    """Teorema del eje separador para dos rectangulos (4, 2)."""
    for caja in (a, b):
        for i in range(2):
            eje = caja[i + 1] - caja[i]
            n = np.array([-eje[1], eje[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def tipo_impacto(yaw_a, pos_a, pos_b):
    """Donde recibe el golpe A: FRONT / REAR / SIDE (angulo de B visto desde A)."""
    rel = pos_b - pos_a
    ang = math.atan2(rel[1], rel[0]) - yaw_a
    ang = (ang + math.pi) % (2 * math.pi) - math.pi
    if abs(ang) <= math.pi / 4:
        return "FRONT"
    if abs(ang) >= 3 * math.pi / 4:
        return "REAR"
    return "SIDE"


# ------------------------------------------------------------- mapa nuScenes
class Calzada:
    def __init__(self, dataroot):
        from nuscenes.map_expansion.map_api import NuScenesMap
        if not (Path(dataroot) / "maps" / "expansion").is_dir():
            raise FileNotFoundError(f"{dataroot}/maps/expansion (map expansion de nuScenes)")
        self._clase, self._root, self._mapas = NuScenesMap, str(dataroot), {}

    def sobre_calzada(self, mapa, x, y):
        if mapa not in self._mapas:
            self._mapas[mapa] = self._clase(dataroot=self._root, map_name=mapa)
        capas = self._mapas[mapa].layers_on_point(float(x), float(y), layer_names=["drivable_area"])
        return bool(capas.get("drivable_area"))


def nombre_mapa(grupo_h5):
    if "map_names" not in grupo_h5:
        return None
    v = np.asarray(grupo_h5["map_names"]).ravel()
    for x in v:
        s = x.decode() if isinstance(x, bytes) else str(x)
        if s and s != "nan":
            return s.split(":")[-1]  # trajdata usa "nusc_trainval:boston-seaport"
    return None


# ------------------------------------------------------------- plausibilidad
# Umbrales de lo que un coche normal puede hacer ANTES del choque (lo de despues no
# importa: en BeamNG manda la fisica). Se aplican solo con el coche en marcha.
YAW_RATE_MAX = 1.0    # rad/s: mas es girar sobre si mismo (con v >= 3 m/s)
ACEL_MAX = 5.0        # m/s^2: aceleracion sostenida (media de 0,3 s); un turismo rapido da ~5
FRENO_MAX = 9.5       # m/s^2: frenada sostenida; una frenada de emergencia en seco llega a ~9-10
DERRAPE_MAX = 35.0    # grados entre la direccion de la velocidad y el morro (con v >= 3 m/s)
LARGO_MAX = 7.0       # m: mas largo = autobus/camion (BeamNG usa turismos)
T_MIN = 1.0           # s: un choque antes de esto no tiene fase previa (coches pegados al inicio)
RADIO_POP = 20.0      # m: un agente que aparece o desaparece (anotacion de nuScenes que empieza o
                      # acaba) a esta distancia de los implicados antes del choque invalida la escena
HUECO_MIN = 2.0       # m: separacion minima entre las cajas al inicio


def _ang(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def cinematica(pos, yaw, validos, i, t_fin):
    """Peores valores de yaw rate, aceleracion, frenada y derrape del agente i en [0, t_fin]."""
    ok = validos[i, : t_fin + 1]
    idx = np.where(ok)[0]
    res = dict(yaw_rate=0.0, acel=0.0, freno=0.0, derrape=0.0)
    if len(idx) < 4:
        return res
    p, y = pos[i, idx], yaw[i, idx]
    dt = np.diff(idx) * DT
    v = np.linalg.norm(np.diff(p, axis=0), axis=-1) / dt
    rapido = v >= 3.0
    yr = np.abs(_ang(np.diff(y))) / dt
    if rapido.any():
        res["yaw_rate"] = float(np.max(yr[rapido]))
        dirv = np.arctan2(*(np.diff(p, axis=0)[:, ::-1].T))
        res["derrape"] = float(np.degrees(np.max(np.abs(_ang(dirv - y[1:]))[rapido])))
    if len(v) >= 4:
        a = np.diff(v) / dt[1:]
        a = np.convolve(a, np.ones(3) / 3, mode="valid")  # media de 0,3 s contra el ruido
        if len(a):
            res["acel"] = float(max(0.0, np.max(a)))
            res["freno"] = float(max(0.0, -np.min(a)))
    return res


def apariciones(validos):
    """(agente, paso, tipo) de cada vez que un agente aparece o desaparece a mitad de escena."""
    res = []
    for k in range(validos.shape[0]):
        cambios = np.where(validos[k, 1:] != validos[k, :-1])[0] + 1
        for s_ in cambios:
            res.append((k, int(s_), "aparece" if validos[k, s_] else "desaparece"))
    return res


def pop_cerca(pos, validos, pops, i, j, t):
    """Agentes que aparecen/desaparecen cerca de i o j antes del choque (paso t)."""
    hay = []
    for k, s_, tipo in pops:
        if k in (i, j) or s_ > t:
            continue
        donde = pos[k, s_] if tipo == "aparece" else pos[k, s_ - 1]
        for m in (i, j):
            ref = pos[m, s_] if validos[m, s_] else pos[m, min(t, pos.shape[1] - 1)]
            if np.isfinite(ref).all() and np.linalg.norm(donde - ref) <= RADIO_POP:
                hay.append(f"{k}{'+' if tipo == 'aparece' else '-'}@{s_ * DT:.1f}s")
                break
    return hay


def hueco_inicial(pos, validos, largo, ancho, i, j):
    """Distancia aproximada entre las cajas en el primer paso en que existen las dos."""
    ambos = np.where(validos[i] & validos[j])[0]
    if not len(ambos):
        return np.inf
    t0 = ambos[0]
    d = np.linalg.norm(pos[i, t0] - pos[j, t0])
    return float(d - (largo[i] + largo[j]) / 4 - (ancho[i] + ancho[j]) / 4)


# ------------------------------------------------------------- analisis
def choques_escena(g, controlados):
    pos = np.asarray(g["centroid"], dtype=float)      # (N, T, 2)
    yaw = np.asarray(g["yaw"], dtype=float)           # (N, T) o (N, T, 1)
    ext = np.asarray(g["extent"], dtype=float)        # (N, T, 3): largo, ancho, alto
    if yaw.ndim == 3:
        yaw = yaw[..., 0]
    n, T = pos.shape[:2]
    # tbsim rellena con (0, 0) los pasos en que un agente aun no ha entrado o ya ha
    # salido de la escena (no con NaN): esos pasos no son posiciones reales.
    validos = np.isfinite(pos).all(-1) & np.isfinite(yaw) & ~(pos == 0).all(-1)
    vel = np.full_like(pos, np.nan)
    vel[:, 1:] = (pos[:, 1:] - pos[:, :-1]) / DT
    vel[:, 1:][~(validos[:, 1:] & validos[:, :-1])] = np.nan  # sin saltos desde/hacia el relleno
    largo = np.nanmedian(ext[..., 0], axis=1)
    ancho = np.nanmedian(ext[..., 1], axis=1)
    radio = np.hypot(largo, ancho) / 2
    pops = apariciones(validos)

    eventos = []
    for i in range(n):
        for j in range(i + 1, n):
            ambos = validos[i] & validos[j]
            if not ambos.any():
                continue
            d = np.linalg.norm(pos[i] - pos[j], axis=-1)
            cerca = np.where(ambos & (d <= radio[i] + radio[j]))[0]
            for t in cerca:
                if solapan(esquinas(pos[i, t], yaw[i, t], largo[i], ancho[i]),
                           esquinas(pos[j, t], yaw[j, t], largo[j], ancho[j])):
                    vi = vel[i, t] if np.isfinite(vel[i, t]).all() else np.zeros(2)
                    vj = vel[j, t] if np.isfinite(vel[j, t]).all() else np.zeros(2)
                    eventos.append(dict(
                        a=i, b=j, t=int(t), t_s=round(t * DT, 1),
                        ya_al_inicio=bool(t == np.where(ambos)[0][0]),
                        controlado_a=i in controlados, controlado_b=j in controlados,
                        v_a=round(float(np.linalg.norm(vi)), 2), v_b=round(float(np.linalg.norm(vj)), 2),
                        v_rel=round(float(np.linalg.norm(vi - vj)), 2),
                        impacto_a=tipo_impacto(yaw[i, t], pos[i, t], pos[j, t]),
                        impacto_b=tipo_impacto(yaw[j, t], pos[j, t], pos[i, t]),
                        xa=float(pos[i, t, 0]), ya=float(pos[i, t, 1]),
                        xb=float(pos[j, t, 0]), yb=float(pos[j, t, 1]),
                    ))
                    ki, kj = cinematica(pos, yaw, validos, i, t), cinematica(pos, yaw, validos, j, t)
                    motivos = []
                    if t * DT < T_MIN:
                        motivos.append("demasiado_pronto")
                    hueco = hueco_inicial(pos, validos, largo, ancho, i, j)
                    if hueco < HUECO_MIN:
                        motivos.append("pegados_al_inicio")
                    if max(ki["yaw_rate"], kj["yaw_rate"]) > YAW_RATE_MAX:
                        motivos.append("giro_imposible")
                    if max(ki["acel"], kj["acel"]) > ACEL_MAX:
                        motivos.append("acel_imposible")
                    if max(ki["freno"], kj["freno"]) > FRENO_MAX:
                        motivos.append("frenada_imposible")
                    if max(ki["derrape"], kj["derrape"]) > DERRAPE_MAX:
                        motivos.append("derrape")
                    if max(largo[i], largo[j]) > LARGO_MAX:
                        motivos.append("vehiculo_grande")
                    cerca = pop_cerca(pos, validos, pops, i, j, t)
                    if cerca:
                        motivos.append("aparece_desaparece_cerca")
                    eventos[-1].update(
                        hueco_inicial=round(hueco, 2),
                        largo_a=round(float(largo[i]), 2), largo_b=round(float(largo[j]), 2),
                        yaw_rate_max=round(max(ki["yaw_rate"], kj["yaw_rate"]), 2),
                        acel_max=round(max(ki["acel"], kj["acel"]), 2),
                        freno_max=round(max(ki["freno"], kj["freno"]), 2),
                        derrape_max=round(max(ki["derrape"], kj["derrape"]), 1),
                        pops_cerca=" ".join(cerca),
                        plausible=not motivos, motivos=" ".join(motivos))
                    break  # solo el primer contacto de cada pareja
    return eventos, n, T


def analizar_ejecucion(carpeta, ctrl_por_escena, calzada, v_min, todos=False):
    import h5py
    h5 = carpeta / "scene_edit_eval" / "data.hdf5"
    if not h5.is_file():
        return None
    filas, candidatas, escenas, n_escenas, sin_ctrl = [], [], [], 0, 0
    with h5py.File(h5, "r") as f:
        for clave in sorted(f.keys()):
            m = re.fullmatch(r"(scene-\d+)_(\d+)", clave)
            if not m:
                continue
            escena, frame = m.group(1), int(m.group(2))
            n_escenas += 1
            info = ctrl_por_escena.get((escena, frame)) or {}
            ctrl, idx = info.get("ctrl"), info.get("idx")
            if ctrl is None:
                sin_ctrl += 1
                ctrl = []
            eventos, n, T = choques_escena(f[clave], set(ctrl))
            mapa = nombre_mapa(f[clave])
            for e in eventos:
                e.update(ejecucion=carpeta.name, escena=escena, frame_inicio=frame, mapa=mapa, idx=idx,
                         n_agentes=n, controlados=" ".join(map(str, ctrl)))
                if calzada is not None and mapa:
                    e["calzada_a"] = calzada.sobre_calzada(mapa, e["xa"], e["ya"])
                    e["calzada_b"] = calzada.sobre_calzada(mapa, e["xb"], e["yb"])
                razones = [] if (e["plausible"] or todos) else e["motivos"].split()
                if e["v_rel"] < v_min:
                    razones.append("v_rel_baja")
                if not (e.get("calzada_a", True) and e.get("calzada_b", True)):
                    razones.append("fuera_calzada")
                e["aceptable"] = not razones
                e["razones"] = " ".join(razones)
                e["primero"] = False
                e["candidato"] = False
                filas.append(e)
            # El accidente de la escena es el PRIMER choque con un agente controlado: tras
            # el, tbsim sigue sin fisica (los coches se atraviesan) y lo demas no es real.
            # Si ese primero no vale, la escena se descarta (no se busca otro mas tarde).
            relevantes = sorted((e for e in eventos if not e["ya_al_inicio"]
                                 and (e["controlado_a"] or e["controlado_b"])), key=lambda e: e["t"])
            if not relevantes:
                continue
            t1 = relevantes[0]["t"]
            primeros = [e for e in relevantes if e["t"] == t1]
            for e in primeros:
                e["primero"] = True
            buenos = [e for e in primeros if e["aceptable"]]
            elegido = max(buenos, key=lambda e: e["v_rel"]) if buenos else primeros[0]
            if buenos:
                elegido["candidato"] = True
            choques = [dict(a=e["a"], b=e["b"], t=e["t"]) for e in relevantes]
            registro = dict(ejecucion=carpeta.name, escena=escena, frame_inicio=frame, idx=idx, mapa=mapa,
                            n_choques=len(relevantes), mejor=elegido, choques=choques,
                            aceptada=bool(buenos), razon="" if buenos else elegido["razones"])
            escenas.append(registro)
            if buenos:
                candidatas.append(registro)
    return dict(filas=filas, candidatas=candidatas, escenas=escenas, n_escenas=n_escenas, sin_ctrl=sin_ctrl)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("grupo", help="carpeta con una subcarpeta por ejecucion (p. ej. evaluaciones/ctrl_30esc)")
    ap.add_argument("--log", action="append", default=[],
                    help="log(s) del job de evaluacion, para saber que agentes estaban controlados")
    ap.add_argument("--nuscenes", default="/workspace/datasets/nuscenes",
                    help="raiz de nuScenes (con maps/expansion)")
    ap.add_argument("--sin_mapa", action="store_true", help="no comprobar si los coches estaban en la calzada")
    ap.add_argument("--v_min", type=float, default=2.0,
                    help="velocidad relativa minima (m/s) para considerar un choque como accidente")
    ap.add_argument("--sin_filtro", action="store_true",
                    help="no exigir plausibilidad (ver columnas motivos/plausible en choques.csv)")
    args = ap.parse_args()

    grupo = Path(args.grupo)
    ctrl = {}
    for log in args.log:
        for ejec, d in leer_controlados(log).items():
            ctrl.setdefault(ejec, {}).update(d)

    calzada = None
    if not args.sin_mapa:
        try:
            calzada = Calzada(args.nuscenes)
        except Exception as e:  # noqa: BLE001
            print(f"AVISO: no puedo cargar los mapas de nuScenes ({e}); sigo sin comprobar la calzada")

    todas_filas, todas_cand, todas_esc = [], [], []
    print(f"\n{'ejecucion':<40} {'escenas':>7} {'con choque':>10} {'choque ctrl':>11} {'candidatas':>10}")
    for carpeta in sorted(p for p in grupo.iterdir() if p.is_dir()):
        r = analizar_ejecucion(carpeta, ctrl.get(carpeta.name, {}), calzada, args.v_min, args.sin_filtro)
        if r is None:
            continue
        escenas_choque = {(e["escena"], e["frame_inicio"]) for e in r["filas"] if not e["ya_al_inicio"]}
        escenas_ctrl = {(e["escena"], e["frame_inicio"]) for e in r["filas"]
                        if not e["ya_al_inicio"] and (e["controlado_a"] or e["controlado_b"])}
        print(f"{carpeta.name:<40} {r['n_escenas']:>7} {len(escenas_choque):>10} "
              f"{len(escenas_ctrl):>11} {len(r['candidatas']):>10}")
        if r["sin_ctrl"]:
            print(f"   AVISO: {r['sin_ctrl']} escenas sin indices de agentes controlados (falta --log de ese job)")
        con_mapa = [e for e in r["filas"] if "calzada_a" in e and not e["ya_al_inicio"]]
        if len(con_mapa) >= 5:
            fuera = sum(not (e["calzada_a"] and e["calzada_b"]) for e in con_mapa) / len(con_mapa)
            if fuera > 0.5:
                print(f"   AVISO: el {fuera:.0%} de los choques sale fuera de la calzada; puede que las "
                      "coordenadas de trajdata no coincidan con las del mapa. Revisalo o usa --sin_mapa.")
        razones = {}
        for esc in r["escenas"]:
            for m in esc["razon"].split():
                razones[m] = razones.get(m, 0) + 1
        if razones:
            print("   escenas descartadas (su PRIMER choque) por: "
                  + ", ".join(f"{k} {v}" for k, v in sorted(razones.items())))
        todas_filas += r["filas"]
        todas_cand += r["candidatas"]
        todas_esc += r["escenas"]

    if not todas_filas:
        sys.exit(f"No hay choques (o no hay data.hdf5) bajo {grupo}")

    campos = ["ejecucion", "escena", "idx", "frame_inicio", "mapa", "n_agentes", "controlados", "a", "b",
              "controlado_a", "controlado_b", "t", "t_s", "ya_al_inicio", "v_a", "v_b", "v_rel",
              "impacto_a", "impacto_b", "calzada_a", "calzada_b", "plausible", "motivos", "hueco_inicial",
              "largo_a", "largo_b", "yaw_rate_max", "acel_max", "freno_max", "derrape_max", "pops_cerca", "aceptable", "razones",
              "primero", "candidato", "xa", "ya", "xb", "yb"]
    with open(grupo / "choques.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(todas_filas)
    todas_cand.sort(key=lambda c: (c["ejecucion"], -c["mejor"]["v_rel"]))
    (grupo / "candidatas.json").write_text(json.dumps(todas_cand, indent=1))
    todas_esc.sort(key=lambda c: (c["ejecucion"], c["escena"]))
    (grupo / "escenas.json").write_text(json.dumps(todas_esc, indent=1))

    print("\nCandidatas (PRIMER choque de cada escena, ordenadas por velocidad relativa):")
    for c in todas_cand:
        e = c["mejor"]
        print(f"  {c['ejecucion']:<36} {c['escena']} (frame {c['frame_inicio']:>3})  t={e['t_s']:>4.1f}s  "
              f"agentes {e['a']}-{e['b']}  v_rel={e['v_rel']:>5.1f} m/s  {e['impacto_a']}/{e['impacto_b']}")
    for ejec in sorted({c["ejecucion"] for c in todas_cand}):
        idx = sorted({c["idx"] for c in todas_cand if c["ejecucion"] == ejec and c["idx"] is not None})
        if idx:
            print(f"\n{ejec}: para verlas con video -> EXTRA=\"--escenas_idx {','.join(map(str, idx))} ...\"")
    print(f"\nGuardado: {grupo / 'choques.csv'}, {grupo / 'candidatas.json'} y {grupo / 'escenas.json'} "
          "(todas las escenas con choque, aceptadas o no, con el motivo)")


if __name__ == "__main__":
    main()
