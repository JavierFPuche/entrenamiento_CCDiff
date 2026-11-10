"""Busca los choques de cada evaluacion de CCDiff y propone escenas candidatas para BeamNG.

Para cada carpeta de evaluacion (<grupo>/<ejecucion>/scene_edit_eval/data.hdf5):
  1. Detecta choques entre agentes con cajas orientadas (centroide, rumbo y tamano
     de cada agente en cada paso de 0,1 s): primer instante de contacto por pareja.
  2. Marca si interviene algun agente CONTROLADO por CCDiff (los demas reproducen
     nuScenes). Los indices controlados se leen del log del job ("set guidance dim").
  3. Comprueba con el mapa de nuScenes si los dos coches estaban sobre la calzada
     (capa drivable_area) en el momento del choque.
  4. Escribe choques.csv (todos los choques) y candidatas.json (una escena por fila,
     con el mejor choque) y muestra un resumen por ejecucion.

Escena candidata = choque con algun agente controlado, que no exista ya en el primer
paso, con los dos coches sobre la calzada y con velocidad relativa >= --v_min m/s.

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


# ------------------------------------------------------------- analisis
def choques_escena(g, controlados):
    pos = np.asarray(g["centroid"], dtype=float)      # (N, T, 2)
    yaw = np.asarray(g["yaw"], dtype=float)           # (N, T) o (N, T, 1)
    ext = np.asarray(g["extent"], dtype=float)        # (N, T, 3): largo, ancho, alto
    if yaw.ndim == 3:
        yaw = yaw[..., 0]
    n, T = pos.shape[:2]
    validos = np.isfinite(pos).all(-1) & np.isfinite(yaw)
    vel = np.full_like(pos, np.nan)
    vel[:, 1:] = (pos[:, 1:] - pos[:, :-1]) / DT
    largo = np.nanmedian(ext[..., 0], axis=1)
    ancho = np.nanmedian(ext[..., 1], axis=1)
    radio = np.hypot(largo, ancho) / 2

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
                    break  # solo el primer contacto de cada pareja
    return eventos, n, T


def analizar_ejecucion(carpeta, ctrl_por_escena, calzada, v_min):
    import h5py
    h5 = carpeta / "scene_edit_eval" / "data.hdf5"
    if not h5.is_file():
        return None
    filas, candidatas, n_escenas, sin_ctrl = [], [], 0, 0
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
                e["candidato"] = bool(
                    (e["controlado_a"] or e["controlado_b"]) and not e["ya_al_inicio"]
                    and e["v_rel"] >= v_min
                    and e.get("calzada_a", True) and e.get("calzada_b", True))
                filas.append(e)
            buenos = [e for e in eventos if e["candidato"]]
            if buenos:
                mejor = max(buenos, key=lambda e: e["v_rel"])
                candidatas.append(dict(ejecucion=carpeta.name, escena=escena, frame_inicio=frame, idx=idx,
                                       mapa=mapa, n_choques=len(buenos), mejor=mejor))
    return dict(filas=filas, candidatas=candidatas, n_escenas=n_escenas, sin_ctrl=sin_ctrl)


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

    todas_filas, todas_cand = [], []
    print(f"\n{'ejecucion':<40} {'escenas':>7} {'con choque':>10} {'choque ctrl':>11} {'candidatas':>10}")
    for carpeta in sorted(p for p in grupo.iterdir() if p.is_dir()):
        r = analizar_ejecucion(carpeta, ctrl.get(carpeta.name, {}), calzada, args.v_min)
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
        todas_filas += r["filas"]
        todas_cand += r["candidatas"]

    if not todas_filas:
        sys.exit(f"No hay choques (o no hay data.hdf5) bajo {grupo}")

    campos = ["ejecucion", "escena", "idx", "frame_inicio", "mapa", "n_agentes", "controlados", "a", "b",
              "controlado_a", "controlado_b", "t", "t_s", "ya_al_inicio", "v_a", "v_b", "v_rel",
              "impacto_a", "impacto_b", "calzada_a", "calzada_b", "candidato", "xa", "ya", "xb", "yb"]
    with open(grupo / "choques.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(todas_filas)
    todas_cand.sort(key=lambda c: (c["ejecucion"], -c["mejor"]["v_rel"]))
    (grupo / "candidatas.json").write_text(json.dumps(todas_cand, indent=1))

    print("\nCandidatas (mejor choque de cada escena, ordenadas por velocidad relativa):")
    for c in todas_cand:
        e = c["mejor"]
        print(f"  {c['ejecucion']:<36} {c['escena']} (frame {c['frame_inicio']:>3})  t={e['t_s']:>4.1f}s  "
              f"agentes {e['a']}-{e['b']}  v_rel={e['v_rel']:>5.1f} m/s  {e['impacto_a']}/{e['impacto_b']}")
    for ejec in sorted({c["ejecucion"] for c in todas_cand}):
        idx = sorted({c["idx"] for c in todas_cand if c["ejecucion"] == ejec and c["idx"] is not None})
        if idx:
            print(f"\n{ejec}: para verlas con video -> EXTRA=\"--escenas_idx {','.join(map(str, idx))} ...\"")
    print(f"\nGuardado: {grupo / 'choques.csv'} y {grupo / 'candidatas.json'}")


if __name__ == "__main__":
    main()
