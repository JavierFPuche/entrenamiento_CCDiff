"""Dibuja como GIF las escenas candidatas de analizar_choques.py, a partir de data.hdf5.

No vuelve a simular nada: usa las trayectorias guardadas, asi que el GIF muestra
exactamente el choque que encontro el analisis (re-simular con --escenas_idx daria
otras muestras de la difusion). Encima del mapa de nuScenes (calzada, pasos de
peatones y aceras) dibuja cada agente como su caja orientada:
  azul     = ego (con --objetivo_ego no esta controlado: reproduce nuScenes)
  rojo     = agentes controlados por CCDiff (el adversario, con contorno amarillo)
  gris     = agentes que reproducen nuScenes
  borde negro grueso = la pareja del PRIMER choque, desde el instante del impacto
  borde naranja      = choques posteriores (sin fisica: no son parte del accidente)
  contorno discontinuo con X = agente que ha dejado de existir (acaba su anotacion)

  python /workspace/scripts/dibujar_choques.py /workspace/evaluaciones/ctrl_30esc
  python ... --ejecucion iter50000_wcolm50_lr1_ctrl10 --escena scene-0095
Salida: <grupo>/gifs/<ejecucion>__<escena>.gif (lo lanza dibujar.sh).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

DT = 0.1


def esquinas(c, yaw, largo, ancho):
    # caja con un pico en el frente para ver hacia donde mira el coche
    d = np.array([[largo / 2, ancho / 2], [largo / 2 + 0.6, 0], [largo / 2, -ancho / 2],
                  [-largo / 2, -ancho / 2], [-largo / 2, ancho / 2]])
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return d @ R.T + c


def fondo_mapa(ax, nuscenes, mapa, caja):
    """Pinta la calzada y alrededores. Devuelve False si no hay mapa disponible."""
    if not nuscenes or not mapa:
        return False
    try:
        from nuscenes.map_expansion.map_api import NuScenesMap
        nmap = NuScenesMap(dataroot=str(nuscenes), map_name=mapa)
    except Exception as e:  # noqa: BLE001
        print(f"   (sin mapa: {e})")
        return False
    estilos = {"drivable_area": "#d9d9d9", "walkway": "#efe6d2", "ped_crossing": "#c7d7ec"}
    registros = nmap.get_records_in_patch(caja, list(estilos), mode="intersect")
    for capa, color in estilos.items():
        for token in registros.get(capa, []):
            rec = nmap.get(capa, token)
            poligonos = rec["polygon_tokens"] if capa == "drivable_area" else [rec["polygon_token"]]
            for pt in poligonos:
                poly = nmap.extract_polygon(pt)
                if poly.is_empty:
                    continue
                x, y = poly.exterior.xy
                ax.fill(x, y, color=color, zorder=0, lw=0)
    return True


def dibujar(g, cand, salida, nuscenes, radio, fps, tras_choque=2.0):
    pos = np.asarray(g["centroid"], dtype=float)
    yaw = np.asarray(g["yaw"], dtype=float)
    if yaw.ndim == 3:
        yaw = yaw[..., 0]
    ext = np.asarray(g["extent"], dtype=float)
    n, T = pos.shape[:2]
    validos = np.isfinite(pos).all(-1) & np.isfinite(yaw) & ~(pos == 0).all(-1)
    largo = np.nanmedian(ext[..., 0], axis=1)
    ancho = np.nanmedian(ext[..., 1], axis=1)

    e = cand["mejor"]
    ctrl = {int(x) for x in str(e.get("controlados", "")).split()}
    ego, adv = cand.get("ego"), cand.get("adversario")
    par, t_choque = {e["a"], e["b"]}, e["t"]
    # momento del primer choque de cada agente (el principal y los posteriores, sin fisica)
    t_de = {}
    for ch in cand.get("choques", [dict(a=e["a"], b=e["b"], t=e["t"])]):
        for k in (ch["a"], ch["b"]):
            t_de[k] = min(t_de.get(k, 10 ** 9), ch["t"])
    cx, cy = (e["xa"] + e["xb"]) / 2, (e["ya"] + e["yb"]) / 2
    caja = (cx - radio, cy - radio, cx + radio, cy + radio)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    hay_mapa = fondo_mapa(ax, nuscenes, cand.get("mapa"), caja)
    if not hay_mapa:
        ax.set_facecolor("#f4f4f4")
    ax.set_xlim(caja[0], caja[2])
    ax.set_ylim(caja[1], caja[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    cajas, rastros, fantasmas, cruces = [], [], [], []
    # ultimo paso valido de cada agente: si deja de existir (fin de la anotacion de
    # nuScenes), se deja su contorno discontinuo con una cruz en la ultima posicion
    ultimo = np.full(n, -1)
    for i in range(n):
        color = "#1f77b4" if i == ego else ("#d62728" if i in ctrl else "#7f7f7f")
        p = Polygon(np.zeros((5, 2)), closed=True, fc=color, ec="#ffbf00" if i == adv else color,
                    alpha=0.85, lw=2.5 if i == adv else 1, zorder=3)
        ax.add_patch(p)
        cajas.append(p)
        (r,) = ax.plot([], [], color=color, lw=1.4, alpha=0.6, zorder=2)
        rastros.append(r)
        f = Polygon(np.zeros((5, 2)), closed=True, fill=False, ec=color, ls="--", lw=1, alpha=0.7, zorder=2)
        f.set_visible(False)
        ax.add_patch(f)
        fantasmas.append(f)
        (x,) = ax.plot([], [], "x", color=color, ms=7, mew=1.5, zorder=2)
        cruces.append(x)
    texto = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(fc="white", ec="none", alpha=0.8), zorder=5)
    titulo = (f"{cand['escena']}  {cand['ejecucion']}\n1er choque agentes {e['a']}-{e['b']} en t={e['t_s']} s, "
              f"v_rel={e['v_rel']} m/s ({e['impacto_a']}/{e['impacto_b']})")
    if not cand.get("aceptada", True):
        titulo += f"\nDESCARTADA: {cand.get('razon', '')}"
    ax.set_title(titulo, fontsize=8, color="black" if cand.get("aceptada", True) else "#b00000")

    def frame(t):
        if t == 0:
            ultimo[:] = -1  # matplotlib puede volver a dibujar desde el principio
        for i in range(n):
            if validos[i, t]:
                ultimo[i] = t
                fantasmas[i].set_visible(False)
                cruces[i].set_data([], [])
                cajas[i].set_xy(esquinas(pos[i, t], yaw[i, t], largo[i], ancho[i]))
                cajas[i].set_visible(True)
                ok = validos[i, : t + 1]
                rastros[i].set_data(pos[i, : t + 1][ok, 0], pos[i, : t + 1][ok, 1])
            else:
                cajas[i].set_visible(False)
                if ultimo[i] >= 0:
                    u = ultimo[i]
                    fantasmas[i].set_xy(esquinas(pos[i, u], yaw[i, u], largo[i], ancho[i]))
                    fantasmas[i].set_visible(True)
                    cruces[i].set_data([pos[i, u, 0]], [pos[i, u, 1]])
            if i in par and t >= t_choque:
                cajas[i].set_edgecolor("black")
                cajas[i].set_linewidth(2.5)
            elif i in t_de and t >= t_de[i]:
                cajas[i].set_edgecolor("#ff7f0e")  # choque posterior (sin fisica)
                cajas[i].set_linewidth(2.0)
        texto.set_text(f"t = {t * DT:4.1f} s" + ("   CHOQUE" if t >= t_choque else ""))
        return cajas + rastros + fantasmas + cruces + [texto]

    # Tras el impacto tbsim no tiene fisica: los coches se atraviesan y siguen. Eso no
    # es parte del accidente (en BeamNG lo resuelve la fisica), asi que se corta.
    fin = T if tras_choque is None else min(T, t_choque + int(round(tras_choque / DT)) + 1)
    anim = FuncAnimation(fig, frame, frames=range(fin), blit=False)
    anim.save(str(salida), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return hay_mapa


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("grupo", help="carpeta del grupo (con candidatas.json de analizar_choques.py)")
    ap.add_argument("--ejecucion", default=None, help="solo esta ejecucion (subcarpeta)")
    ap.add_argument("--escena", action="append", default=None, help="solo estas escenas (se puede repetir)")
    ap.add_argument("--nuscenes", default="/workspace/datasets/nuscenes")
    ap.add_argument("--radio", type=float, default=40.0, help="mitad del lado de la vista (m)")
    ap.add_argument("--fps", type=int, default=10, help="10 = tiempo real")
    ap.add_argument("--tras_choque", type=float, default=2.0,
                    help="segundos que se dibujan despues del impacto (sin fisica, los coches se atraviesan)")
    ap.add_argument("--completo", action="store_true", help="dibujar los 10 s enteros")
    ap.add_argument("--todas", action="store_true",
                    help="dibujar tambien las escenas descartadas (escenas.json), con el motivo en el titulo")
    args = ap.parse_args()

    import h5py
    grupo = Path(args.grupo)
    if args.todas and (grupo / "escenas.json").is_file():
        candidatas = json.loads((grupo / "escenas.json").read_text())
    else:
        candidatas = json.loads((grupo / "candidatas.json").read_text())
    if args.ejecucion:
        candidatas = [c for c in candidatas if c["ejecucion"] == args.ejecucion]
    if args.escena:
        candidatas = [c for c in candidatas if c["escena"] in args.escena]
    destino = grupo / "gifs"
    destino.mkdir(exist_ok=True)

    for c in candidatas:
        h5 = grupo / c["ejecucion"] / "scene_edit_eval" / "data.hdf5"
        clave = f"{c['escena']}_{c['frame_inicio']}"
        prefijo = "" if c.get("aceptada", True) else "DESCARTADA__"
        salida = destino / f"{prefijo}{c['ejecucion']}__{c['escena']}.gif"
        with h5py.File(h5, "r") as f:
            if clave not in f:
                print(f"AVISO: {clave} no esta en {h5}")
                continue
            hay_mapa = dibujar(f[clave], c, salida, args.nuscenes, args.radio, args.fps,
                               None if args.completo else args.tras_choque)
        print(f"{salida}{'' if hay_mapa else '  (sin mapa)'}")


if __name__ == "__main__":
    main()
