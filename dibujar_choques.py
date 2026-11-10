"""Dibuja como GIF las escenas candidatas de analizar_choques.py, a partir de data.hdf5.

No vuelve a simular nada: usa las trayectorias guardadas, asi que el GIF muestra
exactamente el choque que encontro el analisis (re-simular con --escenas_idx daria
otras muestras de la difusion). Encima del mapa de nuScenes (calzada, pasos de
peatones y aceras) dibuja cada agente como su caja orientada:
  rojo     = agentes controlados por CCDiff
  gris     = agentes que reproducen nuScenes
  borde negro grueso = la pareja del choque, desde el instante del impacto

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


def dibujar(g, cand, salida, nuscenes, radio, fps):
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
    par, t_choque = {e["a"], e["b"]}, e["t"]
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

    cajas, rastros = [], []
    for i in range(n):
        color = "#d62728" if i in ctrl else "#7f7f7f"
        p = Polygon(np.zeros((5, 2)), closed=True, fc=color, ec=color, alpha=0.85, lw=1, zorder=3)
        ax.add_patch(p)
        cajas.append(p)
        (r,) = ax.plot([], [], color=color, lw=1.4, alpha=0.6, zorder=2)
        rastros.append(r)
    texto = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(fc="white", ec="none", alpha=0.8), zorder=5)
    titulo = (f"{cand['escena']}  {cand['ejecucion']}\nchoque agentes {e['a']}-{e['b']} en t={e['t_s']} s, "
              f"v_rel={e['v_rel']} m/s ({e['impacto_a']}/{e['impacto_b']})")
    ax.set_title(titulo, fontsize=8)

    def frame(t):
        for i in range(n):
            if validos[i, t]:
                cajas[i].set_xy(esquinas(pos[i, t], yaw[i, t], largo[i], ancho[i]))
                cajas[i].set_visible(True)
                ok = validos[i, : t + 1]
                rastros[i].set_data(pos[i, : t + 1][ok, 0], pos[i, : t + 1][ok, 1])
            else:
                cajas[i].set_visible(False)
            if i in par and t >= t_choque:
                cajas[i].set_edgecolor("black")
                cajas[i].set_linewidth(2.5)
        texto.set_text(f"t = {t * DT:4.1f} s" + ("   CHOQUE" if t >= t_choque else ""))
        return cajas + rastros + [texto]

    anim = FuncAnimation(fig, frame, frames=range(T), blit=False)
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
    args = ap.parse_args()

    import h5py
    grupo = Path(args.grupo)
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
        salida = destino / f"{c['ejecucion']}__{c['escena']}.gif"
        with h5py.File(h5, "r") as f:
            if clave not in f:
                print(f"AVISO: {clave} no esta en {h5}")
                continue
            hay_mapa = dibujar(f[clave], c, salida, args.nuscenes, args.radio, args.fps)
        print(f"{salida}{'' if hay_mapa else '  (sin mapa)'}")


if __name__ == "__main__":
    main()
