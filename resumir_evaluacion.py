"""Resume y compara las evaluaciones de CCDiff (stats.json de scene_editor).

Busca <carpeta>/*/scene_edit_eval/stats.json (una subcarpeta por combinacion
checkpoint + peso de colision) y muestra, para cada metrica, la media sobre las
escenas en una tabla con una columna por combinacion. Solo usa la biblioteca
estandar: funciona en el DGX fuera del contenedor y en el PC.

  python3 resumir_evaluacion.py /raid/$USER/Workspace/VLM_ACCIDENTES_CCDIFF/evaluaciones/ccdiff_5413
  python3 resumir_evaluacion.py <carpeta> --filtro coll offroad     # solo metricas que contengan esos textos
  python3 resumir_evaluacion.py <carpeta> --csv resumen.csv
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

# Metricas que interesan primero para generar accidentes (si existen con ese texto)
PRIORITARIAS = ("coll", "offroad", "off_road", "disk", "ade", "fde", "comfort", "speed", "acc", "jerk")


def numeros(v):
    """Aplana listas anidadas y devuelve los valores numericos finitos."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return [float(v)] if math.isfinite(v) else []
    if isinstance(v, list):
        out = []
        for x in v:
            out.extend(numeros(x))
        return out
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("carpeta", help="carpeta con una subcarpeta por evaluacion")
    ap.add_argument("--filtro", nargs="*", default=None, help="mostrar solo metricas que contengan estos textos")
    ap.add_argument("--csv", default=None, help="guardar tambien la tabla en un CSV")
    args = ap.parse_args()

    raiz = Path(args.carpeta)
    archivos = sorted(raiz.glob("*/**/stats.json"))
    if not archivos:
        sys.exit(f"No hay stats.json bajo {raiz}")

    runs = {}
    for f in archivos:
        nombre = f.relative_to(raiz).parts[0]
        datos = json.loads(f.read_text())
        escenas = len(datos.get("scene_index", []))
        medias = {}
        for k, v in datos.items():
            if k == "scene_index":
                continue
            vals = numeros(v)
            if vals:
                medias[k] = sum(vals) / len(vals)
        runs[nombre] = (escenas, medias)

    claves = sorted({k for _, m in runs.values() for k in m})
    if args.filtro:
        claves = [k for k in claves if any(t.lower() in k.lower() for t in args.filtro)]
    claves.sort(key=lambda k: (not any(p in k.lower() for p in PRIORITARIAS), k))

    nombres = list(runs)
    ancho = max([len(k) for k in claves] + [8])
    print(f"\n{'metrica (media por escena)':<{ancho}}  " + "  ".join(f"{n:>22}" for n in nombres))
    print(f"{'escenas':<{ancho}}  " + "  ".join(f"{runs[n][0]:>22d}" for n in nombres))
    print("-" * (ancho + 24 * len(nombres)))
    for k in claves:
        celdas = []
        for n in nombres:
            v = runs[n][1].get(k)
            celdas.append(f"{v:>22.4f}" if v is not None else f"{'-':>22}")
        print(f"{k:<{ancho}}  " + "  ".join(celdas))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["metrica"] + nombres)
            w.writerow(["escenas"] + [runs[n][0] for n in nombres])
            for k in claves:
                w.writerow([k] + [runs[n][1].get(k, "") for n in nombres])
        print(f"\nCSV guardado en {args.csv}")


if __name__ == "__main__":
    main()
