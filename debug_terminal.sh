#!/bin/bash
# Terminal interactiva dentro del contenedor en una H200 (30 min).
# En el DGX:  bash debug_terminal.sh
# Dentro:     python /workspace/lanzar_entrenamiento.py --comprobar \
#                 --cache /datos/ccdiff/cache --salida /datos/ccdiff/resultados
echo "Running on $(hostname)"
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

IMAGE_NAME="ccdiff_train"
# Carpeta del proyecto: la de este script (o la que indiques con CCDIFF_DIR=...).
PATH_TO_SOURCE_CODE=${CCDIFF_DIR:-$(cd "$(dirname "$0")" && pwd)}
NUSCENES=/raid/${USER}/datasets/nuscenes
# Lo pesado (cache de trajdata ~ decenas de GB, checkpoints) va a /raid, no al home
DATOS=/raid/${USER}/ccdiff_datos
mkdir -p "${DATOS}/cache" "${DATOS}/resultados"
OUTPUT_SQSH=/raid/${USER}/enroot/sqsh/${IMAGE_NAME}_v1.sqsh

# /workspace = esta carpeta (lanzador) | /datos/nuscenes = nuScenes (solo lectura)
# /datos/ccdiff = cache de trajdata y resultados (en /raid)
MOUNTS="${PATH_TO_SOURCE_CODE}:/workspace,${NUSCENES}:/datos/nuscenes:ro,${DATOS}:/datos/ccdiff"

srun \
    --partition=H200 \
    --gres=gpu:h200:1 \
    --mem=256G \
    --cpus-per-task=16 \
    --time=00:30:00 \
    --job-name=ccdiff_bash \
    --container-image="$OUTPUT_SQSH" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace \
    --pty bash
