#!/bin/bash
# Terminal interactiva dentro del contenedor en una H200 (30 min).
# En el DGX:  bash debug_terminal.sh
# Dentro:     python /workspace/scripts/lanzar_entrenamiento.py --comprobar
echo "Running on $(hostname)"
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

IMAGE_NAME="ccdiff_train"
REPO_DIR="VLM_ACCIDENTES_CCDIFF"
# Proyecto (misma estructura que tus otros proyectos de Workspace):
#   scripts/   -> este repositorio (git clone)     datasets/nuscenes/      -> nuScenes
#   entrenados/ -> checkpoints del entrenamiento   datasets/trajdata_cache/ -> cache de trajdata
#   pruebas/   -> ejecuciones de debug             entornos/home/          -> HOME del contenedor
# Se monta entero como /workspace. CCDIFF_DIR=... permite usar otra carpeta.
PATH_TO_SOURCE_CODE=${CCDIFF_DIR:-/raid/${USER}/Workspace/${REPO_DIR}}
OUTPUT_SQSH=/raid/${USER}/enroot/sqsh/${IMAGE_NAME}_v1.sqsh

if [ ! -f "${PATH_TO_SOURCE_CODE}/scripts/lanzar_entrenamiento.py" ]; then
  echo "ERROR: no encuentro ${PATH_TO_SOURCE_CODE}/scripts/lanzar_entrenamiento.py"
  echo "Clona el repositorio en ${PATH_TO_SOURCE_CODE}/scripts (o exporta CCDIFF_DIR=/ruta/al/proyecto)."
  exit 1
fi
mkdir -p "${PATH_TO_SOURCE_CODE}"/{entrenados,pruebas,datasets/trajdata_cache,entornos/home}
MOUNTS="${PATH_TO_SOURCE_CODE}:/workspace"

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
