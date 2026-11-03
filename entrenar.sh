#!/bin/bash
#SBATCH --job-name=ccdiff-train
#SBATCH --partition=H200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=23:59:59
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%j.out

# Entrenamiento completo de CCDiff (100k pasos, configuracion del paper).
# En el DGX:
#   sbatch entrenar.sh                                   # empezar
#   sbatch entrenar.sh --reanudar /datos/ccdiff/resultados/<nombre>/<version>/checkpoints/<archivo>.ckpt
#     (en el DGX esa carpeta es /raid/$USER/ccdiff_datos/resultados/...)  # continuar si se corta a las 24 h
# Cualquier argumento extra se pasa a lanzar_entrenamiento.py (--pasos, --batch, --wandb...).
# Para usar W&B:  export WANDB_APIKEY=...  antes de sbatch, y anade --wandb.

echo "Running on $(hostname)"
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

IMAGE_NAME="ccdiff_train"
# Carpeta del proyecto: aquella desde la que lanzas sbatch (o la que indiques con
# CCDIFF_DIR=...). Se monta como /workspace dentro del contenedor.
PATH_TO_SOURCE_CODE=${CCDIFF_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
if [ ! -f "${PATH_TO_SOURCE_CODE}/lanzar_entrenamiento.py" ]; then
  echo "ERROR: ${PATH_TO_SOURCE_CODE} no es la carpeta del proyecto (falta lanzar_entrenamiento.py)."
  echo "Lanza sbatch desde la carpeta del proyecto, o exporta CCDIFF_DIR=/ruta/a/la/carpeta."
  exit 1
fi
# nuScenes: NUSCENES=... si lo indicas; si no, <proyecto>/datasets/nuscenes; si no, /raid/$USER/datasets/nuscenes
if [ -z "${NUSCENES:-}" ]; then
  if [ -d "${PATH_TO_SOURCE_CODE}/datasets/nuscenes" ]; then
    NUSCENES=${PATH_TO_SOURCE_CODE}/datasets/nuscenes
  else
    NUSCENES=/raid/${USER}/datasets/nuscenes
  fi
fi
if [ ! -d "${NUSCENES}/v1.0-trainval" ]; then
  echo "ERROR: no encuentro ${NUSCENES}/v1.0-trainval (descomprime v1.0-trainval_meta.tgz ahi, o exporta NUSCENES=/ruta)."
  exit 1
fi
echo "nuScenes: ${NUSCENES}"
# Lo pesado (cache de trajdata ~ decenas de GB, checkpoints) va a /raid, no al home
DATOS=/raid/${USER}/ccdiff_datos
mkdir -p "${DATOS}/cache" "${DATOS}/resultados"
OUTPUT_SQSH=/raid/${USER}/enroot/sqsh/${IMAGE_NAME}_v1.sqsh
MOUNTS="${PATH_TO_SOURCE_CODE}:/workspace,${NUSCENES}:/datos/nuscenes:ro,${DATOS}:/datos/ccdiff"

srun \
  --container-image="$OUTPUT_SQSH" \
  --container-mounts="$MOUNTS" \
  --container-workdir=/workspace \
  python /workspace/lanzar_entrenamiento.py --gpus 1 \
    --cache /datos/ccdiff/cache --salida /datos/ccdiff/resultados "$@"
