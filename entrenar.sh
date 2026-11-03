#!/bin/bash
#SBATCH --job-name=ccdiff-train
#SBATCH --partition=H200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=23:59:59
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%j.out

# Entrenamiento completo de CCDiff (100k pasos, configuracion del paper) -> entrenados/
# En el DGX, desde VLM_ACCIDENTES_CCDIFF/scripts:
#   sbatch entrenar.sh                       # empezar
#   sbatch entrenar.sh --reanudar /workspace/entrenados/<nombre>/<version>/checkpoints/<archivo>.ckpt
#                                            # continuar si se corta a las 24 h
# Cualquier argumento extra se pasa a lanzar_entrenamiento.py (--pasos, --batch, --wandb...).
# Para usar W&B:  export WANDB_APIKEY=...  antes de sbatch, y anade --wandb.

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
if [ ! -d "${PATH_TO_SOURCE_CODE}/datasets/nuscenes/v1.0-trainval" ]; then
  echo "ERROR: no encuentro ${PATH_TO_SOURCE_CODE}/datasets/nuscenes/v1.0-trainval"
  echo "Descomprime alli v1.0-trainval_meta.tgz y el map expansion (ver README)."
  exit 1
fi

srun \
  --container-image="$OUTPUT_SQSH" \
  --container-mounts="$MOUNTS" \
  --container-workdir=/workspace \
  python /workspace/scripts/lanzar_entrenamiento.py --gpus 1 "$@"
