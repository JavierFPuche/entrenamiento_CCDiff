#!/bin/bash
#SBATCH --job-name=ccdiff-debug
#SBATCH --partition=H200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%j.out

# PRIMER trabajo a lanzar, equivale a la linea "debug" de scripts/run_train.sh.
#   - Construye la cache de trajdata de nuScenes en /raid/$USER/ccdiff_datos/cache (solo la 1a vez;
#     es lo que mas tarda, por eso 6 h de limite).
#   - Entrena 200 pasos en modo debug: valida cada 5, guarda cada 10 y hace un
#     rollout cada 10. Si termina bien, la cadena completa funciona.
# En el DGX:  mkdir -p logs && sbatch entrenar_debug.sh

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
  python /workspace/lanzar_entrenamiento.py --debug --pasos 200 \
    --cache /datos/ccdiff/cache --salida /datos/ccdiff/resultados
