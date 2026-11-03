#!/bin/bash
#SBATCH --job-name=ccdiff-debug
#SBATCH --partition=H200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --output=/raid/%u/Workspace/VLM_ACCIDENTES_CCDIFF/scripts/logs/%x_%j.out
#SBATCH --chdir=/tmp
# Rutas ABSOLUTAS vistas desde el nodo de calculo (/raid, sin /nfs): asi da igual
# desde donde lances sbatch en el nodo de acceso. Slurm cambia %u por tu usuario.

# PRIMER trabajo a lanzar, equivale a la linea "debug" de scripts/run_train.sh de CCDiff.
#   - Construye la cache de trajdata en datasets/trajdata_cache (solo la 1a vez;
#     es lo que mas tarda, por eso 6 h de limite).
#   - Entrena 200 pasos en modo debug (valida cada 5, guarda cada 10, rollout cada 10)
#     y guarda la salida en pruebas/. Si termina bien, la cadena completa funciona.
# En el DGX, desde VLM_ACCIDENTES_CCDIFF/scripts:  sbatch entrenar_debug.sh

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
  python /workspace/scripts/lanzar_entrenamiento.py --debug --pasos 200
