#!/bin/bash
#SBATCH --job-name=ccdiff-gifs
#SBATCH --partition=H200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/raid/%u/Workspace/VLM_ACCIDENTES_CCDIFF/scripts/logs/%x_%j.out
#SBATCH --chdir=/tmp
# La GPU no se usa: se pide solo porque la particion H200 la puede exigir. Si tu
# cluster permite trabajos sin GPU, quita la linea --gres y entrara antes en cola.

# Dibuja como GIF las escenas candidatas (dibujar_choques.py) a partir de los data.hdf5,
# sin volver a simular. Requiere haber pasado antes analizar.sh por el mismo grupo.
#
# En el DGX, desde VLM_ACCIDENTES_CCDIFF/scripts:
#   GRUPO=ctrl_30esc sbatch dibujar.sh
#   GRUPO=ctrl_30esc EXTRA="--ejecucion iter50000_wcolm50_lr1_ctrl10" sbatch dibujar.sh
# Resultado: evaluaciones/<GRUPO>/gifs/<ejecucion>__<escena>.gif

GRUPO=${GRUPO:?Indica GRUPO=<subcarpeta de evaluaciones>}
EXTRA=${EXTRA:-""}   # p. ej. EXTRA="--radio 60" o EXTRA="--escena scene-0095"

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
PATH_TO_SOURCE_CODE=${CCDIFF_DIR:-/raid/${USER}/Workspace/VLM_ACCIDENTES_CCDIFF}
OUTPUT_SQSH=/raid/${USER}/enroot/sqsh/ccdiff_train_v1.sqsh

srun \
  --container-image="$OUTPUT_SQSH" \
  --container-mounts="${PATH_TO_SOURCE_CODE}:/workspace" \
  --container-workdir=/workspace \
  python /workspace/scripts/dibujar_choques.py "/workspace/evaluaciones/${GRUPO}" ${EXTRA}
