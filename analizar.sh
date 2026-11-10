#!/bin/bash
#SBATCH --job-name=ccdiff-choques
#SBATCH --partition=H200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/raid/%u/Workspace/VLM_ACCIDENTES_CCDIFF/scripts/logs/%x_%j.out
#SBATCH --chdir=/tmp
# La GPU no se usa: se pide solo porque la particion H200 la puede exigir. Si tu
# cluster permite trabajos sin GPU, quita la linea --gres y entrara antes en cola.

# Busca los choques en las evaluaciones de un grupo y propone escenas candidatas para
# BeamNG (analizar_choques.py). Necesita el log del job de evaluacion para saber que
# agentes controlaba CCDiff en cada escena.
#
# En el DGX, desde VLM_ACCIDENTES_CCDIFF/scripts:
#   GRUPO=ctrl_30esc JOBS="5483" sbatch analizar.sh
#   GRUPO=pesos_30esc JOBS="5478 5481" sbatch analizar.sh     # si el grupo vino de varios jobs
# Resultado: evaluaciones/<GRUPO>/choques.csv y candidatas.json, y resumen en el log.

GRUPO=${GRUPO:?Indica GRUPO=<subcarpeta de evaluaciones>}
JOBS=${JOBS:?Indica JOBS=\"<id del job de evaluacion> ...\"}
EXTRA=${EXTRA:-""}   # p. ej. EXTRA="--v_min 3" o EXTRA="--sin_mapa"

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
PATH_TO_SOURCE_CODE=${CCDIFF_DIR:-/raid/${USER}/Workspace/VLM_ACCIDENTES_CCDIFF}
OUTPUT_SQSH=/raid/${USER}/enroot/sqsh/ccdiff_train_v1.sqsh

LOGS=""
for J in ${JOBS}; do
  F=$(ls "${PATH_TO_SOURCE_CODE}"/scripts/logs/*_"${J}".out 2>/dev/null | head -1)
  if [ -z "${F}" ]; then
    echo "AVISO: no encuentro el log del job ${J}"
  else
    LOGS="${LOGS} --log /workspace/scripts/logs/$(basename "${F}")"
  fi
done

srun \
  --container-image="$OUTPUT_SQSH" \
  --container-mounts="${PATH_TO_SOURCE_CODE}:/workspace" \
  --container-workdir=/workspace \
  python /workspace/scripts/analizar_choques.py "/workspace/evaluaciones/${GRUPO}" ${LOGS} ${EXTRA}
