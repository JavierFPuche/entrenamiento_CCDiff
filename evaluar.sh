#!/bin/bash
#SBATCH --job-name=ccdiff-eval
#SBATCH --partition=H200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=/raid/%u/Workspace/VLM_ACCIDENTES_CCDIFF/scripts/logs/%x_%j.out
#SBATCH --chdir=/tmp
# Rutas ABSOLUTAS vistas desde el nodo de calculo (/raid, sin /nfs). Slurm cambia %u por tu usuario.

# Evaluacion de CCDiff en bucle cerrado SIN BeamNG (scene_editor.py, como scripts/run_eval.sh).
# Prueba cada checkpoint con cada peso del guiado de colision sobre las MISMAS escenas:
#   w_collision = -50 -> guiado hacia el choque (generar accidentes)
#   w_collision =   0 -> sin guiado de colision (trafico normal, referencia)
# Resultados en VLM_ACCIDENTES_CCDIFF/evaluaciones/<entrenamiento>/<ckpt>_wcol<peso>/
# y resumen al final del log (resumir_evaluacion.py).
#
# En el DGX, desde VLM_ACCIDENTES_CCDIFF/scripts:
#   sbatch evaluar.sh
#   ESCENAS=50 CKPTS="iter100000.ckpt" WCOLS="-50" sbatch evaluar.sh      # otra combinacion
# ~80-100 s por escena: 20 escenas x 2 checkpoints x 2 pesos ~ 2-3 h.

ENTRENAMIENTO=${ENTRENAMIENTO:-ccdiff_5413/run0}
CKPTS=${CKPTS:-"iter100000.ckpt iter50000.ckpt"}
WCOLS=${WCOLS:-"-50 0"}
ESCENAS=${ESCENAS:-20}

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

for CKPT in ${CKPTS}; do
  for W in ${WCOLS}; do
    echo "================ ${CKPT}  w_collision=${W} ================"
    srun \
      --container-image="$OUTPUT_SQSH" \
      --container-mounts="$MOUNTS" \
      --container-workdir=/workspace \
      python /workspace/scripts/lanzar_evaluacion.py \
        --ckpt_dir "/workspace/entrenados/${ENTRENAMIENTO}" --ckpt "${CKPT}" \
        --w_collision "${W}" --escenas "${ESCENAS}" \
      || echo "AVISO: fallo la evaluacion de ${CKPT} con w_collision=${W}"
  done
done

echo "================ RESUMEN ================"
python3 "${PATH_TO_SOURCE_CODE}/scripts/resumir_evaluacion.py" \
  "${PATH_TO_SOURCE_CODE}/evaluaciones/$(dirname "${ENTRENAMIENTO}")"
