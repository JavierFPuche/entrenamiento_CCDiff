#!/bin/bash
#SBATCH --job-name=ccdiff-install
#SBATCH --partition=H200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out

# En el DGX, DESDE la carpeta del proyecto:  mkdir -p logs && sbatch pull_and_install.sh
# Descarga la imagen con enroot y comprueba que PyTorch ve la H200 (sm_90).

echo "Running on $(hostname)"
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

DH_USERNAME="javierfuentesprobesafe"
IMAGE_NAME="ccdiff_train"
# Carpeta del proyecto: aquella desde la que lanzas sbatch (o la que indiques con
# CCDIFF_DIR=...). Se monta como /workspace dentro del contenedor.
PATH_TO_SOURCE_CODE=${CCDIFF_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
if [ ! -f "${PATH_TO_SOURCE_CODE}/lanzar_entrenamiento.py" ]; then
  echo "ERROR: ${PATH_TO_SOURCE_CODE} no es la carpeta del proyecto (falta lanzar_entrenamiento.py)."
  echo "Lanza sbatch desde la carpeta del proyecto, o exporta CCDIFF_DIR=/ruta/a/la/carpeta."
  exit 1
fi
OUTPUT_SQSH=/raid/${USER}/enroot/sqsh/${IMAGE_NAME}_v1.sqsh

mkdir -p "/raid/${USER}/enroot/sqsh/"

if [ ! -f "$OUTPUT_SQSH" ]; then
  echo "Descargando la imagen de Docker Hub con Enroot..."
  enroot import -o "$OUTPUT_SQSH" docker://${DH_USERNAME}/${IMAGE_NAME}:v1
else
  echo "El archivo $OUTPUT_SQSH ya existe, omitiendo la descarga."
  echo "(Si has subido una imagen nueva, borra el .sqsh y vuelve a lanzar este script.)"
fi

srun \
  --gpus=1 \
  --container-image="$OUTPUT_SQSH" \
  --container-mounts="$PATH_TO_SOURCE_CODE:/workspace" \
  bash -c '
    echo "🔄 Comprobando el contenedor..."
    ls -l /workspace
    python - <<PY
import torch, pytorch_lightning as pl, trajdata, tbsim, ccdiff
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| PL", pl.__version__)
print("GPU:", torch.cuda.get_device_name(0), "| capacidad", torch.cuda.get_device_capability(0))
print("Arquitecturas compiladas:", torch.cuda.get_arch_list())
a = torch.randn(2048, 2048, device="cuda")
print("matmul en GPU OK, suma =", float((a @ a).sum()))
PY
    echo "✅ ¡Contenedor descargado y funcionando correctamente!"
  '
