#!/bin/bash
# Instala CCDiff y sus dependencias (trajdata fork AIasd, Pplan, tbsim) con Python 3.8.
# Lo ejecuta el Dockerfile como root. Se puede probar fuera de Docker con:
#   PY=/ruta/python3.8 SRC=/tmp/ccdiff_src TORCH_INDEX=https://download.pytorch.org/whl/cpu bash instalar_ccdiff.sh
#
# Por que pip 24.0 y no uv (a diferencia de tus otras imagenes):
#   tbsim fija pytorch-lightning==1.8.3.post0, cuyos metadatos llevan "torch>=1.9.*",
#   un especificador no estandar. pip >= 24.1 y uv lo rechazan; pip 24.0 aun lo acepta
#   (por eso el requirements.txt de tbsim fija pip==24.0).
set -euo pipefail

# Proteccion: este script es para el build de Docker. Fuera de un contenedor solo
# se ejecuta si se indica a proposito el interprete (PY=...), p. ej. un venv de 3.8.
if [ ! -f /.dockerenv ] && [ -z "${PY:-}" ]; then
  echo "Este script lo ejecuta el Dockerfile durante 'docker build'; no hace falta lanzarlo a mano."
  echo "Si de verdad quieres instalar fuera de Docker, indica un Python 3.8 de un venv:"
  echo "  PY=/ruta/venv38/bin/python SRC=/ruta/codigo bash instalar_ccdiff.sh"
  exit 1
fi
set -x

PY=${PY:-python3}
SRC=${SRC:-/opt/ccdiff}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu118}
AQUI=$(cd "$(dirname "$0")" && pwd)
C="-c ${CONSTRAINTS:-$AQUI/constraints.txt}"
PIP="$PY -m pip install --no-cache-dir"

# Commits fijos (30/10/2026) para que la imagen sea reproducible
CCDIFF_REPO=https://github.com/HenryLHH/CCDiff.git
CCDIFF_SHA=beaddb81c41186c6bc7ed284a4c1f0b9deddc6ad
TRAJDATA_REPO=https://github.com/AIasd/trajdata.git
TRAJDATA_SHA=79ea54ce36d8b2b09b4bb4b43579ef2520352158
PPLAN_REPO=https://github.com/NVlabs/spline-planner.git
PPLAN_SHA=c3d172917f337531209149546670503fa8ed1d3c

clonar() {  # clonar <repo> <commit> <destino>
  mkdir -p "$3" && cd "$3"
  git init -q && git remote add origin "$1"
  git fetch -q --depth 1 origin "$2" && git checkout -q FETCH_HEAD
  cd - >/dev/null
}

# 1) pip compatible con pytorch-lightning 1.8.3.post0
$PIP "pip==24.0" "setuptools<70" wheel

# 2) PyTorch (CUDA 11.8 -> soporta sm_90 de las H200)
$PIP torch==2.0.1 torchvision==0.15.2 torchtext==0.15.2 --index-url "$TORCH_INDEX"

# 3) Codigo fuente
clonar "$CCDIFF_REPO" "$CCDIFF_SHA" "$SRC/CCDiff"
clonar "$TRAJDATA_REPO" "$TRAJDATA_SHA" "$SRC/CCDiff/third_party/trajdata"
clonar "$PPLAN_REPO" "$PPLAN_SHA" "$SRC/CCDiff/third_party/Pplan"

# 4) trajdata (fork AIasd)
$PIP $C -r "$SRC/CCDiff/third_party/trajdata/trajdata_requirements.txt"
$PIP $C -e "$SRC/CCDiff/third_party/trajdata"

# 5) Pplan (sin dependencias propias)
$PIP $C -e "$SRC/CCDiff/third_party/Pplan"

# 6) tbsim: sus install_requires fijan torchtext==0.14.1 (que arrastra torch 1.13.1),
#    asi que se instalan a mano con torchtext 0.15.2 y tbsim va con --no-deps.
$PIP $C "pytorch-lightning==1.8.3.post0" "numpy==1.23.4" wandb pyemd h5py \
        "imageio-ffmpeg==0.4.9" casadi "protobuf==3.20.1" "einops==0.6.0" "imageio==2.9.0" \
        tensorboard
#    l5kit 1.5.0 (lo importa tbsim) pide torch<2 y numpy~=1.19: sin sus pines.
$PIP --no-deps "l5kit==1.5.0"
$PIP $C opencv-contrib-python-headless pymap3d scipy transforms3d zarr pyyaml ptable \
        shapely typing-extensions bokeh
$PIP --no-deps -e "$SRC/CCDiff/third_party/tbsim"

# 7) CCDiff: su setup.py fija torch==1.13.1 -> --no-deps
$PIP $C "networkx==2.8.8"
$PIP --no-deps -e "$SRC/CCDiff"

# 8) Comprobaciones
$PY -m pip check || echo "AVISO: pip check informa conflictos (esperados: l5kit/tbsim/ccdiff piden torch<2)"
cd "$SRC/CCDiff"
$PY - <<'PY'
import torch, torchvision, torchtext, pytorch_lightning as pl, numpy as np
import trajdata, tbsim, ccdiff, l5kit, Pplan
print("torch", torch.__version__, "| PL", pl.__version__, "| numpy", np.__version__)
# Importa toda la cadena de entrenamiento sin ejecutarla
import importlib.util, sys
spec = importlib.util.spec_from_file_location("train", "ccdiff/examples/train.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from tbsim.configs.registry import get_registered_experiment_config
cfg = get_registered_experiment_config("trajdata_nusc_ccdiff")
print("config trajdata_nusc_ccdiff OK:", cfg.algo.name, "| pasos", cfg.train.training.num_steps)
PY
echo "INSTALACION CCDIFF OK"
