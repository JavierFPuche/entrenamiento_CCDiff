# Imagen para entrenar CCDiff (CVPR 2025) en el DGX (H200).
# Misma estructura que tus otras imagenes: usuario con tu UID/GID, todo se
# instala como root y al final se cambia al usuario sin privilegios.
#
# Diferencias con la plantilla (y por que):
#   - CUDA 11.8 + Python 3.8: CCDiff/tbsim/trajdata estan fijados a Python 3.8,
#     y torch 2.0.1+cu118 es la version mas cercana a la suya (1.13.1) que
#     funciona en sm_90 (H200).
#   - pip 24.0 en vez de uv: pytorch-lightning 1.8.3.post0 (fijado por tbsim)
#     tiene metadatos que uv y pip>=24.1 rechazan.
#   - El codigo de CCDiff va DENTRO de la imagen (/opt/ccdiff, instalacion
#     editable, commits fijos). /workspace se monta con los lanzadores y las salidas.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base

ARG USER
ARG UID
ARG GID
ENV DEBIAN_FRONTEND=noninteractive

# Usuario (igual que tu plantilla)
RUN apt-get update \
    && groupadd -g $GID $USER \
    && useradd --uid $UID --gid $GID -m $USER \
    && apt-get install -y sudo \
    && echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER

# Cada apt-get install con su propio apt-get update en el mismo RUN (como en tu plantilla)
RUN apt-get update \
    && apt-get install -y build-essential cmake git curl ca-certificates vim wget tmux \
        libgl1 libglib2.0-0 libsm6 libxext6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- Python 3.8 (deadsnakes) -------------------------------------------------
# python3.8-tk: tbsim/utils/guidance_metrics.py hace "from turtle import speed"
# (import accidental que no se usa), y turtle necesita tkinter. Sin el, el
# import de train.py falla con "No module named 'tkinter'".
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.8 python3.8-venv python3.8-dev python3.8-distutils \
        python3.8-tk \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3.8 -m ensurepip --upgrade

ENV PATH="${PATH}:/home/$USER/.local/bin"

# --- CCDiff + trajdata (AIasd) + Pplan + tbsim -------------------------------
COPY constraints.txt instalar_ccdiff.sh /tmp/
RUN PY=python3.8 SRC=/opt/ccdiff CONSTRAINTS=/tmp/constraints.txt bash /tmp/instalar_ccdiff.sh \
    && chown -R $UID:$GID /opt/ccdiff

ENV CCDIFF_SRC=/opt/ccdiff/CCDiff
ENV PYTHONUNBUFFERED=1

USER $USER
WORKDIR /workspace
CMD [ "/bin/bash" ]
