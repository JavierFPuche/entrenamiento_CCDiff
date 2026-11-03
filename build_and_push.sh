#!/bin/bash
# Construye la imagen en tu PC (robesafe-106) y la sube a Docker Hub.
# Ejecutar desde la carpeta entrenamiento_CCDiff:  bash build_and_push.sh

DH_USERNAME="javierfuentesprobesafe"
IMAGE_NAME="ccdiff_train"
TAG_NAME="v1"

MY_UID=$(id -u)
MY_GID=$(id -g)

set -e

echo "Construyendo la imagen local (tarda: compila Python 3.8 + dependencias)..."
docker build ./Docker -t ${IMAGE_NAME}:${TAG_NAME} \
  --build-arg USER=${USER} \
  --build-arg UID=${MY_UID} \
  --build-arg GID=${MY_GID}

echo "Etiquetando para Docker Hub..."
docker tag ${IMAGE_NAME}:${TAG_NAME} ${DH_USERNAME}/${IMAGE_NAME}:${TAG_NAME}

echo "Subiendo a la nube..."
docker push ${DH_USERNAME}/${IMAGE_NAME}:${TAG_NAME}

echo "✅ Imagen subida correctamente como ${DH_USERNAME}/${IMAGE_NAME}:${TAG_NAME}"
