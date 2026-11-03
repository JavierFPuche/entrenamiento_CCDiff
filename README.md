# Entrenamiento de CCDiff en el DGX (H200)

CCDiff (Cruise/CMU, CVPR 2025) no publica checkpoints, así que hay que entrenarlo: 100 000 pasos sobre nuScenes, que es la configuración del paper. Esta carpeta sigue la misma estructura que tus otros entrenamientos: imagen Docker → Docker Hub → enroot → Slurm.

| Archivo | Dónde se usa | Qué hace |
|---|---|---|
| `Docker/Dockerfile` | PC | Imagen: CUDA 11.8, Python 3.8, CCDiff con trajdata (fork AIasd), Pplan y tbsim |
| `Docker/instalar_ccdiff.sh` | PC (lo llama el Dockerfile) | Instala todo con commits fijos y comprueba los imports |
| `Docker/constraints.txt` | PC | Versiones que ninguna dependencia puede cambiar |
| `build_and_push.sh` | PC | `docker build` + push a `javierfuentesprobesafe/ccdiff_train:v1` |
| `Makefile.ccdiff` | PC | `make -f Makefile.ccdiff build`, `... run` (terminal) y `... debug` (prueba en la 4090). Se llama así porque las herramientas remotas no pueden escribir archivos `Makefile`; si quieres, renómbralo con `mv Makefile.ccdiff Makefile` |
| `pull_and_install.sh` | DGX (`sbatch`) | `enroot import` y prueba de la H200 |
| `debug_terminal.sh` | DGX | Terminal interactiva en el contenedor (30 min) |
| `entrenar_debug.sh` | DGX (`sbatch`) | **Primer trabajo:** construye la caché de trajdata y entrena 200 pasos en modo debug |
| `entrenar.sh` | DGX (`sbatch`) | Entrenamiento completo, con opción de reanudar |
| `lanzar_entrenamiento.py` | dentro del contenedor | Adapta la configuración de CCDiff al DGX y ejecuta su `train.py` sin modificarlo |

## Diferencias con tus otras imágenes y por qué

- **PyTorch 2.0.1 + CUDA 11.8, no el 1.13.1 de CCDiff.** La versión 1.13.1 solo existe para CUDA 11.6/11.7 y no tiene kernels para `sm_90` (H100/H200), así que falla con *"no kernel image is available"*. La 2.0.1 es la más cercana con soporte para `sm_90` y sigue admitiendo Python 3.8, que es lo que exigen CCDiff, tbsim y trajdata. Las dependencias que fijan torch 1.13 (`ccdiff`, `tbsim` y `l5kit`) se instalan con `--no-deps`, y lo que necesitan se instala a mano. `pip check` avisará de esos conflictos; es lo esperado.
- **pip 24.0 en lugar de uv.** tbsim fija `pytorch-lightning==1.8.3.post0`, cuyos metadatos (`torch>=1.9.*`) no son estándar. uv y pip ≥ 24.1 lo rechazan. Comprobado: esa versión de Lightning no usa `torch._six`, así que funciona con torch 2.
- **El código de CCDiff va dentro de la imagen** (`/opt/ccdiff`, commits del 30/10/2026). `/workspace` se monta con esta carpeta (la carpeta desde la que lanzas `sbatch`, puede estar en cualquier sitio, p. ej. `~/VLM_ACCIDENTES_CCDIFF`). Lo pesado va a `/raid/$USER/ccdiff_datos`, montado como `/datos/ccdiff`: la caché de trajdata (`cache/`) y los checkpoints (`resultados/`).
- **Rutas fijas de CCDiff.** Sus configs traen `/mnt/disks/sdb/nuscenes` y `~/.unified_data_cache` en varios sitios (entrenamiento y rollouts). `lanzar_entrenamiento.py` las sustituye todas por los montajes del DGX sin tocar el repositorio.
- **Problemas de `train.py` dentro de `sbatch`, ya resueltos en el lanzador:**
  - Pregunta por teclado si la carpeta de salida existe, así que el lanzador usa un nombre único por trabajo.
  - No puede reanudar, así que el lanzador inyecta `ckpt_path`.
  - Exige W&B, así que el lanzador usa TensorBoard por defecto.

## Orden de trabajo

### 1. En tu PC (`robesafe-106`)
```bash
cd /mnt/secundario/doctorado/world_model/entrenamientos/entrenamiento_CCDiff
docker login                       # cuenta javierfuentesprobesafe
bash build_and_push.sh             # la instalación termina con "INSTALACION CCDIFF OK"
```

### 2. Datos en el DGX
nuScenes requiere una cuenta en nuscenes.org. Para entrenar **no** hacen falta las imágenes ni el LiDAR, solo:
- `v1.0-trainval_meta` (metadatos, unos 0,4 GB)
- **Map expansion v1.3**: descomprímelo dentro de `maps/` (genera `maps/expansion/` y `maps/basemap/`)

Estructura esperada en `/raid/$USER/datasets/nuscenes/`:
```
nuscenes/
├── v1.0-trainval/     (los .json de metadatos)
└── maps/
    ├── expansion/     (boston-seaport.json, singapore-*.json)
    └── *.png          (máscaras raster de los 4 mapas, vienen con los metadatos)
```

### 3. En el DGX
```bash
# Copia esta carpeta al DGX (donde quieras) y lanza TODO desde dentro de ella
mkdir -p logs
sbatch pull_and_install.sh         # descarga la imagen y prueba la H200
cat logs/ccdiff-install_*.out      # debe listar sm_90 y "matmul en GPU OK"

sbatch entrenar_debug.sh           # caché de trajdata + 200 pasos de prueba
tail -f logs/ccdiff-debug_*.out

sbatch entrenar.sh                 # entrenamiento completo
```
Antes de gastar horas de GPU puedes revisar la configuración final con `bash debug_terminal.sh` y luego, dentro, `python /workspace/lanzar_entrenamiento.py --comprobar`.

### 4. Si el trabajo llega al límite de 24 h
```bash
ls /raid/$USER/ccdiff_datos/resultados/ccdiff_<job>/*/checkpoints/
# la misma carpeta se ve como /datos/ccdiff dentro del contenedor:
sbatch entrenar.sh --reanudar /datos/ccdiff/resultados/ccdiff_<job>/<version>/checkpoints/<archivo>.ckpt
```

### 5. Seguimiento
```bash
# TensorBoard (desde una sesión del DGX con el puerto reenviado)
tensorboard --logdir /raid/$USER/ccdiff_datos/resultados
```
Para usar W&B: `export WANDB_APIKEY=...` y `sbatch entrenar.sh --wandb`.

## Pendiente de verificar en el primer arranque real
- **Tiempo por paso en la H200.** No hay referencia publicada, así que el `entrenar_debug.sh` dará la primera medida. Con ella se decide si 24 h bastan o si hay que reanudar.
- **Batch.** El config de CCDiff usa `batch_size = 4` (comentado `# 100`). Con 141 GB en la H200 seguramente cabe bastante más (`--batch`). Si se cambia, conviene revisar el learning rate.
- **Caché de trajdata.** La primera vez trajdata procesa todo nuScenes. Se guarda en `/raid/$USER/ccdiff_datos/cache` y se reutiliza en los siguientes trabajos.
