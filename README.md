# Entrenamiento de CCDiff en el DGX (H200)

CCDiff (Cruise/CMU, CVPR 2025) no publica checkpoints, así que hay que entrenarlo: 100 000 pasos sobre nuScenes, que es la configuración del paper. Esta carpeta sigue la misma estructura que tus otros entrenamientos: imagen Docker → Docker Hub → enroot → Slurm.

## Estructura del proyecto en el DGX

Igual que tus otros proyectos de `Workspace` (`DIFUSER_SEG`, `RF-DETR`...). Desde el nodo de acceso se ve como `/nfs/raid/$USER/Workspace/VLM_ACCIDENTES_CCDIFF`, y desde los nodos de cálculo como `/raid/$USER/Workspace/VLM_ACCIDENTES_CCDIFF`. Se monta **entera** como `/workspace` dentro del contenedor:

```
VLM_ACCIDENTES_CCDIFF/                  (dentro del contenedor: /workspace)
├── scripts/              este repositorio (git clone); desde aquí se lanza sbatch
│   └── logs/             salidas de Slurm (%x_%j.out)
├── datasets/
│   ├── nuscenes/         v1.0-trainval/ + maps/
│   └── trajdata_cache/   caché que trajdata crea la 1.ª vez (se reutiliza)
├── entrenados/           checkpoints y TensorBoard del entrenamiento completo
├── evaluaciones/         una carpeta por checkpoint × peso de colisión (stats.json, vídeos en viz/)
├── pruebas/              salidas de entrenar_debug.sh
└── entornos/home/        HOME del contenedor (cachés de wandb, matplotlib...)
```
Los scripts crean solos las carpetas que falten, salvo `datasets/nuscenes`.

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
| `evaluar.sh` | DGX (`sbatch`) | Evaluación en bucle cerrado **sin BeamNG**: cada checkpoint × cada peso de colisión, mismas escenas |
| `lanzar_evaluacion.py` | dentro del contenedor | Ejecuta el `scene_editor.py` de CCDiff (como su `run_eval.sh`) con los mismos arreglos que el entrenamiento |
| `resumir_evaluacion.py` | DGX o PC (Python estándar) | Tabla comparativa de las métricas (`stats.json`) de todas las evaluaciones |

## Diferencias con tus otras imágenes y por qué

- **PyTorch 2.0.1 + CUDA 11.8, no el 1.13.1 de CCDiff.** La versión 1.13.1 solo existe para CUDA 11.6/11.7 y no tiene kernels para `sm_90` (H100/H200), así que falla con *"no kernel image is available"*. La 2.0.1 es la más cercana con soporte para `sm_90` y sigue admitiendo Python 3.8, que es lo que exigen CCDiff, tbsim y trajdata. Las dependencias que fijan torch 1.13 (`ccdiff`, `tbsim` y `l5kit`) se instalan con `--no-deps`, y lo que necesitan se instala a mano. `pip check` avisará de esos conflictos; es lo esperado.
- **pip 24.0 en lugar de uv.** tbsim fija `pytorch-lightning==1.8.3.post0`, cuyos metadatos (`torch>=1.9.*`) no son estándar. uv y pip ≥ 24.1 lo rechazan. Comprobado: esa versión de Lightning no usa `torch._six`, así que funciona con torch 2.
- **El código de CCDiff va dentro de la imagen** (`/opt/ccdiff`, commits del 30/10/2026). `/workspace` es el proyecto entero (ver la estructura de arriba).
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

Estructura esperada en `VLM_ACCIDENTES_CCDIFF/datasets/nuscenes/`:
```
nuscenes/
├── v1.0-trainval/     (los .json de metadatos)
└── maps/
    ├── expansion/     (boston-seaport.json, singapore-*.json)
    └── *.png          (máscaras raster de los 4 mapas, vienen con los metadatos)
```

### 3. En el DGX
```bash
cd /nfs/raid/$USER/Workspace/VLM_ACCIDENTES_CCDIFF
git clone git@github.com:JavierFPuche/entrenamiento_CCDiff.git scripts   # la 1.ª vez; luego: git pull
cd scripts && mkdir -p logs
sbatch pull_and_install.sh         # descarga la imagen y prueba la H200
cat logs/ccdiff-install_*.out      # debe listar sm_90 y "matmul en GPU OK"

sbatch entrenar_debug.sh           # caché de trajdata + 200 pasos de prueba
tail -f logs/ccdiff-debug_*.out

sbatch entrenar.sh                 # entrenamiento completo
```
Antes de gastar horas de GPU puedes revisar la configuración final con `bash debug_terminal.sh` y luego, dentro, `python /workspace/scripts/lanzar_entrenamiento.py --comprobar`.

### 4. Si el trabajo llega al límite de 24 h
```bash
ls ../entrenados/ccdiff_<job>/*/checkpoints/
# dentro del contenedor esa carpeta es /workspace/entrenados:
sbatch entrenar.sh --reanudar /workspace/entrenados/ccdiff_<job>/<version>/checkpoints/<archivo>.ckpt
```

### 5. Seguimiento
```bash
# TensorBoard (desde una sesión del DGX con el puerto reenviado)
tensorboard --logdir /nfs/raid/$USER/Workspace/VLM_ACCIDENTES_CCDIFF/entrenados
```
Para usar W&B: `export WANDB_APIKEY=...` y `sbatch entrenar.sh --wandb`.

### 6. Evaluación (sin BeamNG)
Simula escenas de validación de nuScenes en bucle cerrado con el motor de tbsim. Es el mismo tipo de simulación que los *rollouts* del entrenamiento, y sirve para elegir checkpoint y comprobar el guiado de colisión antes de pasar a BeamNG.
```bash
sbatch evaluar.sh            # iter100000 e iter50000, con w_collision = -50 (choques) y 0 (referencia), 20 escenas
ESCENAS=50 CKPTS="iter100000.ckpt" WCOLS="-50 -20 0" sbatch evaluar.sh   # otras combinaciones
python3 resumir_evaluacion.py ../evaluaciones/ccdiff_5413 --filtro coll offroad
```
Cada escena tarda unos 80-100 s (20 replanificaciones × ~4 s). El resumen de todas las combinaciones sale al final del log.

### Resultado del primer entrenamiento (`ccdiff_5413`, 04/11/2026)
- 100 000 pasos en unas 9,5 h (2,6 pasos/s en la H200, batch 4), 5 épocas de 20 312 pasos.
- La pérdida de entrenamiento se estabiliza hacia los 10 000-20 000 pasos (unos 0,23, suavizada, al final).
- La pérdida de validación usa solo 5 lotes y oscila con un periodo de una época. Por eso **no sirve para elegir checkpoint**: se elige con `evaluar.sh`.

## Pendiente de verificar en el primer arranque real
- **Tiempo por paso en la H200.** No hay referencia publicada, así que el `entrenar_debug.sh` dará la primera medida. Con ella se decide si 24 h bastan o si hay que reanudar.
- **Batch.** El config de CCDiff usa `batch_size = 4` (comentado `# 100`). Con 141 GB en la H200 seguramente cabe bastante más (`--batch`). Si se cambia, conviene revisar el learning rate.
- **Caché de trajdata.** La primera vez trajdata procesa todo nuScenes. Se guarda en `datasets/trajdata_cache` y se reutiliza en los siguientes trabajos.
