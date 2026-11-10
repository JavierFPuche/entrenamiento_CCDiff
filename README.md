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

**Fallo de CCDiff corregido en `lanzar_evaluacion.py`.** Con `--part_control`, `PartialPerturbationGuidance.perturb` crea el optimizador sobre `x[:, idx, :ts]`. Con `idx` como lista, eso es una copia, así que el guiado nunca modifica la trayectoria. La primera evaluación (job 5435) dio resultados idénticos con `w_collision` −50 y 0 por este motivo.
- `lanzar_evaluacion.py` optimiza el tensor completo y anula el gradiente fuera de los agentes y pasos controlados.
- Al final del log aparece `pasos de guiado: N, con cambio en la trayectoria: M`, y M debe ser mayor que 0.
- `--sin_parche_guiado` reproduce el comportamiento original.
- **Segunda corrección (09/11).** Las pérdidas de guiado dan gradientes NaN en algunos elementos. La primera versión del parche los aplicaba y llenaba de NaN las trayectorias de todos los agentes, también con peso 0. Por eso **los jobs 5437 y 5448 no son válidos**: con `w_collision=0 w_offroad=0` daban ADE 0,44 frente a 1,30 sin parche. Ahora los gradientes no finitos se anulan y nunca se introduce un NaN en la trayectoria.
- El log indica cuántos elementos se han anulado. Comprobación de control: con `WCOLS="0" WOFFS="0"` el resultado tiene que coincidir con `--sin_parche_guiado`.
- Control verificado el 10/11 (job 5478, 30 escenas): `wcol0_woff0` y `wcol0_woff0_sinparche` dan resultados idénticos.

**Intensidad del guiado.** En cada paso de difusión se crea un Adam nuevo y se da 1 paso, así que el primer paso de Adam vale `lr · signo(gradiente)`. El tamaño del empujón lo marca `lr` (0,3 por defecto) y no `w_collision`; el peso solo reparte la dirección entre colisión y offroad. Por eso −50 y −200 dan casi lo mismo.

Resultados con iter50000 y 30 escenas:

| configuración | colisiones | ADE |
| --- | --- | --- |
| sin guiado | 6,7 % | 1,29 |
| `w_collision` −50 | 8,9 % | 1,45 |
| `w_collision` −200 | 9,1 % | 1,46 |

Para guiar con más fuerza hay que subir `--guia_lr` y/o `--guia_pasos`, controlar más agentes con `--controlables`, o ampliar el horizonte (`--n_step_action`). En `evaluar.sh` se hace con `LRS="0.3 1 3"` y `EXTRA="..."`.

### Resultado del primer entrenamiento (`ccdiff_5413`, 04/11/2026)
- 100 000 pasos en unas 9,5 h (2,6 pasos/s en la H200, batch 4), 5 épocas de 20 312 pasos.
- La pérdida de entrenamiento se estabiliza hacia los 10 000-20 000 pasos (unos 0,23, suavizada, al final).
- La pérdida de validación usa solo 5 lotes y oscila con un periodo de una época. Por eso **no sirve para elegir checkpoint**: se elige con `evaluar.sh`.

## Pendiente de verificar en el primer arranque real
- **Tiempo por paso en la H200.** No hay referencia publicada, así que el `entrenar_debug.sh` dará la primera medida. Con ella se decide si 24 h bastan o si hay que reanudar.
- **Batch.** El config de CCDiff usa `batch_size = 4` (comentado `# 100`). Con 141 GB en la H200 seguramente cabe bastante más (`--batch`). Si se cambia, conviene revisar el learning rate.
- **Caché de trajdata.** La primera vez trajdata procesa todo nuScenes. Se guarda en `datasets/trajdata_cache` y se reutiliza en los siguientes trabajos.

### Choques y escenas candidatas para BeamNG

`analizar_choques.py` lee los `data.hdf5` de un grupo de evaluaciones y busca los choques entre agentes. Para cada pareja usa cajas orientadas con centroide, rumbo y tamaño, y se queda con el primer contacto. Cada choque se clasifica según:
- si interviene algún agente controlado por CCDiff (los índices se sacan del log del job);
- si el solape ya existía en el primer paso (artefacto de los datos);
- la velocidad relativa y el tipo de impacto;
- si los dos coches estaban sobre la calzada (`drivable_area` del mapa de nuScenes).

Una escena es **candidata** si tiene algún choque con un agente controlado que no existía al inicio, con los dos coches sobre la calzada y una velocidad relativa de al menos `--v_min` (2 m/s por defecto).

```bash
GRUPO=ctrl_30esc JOBS="5483" sbatch analizar.sh
```

El resultado se guarda en `evaluaciones/<GRUPO>/choques.csv` (todos los choques) y `candidatas.json` (el mejor choque de cada escena).

tbsim rellena con (0, 0) los pasos en que un agente aún no ha entrado o ya ha salido de la escena; el análisis los descarta. La primera versión no lo hacía y daba miles de choques falsos en el origen.

Para revisar las candidatas **no hay que volver a simular**: la difusión es aleatoria y con otra lista de escenas saldrían otras trayectorias. `dibujar_choques.py` dibuja como GIF lo que ya está guardado en `data.hdf5`, sobre el mapa de nuScenes. Los agentes controlados salen en rojo, los demás en gris, y la pareja que choca con borde negro desde el impacto.

```bash
GRUPO=ctrl_30esc sbatch dibujar.sh      # -> evaluaciones/ctrl_30esc/gifs/
```

Resultados del 10/11 (`ctrl_30esc`, iter50000, 10 agentes controlados, lr 1, 30 escenas de Boston):

| configuración | escenas candidatas | choques válidos | velocidad relativa mediana |
| --- | --- | --- | --- |
| sin guiado de colisión | 9 | 24 | 4,7 m/s |
| `w_collision` −50 | 18 | 53 | 7,7 m/s |

Con guiado, 32 de los 48 choques candidatos son alcances. Hay 11 choques en el primer 1,5 s que hay que revisar a ojo.

**Filtro de plausibilidad (11/11).** Al revisar los GIF salieron cuatro tipos de incoherencia:
- choques en el primer segundo entre coches que ya nacen pegados en nuScenes (cola de semáforo);
- agentes controlados que giran sobre sí mismos o se desplazan de lado;
- autobuses y camiones, que en nuScenes a veces son dos cajas solapadas (articulados);
- después del impacto los coches se atraviesan, porque tbsim no tiene física.

Ahora `analizar_choques.py` solo da por candidato un choque que, hasta el impacto, cumple todo esto:
- ocurre a partir de 1 s y los coches empiezan separados al menos 2 m;
- el giro no pasa de 1 rad/s, la aceleración o frenada media en 0,3 s no pasa de 6 m/s² y el derrape no pasa de 35°, todo con el coche a 3 m/s o más;
- los dos vehículos miden como mucho 7 m.

El motivo de cada descarte queda en la columna `motivos` de `choques.csv`, y los umbrales están al principio del archivo. `--sin_filtro` vuelve al criterio anterior. Lo que pasa después del impacto no se filtra, porque en BeamNG lo resuelve la física; por eso los GIF se cortan 2 s después (`--completo` para verlos enteros).
