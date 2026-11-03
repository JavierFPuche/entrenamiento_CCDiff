"""Lanza el entrenamiento de CCDiff dentro del contenedor, sin tocar su codigo.

Hace lo mismo que `bash scripts/run_train.sh` del repositorio, pero adapta la
configuracion al DGX antes de arrancar:

  - Rutas: CCDiff trae fijadas "/mnt/disks/sdb/nuscenes" y "~/.unified_data_cache"
    en varias configs (entrenamiento y rollouts). Aqui se sustituyen TODAS por
    --nuscenes y --cache (por defecto, las del proyecto montado como /workspace).
  - Registro: por defecto TensorBoard. Con --wandb y la variable WANDB_APIKEY, W&B.
  - Workers de datos segun SLURM_CPUS_PER_TASK, y numero de GPUs.
  - Nombre unico por trabajo (train.py pregunta por teclado si la carpeta de
    salida ya existe, y dentro de sbatch eso rompe el trabajo).
  - --reanudar <checkpoint.ckpt>: continua un entrenamiento cortado por el
    limite de tiempo de Slurm (train.py no lo permite por si solo).

Uso dentro del contenedor (lo llaman entrenar_debug.sh y entrenar.sh):
  python /workspace/scripts/lanzar_entrenamiento.py --debug
  python /workspace/scripts/lanzar_entrenamiento.py --gpus 1
  python /workspace/scripts/lanzar_entrenamiento.py --reanudar /workspace/entrenados/<nombre>/.../last.ckpt
"""
import argparse
import os
import runpy
import sys
import time
from pathlib import Path

CCDIFF = Path(os.environ.get("CCDIFF_SRC", "/opt/ccdiff/CCDiff"))
CONFIG = "trajdata_nusc_ccdiff"


def parchear(cfg, nusc, cache, cambios, ruta="cfg"):
    """Recorre la configuracion (Dict anidados) y sustituye rutas de datos y cache."""
    for clave in list(cfg.keys()):
        valor = cfg[clave]
        if clave == "trajdata_cache_location":
            cfg[clave] = str(cache)
            cambios.append(f"{ruta}.{clave}")
        elif clave == "trajdata_data_dirs" and isinstance(valor, dict):
            for k in list(valor.keys()):
                valor[k] = str(nusc)
            cambios.append(f"{ruta}.{clave}")
        elif isinstance(valor, dict):
            parchear(valor, nusc, cache, cambios, f"{ruta}.{clave}")


def main():
    ap = argparse.ArgumentParser(description="Entrenamiento de CCDiff en el DGX")
    ap.add_argument("--debug", action="store_true",
                    help="modo debug de CCDiff: validacion/guardado/rollout cada pocos pasos, sin logger")
    ap.add_argument("--nombre", default=None, help="nombre del experimento (por defecto ccdiff_<job>)")
    # Rutas dentro del contenedor = estructura del proyecto (montado como /workspace):
    #   datasets/nuscenes  datasets/trajdata_cache  entrenados/  pruebas/  entornos/home
    ap.add_argument("--nuscenes", default="/workspace/datasets/nuscenes", help="carpeta raiz de nuScenes")
    ap.add_argument("--cache", default="/workspace/datasets/trajdata_cache",
                    help="cache de trajdata (se crea la 1a vez: tarda y conviene conservarla)")
    ap.add_argument("--salida", default=None,
                    help="por defecto /workspace/entrenados (o /workspace/pruebas con --debug)")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--pasos", type=int, default=None, help="sustituye los 100k pasos por defecto")
    ap.add_argument("--batch", type=int, default=None, help="batch de entrenamiento (CCDiff: 4)")
    ap.add_argument("--workers", type=int, default=None,
                    help="workers de datos (por defecto SLURM_CPUS_PER_TASK - 2, maximo 16)")
    ap.add_argument("--wandb", action="store_true", help="usar W&B (necesita WANDB_APIKEY)")
    ap.add_argument("--reanudar", default=None, help="checkpoint .ckpt desde el que continuar")
    ap.add_argument("--comprobar", action="store_true",
                    help="muestra la configuracion final y sale sin entrenar")
    args = ap.parse_args()

    job = os.environ.get("SLURM_JOB_ID") or time.strftime("%Y%m%d_%H%M%S")
    nombre = args.nombre or f"ccdiff_{'debug_' if args.debug else ''}{job}"
    nusc, cache = Path(args.nuscenes), Path(args.cache)
    if args.salida is None:
        args.salida = "/workspace/pruebas" if args.debug else "/workspace/entrenados"
    if not (nusc / "v1.0-trainval").is_dir():
        msg = f"No encuentro {nusc}/v1.0-trainval: revisa el montaje de nuScenes (--nuscenes)."
        if not args.comprobar:
            sys.exit(msg)
        print("AVISO:", msg)
    cache.mkdir(parents=True, exist_ok=True)
    # HOME escribible y persistente para cachés de wandb, matplotlib, torch...
    os.environ["HOME"] = os.environ.get("CCDIFF_HOME", "/workspace/entornos/home")
    Path(os.environ["HOME"]).mkdir(parents=True, exist_ok=True)

    # --- Configuracion registrada: se modifica ANTES de que train.py la clone ---
    sys.path.insert(0, str(CCDIFF))
    from tbsim.configs.registry import EXP_CONFIG_REGISTRY

    cfg = EXP_CONFIG_REGISTRY[CONFIG]
    cambios = []
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    workers = args.workers if args.workers is not None else max(2, min(16, cpus - 2))
    usar_wandb = args.wandb and "WANDB_APIKEY" in os.environ
    if args.wandb and not usar_wandb:
        print("AVISO: --wandb sin WANDB_APIKEY -> se usa TensorBoard")
    with cfg.unlocked():
        parchear(cfg, nusc, cache, cambios)
        cfg.devices.num_gpus = args.gpus
        cfg.train.training.num_data_workers = workers
        cfg.train.validation.num_data_workers = max(1, workers // 2)
        cfg.train.logging.log_tb = not usar_wandb
        cfg.train.logging.log_wandb = usar_wandb
        if args.pasos:
            cfg.train.training.num_steps = args.pasos
        if args.batch:
            cfg.train.training.batch_size = args.batch

    print(f"[lanzador] experimento: {nombre}")
    print(f"[lanzador] nuScenes: {nusc} | cache: {cache} | salida: {args.salida}")
    print(f"[lanzador] GPUs: {args.gpus} | workers: {workers} | pasos: {cfg.train.training.num_steps} "
          f"| batch: {cfg.train.training.batch_size} | log: {'W&B' if usar_wandb else 'TensorBoard'}")
    print(f"[lanzador] rutas sustituidas en: {', '.join(cambios)}")
    if args.comprobar:
        from tbsim.configs.registry import get_registered_experiment_config
        final = get_registered_experiment_config(CONFIG)   # lo que recibira train.py
        print("[lanzador] train.trajdata_data_dirs =", dict(final.train.trajdata_data_dirs))
        print("[lanzador] train.trajdata_cache_location =", final.train.trajdata_cache_location)
        print("[lanzador] eval.nusc.trajdata_data_dirs =", dict(final.eval.nusc.trajdata_data_dirs))
        print("[lanzador] eval.trajdata.trajdata_cache_location =",
              final.eval.trajdata.trajdata_cache_location)
        return

    # --- Reanudar: Trainer.fit con ckpt_path --------------------------------
    if args.reanudar:
        import pytorch_lightning as pl
        ckpt = str(Path(args.reanudar).resolve())
        if not Path(ckpt).is_file():
            sys.exit(f"No existe el checkpoint {ckpt}")
        fit_original = pl.Trainer.fit

        def fit(self, *a, **k):
            k.setdefault("ckpt_path", ckpt)
            print(f"[lanzador] reanudando desde {ckpt}")
            return fit_original(self, *a, **k)
        pl.Trainer.fit = fit

    # --- train.py tal cual -------------------------------------------------
    argv = ["train.py", "--config_name", CONFIG, "--name", nombre,
            "--dataset_path", str(nusc), "--output_dir", args.salida]
    if args.debug:
        argv += ["--debug", "--remove_exp_dir"]
    if usar_wandb:
        argv += ["--wandb_project_name", "ccdiff"]
    os.chdir(CCDIFF)
    sys.argv = argv
    runpy.run_path(str(CCDIFF / "ccdiff" / "examples" / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
