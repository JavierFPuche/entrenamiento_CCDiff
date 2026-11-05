"""Evalua un checkpoint de CCDiff en bucle cerrado (sin BeamNG), dentro del contenedor.

Equivale a scripts/run_eval.sh de CCDiff: ejecuta ccdiff/examples/scene_editor.py
sobre escenas de validacion de nuScenes. Los agentes se simulan con el motor
cinematico de tbsim (como en los rollouts del entrenamiento), no con BeamNG.

  --w_collision -50  guiado HACIA la colision (generar accidentes, como en el paper)
  --w_collision 0    sin guiado de colision: trafico "normal" de referencia

Reutiliza los arreglos de lanzar_entrenamiento.py: rutas de nuScenes/cache,
~/.unified_data_cache, os.cpu_count(), Lightning sin manejador de Slurm y el
parche de deepcopy de torchvision (al cargar el checkpoint se crea la copia EMA).

Salida en --salida (por defecto /workspace/evaluaciones/<entrenamiento>/<ckpt>_wcol<peso>/scene_edit_eval/):
  stats.json (metricas por escena), config.json, viz/ (videos), data.hdf5

Uso (lo llama evaluar.sh):
  python /workspace/scripts/lanzar_evaluacion.py --ckpt iter100000.ckpt --w_collision -50
  python /workspace/scripts/lanzar_evaluacion.py --comprobar
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import lanzar_entrenamiento as le  # noqa: E402  (mismos arreglos que el entrenamiento)

REGISTRO = "trajdata_nusc_ccdiff"


def main():
    ap = argparse.ArgumentParser(description="Evaluacion de CCDiff con scene_editor (sin BeamNG)")
    ap.add_argument("--ckpt_dir", default="/workspace/entrenados/ccdiff_5413/run0",
                    help="carpeta del entrenamiento (con config.json y checkpoints/)")
    ap.add_argument("--ckpt", default="iter100000.ckpt",
                    help="texto UNICO del nombre del checkpoint, p. ej. iter100000.ckpt")
    ap.add_argument("--w_collision", type=float, default=-50.0,
                    help="peso del guiado de colision (negativo = provocar choques; 0 = sin guiado)")
    ap.add_argument("--w_offroad", type=float, default=1.0)
    ap.add_argument("--controlables", type=int, default=5,
                    help="agentes que controla CCDiff (los demas reproducen nuScenes)")
    ap.add_argument("--escenas", type=int, default=20,
                    help="numero de escenas de validacion (CCDiff usa 100; ~80 s por escena)")
    ap.add_argument("--n_step_action", type=int, default=5, help="replanificar cada N pasos (0,1 s)")
    ap.add_argument("--sin_video", action="store_true", help="no renderizar videos (mas rapido)")
    ap.add_argument("--nuscenes", default="/workspace/datasets/nuscenes")
    ap.add_argument("--cache", default="/workspace/datasets/trajdata_cache")
    ap.add_argument("--salida", default=None)
    ap.add_argument("--comprobar", action="store_true", help="muestra la configuracion y sale")
    args = ap.parse_args()

    n_cpus, cpus_nodo = le.ajustar_slurm_y_cpus()
    os.environ["HOME"] = os.environ.get("CCDIFF_HOME", "/workspace/entornos/home")
    Path(os.environ["HOME"]).mkdir(parents=True, exist_ok=True)
    nusc, cache = Path(args.nuscenes), Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    le.enlazar_cache_home(cache)

    peso = f"{args.w_collision:g}".replace("-", "m")
    nombre = f"{Path(args.ckpt).stem}_wcol{peso}"
    salida = Path(args.salida or f"/workspace/evaluaciones/{Path(args.ckpt_dir).parent.name}/{nombre}")

    print(f"[evaluacion] checkpoint: {args.ckpt_dir} :: {args.ckpt}")
    print(f"[evaluacion] w_collision={args.w_collision} w_offroad={args.w_offroad} "
          f"controlables={args.controlables} escenas={args.escenas} CPUs={n_cpus}/{cpus_nodo}")
    print(f"[evaluacion] salida: {salida}")

    # --- Configuracion: SceneEditingConfig trae rutas fijas y 100 escenas ----
    sys.path.insert(0, str(le.CCDIFF))
    from tbsim.configs.scene_edit_config import SceneEditingConfig

    init_original = SceneEditingConfig.__init__
    escenas = list(range(args.escenas))

    def init(self, *a, **k):
        init_original(self, *a, **k)
        cambios = []
        # scene_editor.py sigue modificando la configuracion despues: hay que
        # devolverla con el MISMO estado de bloqueo con el que se creo.
        try:
            bloqueada = object.__getattribute__(self, "__frozen")
        except AttributeError:
            bloqueada = False
        self.unlock()
        le.parchear(self, nusc, cache, cambios)
        self.trajdata.num_scenes_to_evaluate = len(escenas)
        self.trajdata.eval_scenes = escenas
        self.lock(bloqueada)
        print(f"[evaluacion] rutas sustituidas en: {', '.join(cambios)}")
    SceneEditingConfig.__init__ = init

    if args.comprobar:
        cfg = SceneEditingConfig(registered_name=REGISTRO)
        print("[evaluacion] trajdata:", dict(cfg.trajdata.trajdata_data_dirs),
              cfg.trajdata.trajdata_cache_location, "| escenas:", cfg.trajdata.eval_scenes)
        return

    if not (nusc / "v1.0-trainval").is_dir():
        sys.exit(f"No encuentro {nusc}/v1.0-trainval")
    if not list(Path(args.ckpt_dir).glob(f"**/*{args.ckpt}*")):
        sys.exit(f"No hay ningun checkpoint que contenga '{args.ckpt}' en {args.ckpt_dir}")

    le.parchear_deepcopy_feature_extractor()

    argv = ["scene_editor.py",
            "--results_root_dir", str(salida), "--num_scenes_per_batch", "1",
            "--dataset_path", str(nusc), "--env", "trajdata",
            "--policy_ckpt_dir", args.ckpt_dir, "--policy_ckpt_key", args.ckpt,
            "--eval_class", "CCDiff", "--editing_source", "config", "heuristic",
            "--registered_name", REGISTRO,
            "--part_control", "--controllable_agent", str(args.controlables),
            "--w_collision", str(args.w_collision), "--w_offroad", str(args.w_offroad),
            "--n_step_action", str(args.n_step_action), "--save_every_n_frames", "5"]
    if not args.sin_video:
        argv.append("--render")
    os.chdir(le.CCDIFF)
    sys.argv = argv
    runpy.run_path(str(le.CCDIFF / "ccdiff" / "examples" / "scene_editor.py"), run_name="__main__")


if __name__ == "__main__":
    main()
