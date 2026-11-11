"""Evalua un checkpoint de CCDiff en bucle cerrado (sin BeamNG), dentro del contenedor.

Equivale a scripts/run_eval.sh de CCDiff: ejecuta ccdiff/examples/scene_editor.py
sobre escenas de validacion de nuScenes. Los agentes se simulan con el motor
cinematico de tbsim (como en los rollouts del entrenamiento), no con BeamNG.

  --w_collision -50  guiado HACIA la colision (generar accidentes, como en el paper)
  --w_collision 0    sin guiado de colision: trafico "normal" de referencia

Corrige ademas un fallo de CCDiff: con --part_control el guiado (w_collision,
w_offroad) no modificaba la trayectoria (ver parchear_guiado_parcial).

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

# Veces que el guiado ha movido de verdad la trayectoria (para comprobarlo en el log)
CONTADOR_GUIADO = {"llamadas": 0, "con_cambio": 0, "grad_no_finito": 0}


def parchear_guiado_parcial():
    """Arregla PartialPerturbationGuidance.perturb de CCDiff (--part_control).

    El original crea el optimizador sobre x[:, idx, :ts] con idx una LISTA:
    eso es indexacion avanzada, que devuelve una COPIA. Dentro de p_sample_loop
    (torch.no_grad) esa copia es una hoja sin gradiente, Adam la acepta, nunca
    recibe .grad y opt.step() no hace nada: el guiado no cambia la trayectoria
    y w_collision no tiene ningun efecto.

    Aqui se optimiza el tensor completo y se anula el gradiente fuera de los
    agentes controlados y de los primeros `controllable_ts` pasos, que es lo que
    pretendia el codigo original.
    """
    import torch
    from ccdiff.utils import guidance_loss as gl

    def perturb(self, x_initial, data_batch, opt_params, num_samp=1, decoder=None, return_grad_of=None):
        assert self.current_guidance is not None, "Must instantiate guidance object before calling"
        perturb_th = opt_params["perturb_th"]
        x_guidance = x_initial
        if not x_guidance.requires_grad:
            x_guidance.requires_grad_()
        hoja = x_guidance  # tensor hoja que optimiza Adam (en 2D x_guidance pasa a ser una vista)
        if len(x_guidance.shape) == 2:
            NM, _ = x_guidance.shape
            x_guidance_reshaped = x_guidance.clone()
            x_guidance = x_guidance.reshape(int(NM // num_samp), num_samp, -1)
        elif len(x_guidance.shape) == 4:
            with torch.enable_grad():
                BN, M, T, _ = x_guidance.shape
                B = int(BN // num_samp)
                x_guidance_reshaped = x_guidance.reshape(B, num_samp, M, T, -1).permute(0, 2, 1, 3, 4).reshape(B * M * num_samp, T, -1)
        else:
            x_guidance_reshaped = x_guidance

        mascara = None
        if self.available_idx is not None and hoja.dim() in (3, 4):
            idx = list(self.available_idx[: x_guidance.shape[-3]])
            ts = getattr(self, "controllable_ts", None)
            mascara = torch.zeros_like(hoja, dtype=torch.bool)
            if hoja.dim() == 3:
                mascara[idx, :ts] = True
            else:
                mascara[:, idx, :ts] = True
        clase = torch.optim.Adam if opt_params["optimizer"] == "adam" else torch.optim.SGD
        opt = clase([hoja], lr=opt_params["lr"])

        per_losses = dict()
        for _ in range(opt_params["grad_steps"]):
            with torch.enable_grad():
                x_dec = decoder(x_guidance_reshaped) if decoder is not None else x_guidance_reshaped
                bsize = int(x_dec.size(0) / num_samp)
                x_all = self.transform(x_dec, data_batch, self.transform_params, bsize=bsize, num_samp=num_samp)
                tot_loss, per_losses = self.current_guidance.compute_guidance_loss(
                    x_all.reshape((bsize, num_samp, -1, 6)), data_batch)
            antes = hoja.detach().clone()
            tot_loss.backward()
            if hoja.grad is not None:
                # Las perdidas de guiado dan gradientes NaN/inf en algunos elementos
                # (agentes de relleno, distancias nulas...). Con el original no
                # importaba porque nunca se aplicaban; aqui se anulan, igual que el
                # gradiente fuera de los agentes/pasos controlados (0*NaN = NaN, por
                # eso se usa torch.where y no una multiplicacion).
                g = hoja.grad
                finito = torch.isfinite(g)
                CONTADOR_GUIADO["grad_no_finito"] += int((~finito).sum())
                valido = finito if mascara is None else (finito & mascara)
                hoja.grad = torch.where(valido, g, torch.zeros_like(g))
            opt.step()
            opt.zero_grad()
            with torch.no_grad():
                if perturb_th is not None:
                    delta = torch.clip(hoja - x_initial, -1 * perturb_th, perturb_th)
                    hoja.data = x_initial + delta
                # Nunca introducir NaN/inf donde antes habia un valor valido
                malo = ~torch.isfinite(hoja) & torch.isfinite(antes)
                if malo.any():
                    hoja.data[malo] = antes[malo]
            CONTADOR_GUIADO["llamadas"] += 1
            ahora = hoja.detach()
            distinto = (ahora != antes) & ~(torch.isnan(ahora) & torch.isnan(antes))
            if distinto.any():
                CONTADOR_GUIADO["con_cambio"] += 1
                if CONTADOR_GUIADO["con_cambio"] == 1:
                    print("[evaluacion] guiado activo: la trayectoria se modifica (primer paso con cambio)")
        return x_guidance, per_losses

    gl.PartialPerturbationGuidance.perturb = perturb


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
    ap.add_argument("--escenas_idx", default=None,
                    help="lista de indices de escena separados por comas (anula --escenas), p. ej. 3,7,12; "
                         "los da analizar_choques.py para volver a simular con video solo las candidatas")
    ap.add_argument("--n_step_action", type=int, default=5, help="replanificar cada N pasos (0,1 s)")
    ap.add_argument("--sin_video", action="store_true", help="no renderizar videos (mas rapido)")
    ap.add_argument("--nuscenes", default="/workspace/datasets/nuscenes")
    ap.add_argument("--cache", default="/workspace/datasets/trajdata_cache")
    ap.add_argument("--salida", default=None, help="carpeta exacta de salida (anula --grupo)")
    ap.add_argument("--grupo", default=None,
                    help="subcarpeta de evaluaciones/ (por defecto el nombre del entrenamiento)")
    ap.add_argument("--comprobar", action="store_true", help="muestra la configuracion y sale")
    ap.add_argument("--guia_lr", type=float, default=None,
                    help="lr del optimizador de guiado (CCDiff: 0.3). Con Adam y 1 paso, el tamano del "
                         "empujon depende del lr, NO de w_collision (el peso solo reparte entre perdidas)")
    ap.add_argument("--guia_pasos", type=int, default=None,
                    help="pasos de gradiente del guiado en cada paso de difusion (CCDiff: 1)")
    ap.add_argument("--sin_parche_guiado", action="store_true",
                    help="usar el perturb ORIGINAL de CCDiff (con --part_control el guiado no hace nada)")
    args = ap.parse_args()

    n_cpus, cpus_nodo = le.ajustar_slurm_y_cpus()
    os.environ["HOME"] = os.environ.get("CCDIFF_HOME", "/workspace/entornos/home")
    Path(os.environ["HOME"]).mkdir(parents=True, exist_ok=True)
    nusc, cache = Path(args.nuscenes), Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    le.enlazar_cache_home(cache)

    def txt(v):
        return f"{v:g}".replace("-", "m")
    nombre = f"{Path(args.ckpt).stem}_wcol{txt(args.w_collision)}"
    if args.w_offroad != 1.0:
        nombre += f"_woff{txt(args.w_offroad)}"
    if args.guia_lr is not None:
        nombre += f"_lr{txt(args.guia_lr)}"
    if args.guia_pasos is not None:
        nombre += f"_gs{args.guia_pasos}"
    if args.controlables != 5:
        nombre += f"_ctrl{args.controlables}"
    if args.sin_parche_guiado:
        nombre += "_sinparche"
    grupo = args.grupo or Path(args.ckpt_dir).parent.name
    salida = Path(args.salida or f"/workspace/evaluaciones/{grupo}/{nombre}")

    print(f"[evaluacion] checkpoint: {args.ckpt_dir} :: {args.ckpt}")
    print(f"[evaluacion] w_collision={args.w_collision} w_offroad={args.w_offroad} "
          f"controlables={args.controlables} escenas={args.escenas_idx or args.escenas} CPUs={n_cpus}/{cpus_nodo}")
    print(f"[evaluacion] salida: {salida}")

    # --- Configuracion: SceneEditingConfig trae rutas fijas y 100 escenas ----
    sys.path.insert(0, str(le.CCDIFF))
    from tbsim.configs.scene_edit_config import SceneEditingConfig

    init_original = SceneEditingConfig.__init__
    if args.escenas_idx:
        escenas = [int(x) for x in args.escenas_idx.split(",") if x.strip()]
    else:
        escenas = list(range(args.escenas))
    # scene_editor.py toma los agentes controlables de un archivo fijo del paper
    # (data/0_scene_idx.pkl + data/10_control_idx_test_init.pkl) que solo cubre las
    # primeras 100 escenas de validacion: con mas, falla con IndexError en la 101.
    try:
        import joblib
        n_max = len(joblib.load(str(le.CCDIFF / "data" / "0_scene_idx.pkl")))
    except Exception:  # noqa: BLE001
        n_max = 100
    fuera = [i for i in escenas if i >= n_max]
    if fuera:
        print(f"[evaluacion] AVISO: CCDiff solo tiene agentes controlables para {n_max} escenas; "
              f"se quitan los indices {fuera[0]}..{fuera[-1]}")
        escenas = [i for i in escenas if i < n_max]

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
        if args.guia_lr is not None:
            self.guidance_optimization_params["lr"] = args.guia_lr
        if args.guia_pasos is not None:
            self.guidance_optimization_params["grad_steps"] = args.guia_pasos
        self.lock(bloqueada)
        print(f"[evaluacion] rutas sustituidas en: {', '.join(cambios)}")
        print(f"[evaluacion] optimizador de guiado: {dict(self.guidance_optimization_params)}")
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
    if not args.sin_parche_guiado:
        parchear_guiado_parcial()
        print("[evaluacion] parche de guiado parcial aplicado")

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
    try:
        runpy.run_path(str(le.CCDIFF / "ccdiff" / "examples" / "scene_editor.py"), run_name="__main__")
    finally:
        if not args.sin_parche_guiado:
            c = CONTADOR_GUIADO
            print(f"[evaluacion] pasos de guiado: {c['llamadas']}, con cambio en la trayectoria: {c['con_cambio']}, "
                  f"elementos de gradiente NaN/inf anulados: {c['grad_no_finito']}")


if __name__ == "__main__":
    main()
