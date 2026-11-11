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


def parchear_objetivo_ego(semilla=0, sector_grados=100.0, radio_busqueda=60.0, n_candidatos=3):
    """Accidentes CON EL EGO sin que el ego "sepa" nada (--objetivo_ego).

    - El ego NO se controla: reproduce su trayectoria real de nuScenes (no anticipa nada).
    - Se elige UN adversario por escena: un agente en movimiento, a menos de
      `radio_busqueda` m y dentro de +-`sector_grados` del frente del ego (no por detras:
      ese golpe no se veria en la camara frontal), de <= 7 m, con >= 2 m de hueco con
      cualquier otro coche y que a velocidad constante ya se acerque al ego en la ventana
      del clip (puntuacion = distancia minima + penalizacion si no cae en 2-5 s). Se sortea
      entre los `n_candidatos` mejor puntuados (semilla fija por escena) para dar variedad.
    - Agentes controlados: el adversario primero y despues los mas cercanos al ego
      (sin el ego). CCDiff los tomaba de un archivo fijo que solo cubre 100 escenas.
    - Guiado nuevo "intercepta_ego" (largo alcance): solo para el adversario, minimiza la
      distancia minima entre su plan y el del ego en el horizonte. La perdida de colision
      original solo actua a <~0,2 m de contacto, asi que por si sola no lleva a nadie al ego.
    - Guiado nuevo "evita_otros": el adversario esquiva a todos los demas agentes; sin el,
      en la prueba del 11/11 el adversario chocaba con otros coches de camino al ego.
    - La perdida de colision original se limita a los pares con el ego (excluded_agents).
    En el log queda "[evaluacion] ego: <i> adversario: <j>" para analizar_choques.py.
    """
    import joblib
    import numpy as np
    import torch
    from tbsim.envs import env_trajdata as et
    from tbsim.utils import scene_edit_utils as seu
    from tbsim.utils import guidance_loss as gl

    estado = {"env": None, "ego": None, "adversario": None}
    T_ALCANCE, PUNTOS_MAX = 6.0, 20.0

    def holgura_inicial(c, yaw, ext, ie):
        """Hueco minimo (m) entre cada agente y cualquier otro que no sea el ego, con cajas
        orientadas. Un adversario pegado a otro coche choca con el antes de 2 s
        (escena 0093 del 11/11: el adversario empezaba rozando un autobus)."""
        n = len(c)
        try:
            from shapely.geometry import Polygon as P
        except ImportError:
            dd = np.linalg.norm(c[:, None] - c[None], axis=-1) - ext[:, None, 0] / 2 - ext[None, :, 0] / 2
            np.fill_diagonal(dd, np.inf)
            dd[:, ie] = np.inf
            return np.where(np.isfinite(c).all(-1), dd.min(1), -1.0)
        polis = []
        for i in range(n):
            if not np.isfinite(c[i]).all() or (c[i] == 0).all():
                polis.append(None)
                continue
            l, w = ext[i, 0] / 2, ext[i, 1] / 2
            co, si = np.cos(yaw[i]), np.sin(yaw[i])
            esq = [(c[i, 0] + co * x - si * y, c[i, 1] + si * x + co * y)
                   for x, y in ((l, w), (l, -w), (-l, -w), (-l, w))]
            polis.append(P(esq))
        h = np.full(n, np.inf)
        for i in range(n):
            if polis[i] is None:
                h[i] = -1.0
                continue
            for j in range(n):
                if j in (i, ie) or polis[j] is None:
                    continue
                if np.linalg.norm(c[i] - c[j]) < 15.0:
                    h[i] = min(h[i], polis[i].distance(polis[j]))
        return h

    reset_original = et.EnvUnifiedSimulation.reset

    def reset(self, *a, **k):
        res = reset_original(self, *a, **k)
        estado["env"] = self
        return res
    et.EnvUnifiedSimulation.reset = reset

    class Identidad(dict):
        def __getitem__(self, clave):
            return clave

        def __len__(self):
            return 10 ** 6

    class AdversarioYCercanos(dict):
        def __getitem__(self, clave):
            env = estado["env"]
            nombres = list(env.current_agent_names)
            estado["ego"] = estado["adversario"] = None
            if "ego" not in nombres:
                print("[evaluacion] AVISO: escena sin agente 'ego'; se controlan los primeros")
                return list(range(len(nombres)))
            ie = nombres.index("ego")
            obs = env.get_observation()["agents"]
            c = np.asarray(obs["centroid"], dtype=float)
            yaw = np.asarray(obs["yaw"], dtype=float).reshape(-1)
            vel = np.abs(np.asarray(obs.get("curr_speed", np.ones(len(c))), dtype=float).reshape(-1))
            ext = np.asarray(obs.get("extent", np.tile([4.5, 1.9, 1.5], (len(c), 1))), dtype=float)
            d = np.linalg.norm(c - c[ie], axis=-1)
            rumbo = np.degrees(np.arctan2(c[:, 1] - c[ie, 1], c[:, 0] - c[ie, 0]) - yaw[ie])
            rumbo = (rumbo + 180) % 360 - 180
            orden = [int(i) for i in np.argsort(d) if i != ie and np.isfinite(d[i])]
            # Alcance a velocidad constante: distancia minima con el ego en los proximos
            # T_ALCANCE s y en que instante. Un adversario que ya "va hacia" el ego necesita
            # poco empujon (el guiado solo da lr*signo por paso) y choca dentro de la ventana.
            ts = np.arange(0.0, T_ALCANCE + 1e-6, 0.1)
            v = vel[:, None] * np.stack([np.cos(yaw), np.sin(yaw)], -1)                    # (N, 2)
            rel = (c - c[ie])[:, None, :] + (v - v[ie])[:, None, :] * ts[None, :, None]    # (N, T, 2)
            dist_t = np.linalg.norm(rel, axis=-1)
            k = np.argmin(dist_t, axis=1)
            dmin_cv, tmin_cv = dist_t[np.arange(len(c)), k], ts[k]
            # puntuacion: cerca a velocidad constante y con la maxima aproximacion en la
            # ventana del clip (2-5 s); fuera de ella se penaliza
            puntos = dmin_cv + 3.0 * np.maximum(0, 2.0 - tmin_cv) + 3.0 * np.maximum(0, tmin_cv - 5.0)
            hueco = holgura_inicial(c, yaw, ext, ie)
            razones = {}

            def apto(i):
                for ok, r in ((d[i] <= radio_busqueda, "lejos"), (abs(rumbo[i]) <= sector_grados, "detras"),
                              (vel[i] > 1.0, "parado"), (ext[i, 0] <= 7.0, "grande"),
                              (hueco[i] >= 2.0, "pegado"), (puntos[i] <= PUNTOS_MAX, "inalcanzable")):
                    if not ok:
                        razones[r] = razones.get(r, 0) + 1
                        return False
                return True
            validos = sorted((i for i in orden if apto(i)), key=lambda i: puntos[i])
            adv = None
            if validos:
                rng = np.random.default_rng(semilla * 100003 + int(clave))
                adv = int(rng.choice(validos[:n_candidatos]))
                orden = [adv] + [i for i in orden if i != adv]
            estado["ego"], estado["adversario"] = ie, adv
            extra = (f" (d={d[adv]:.0f} m, dmin_cv={dmin_cv[adv]:.1f} m en t={tmin_cv[adv]:.1f} s, "
                     f"{len(validos)} aptos)" if adv is not None else f" (descartes: {razones})")
            print(f"[evaluacion] ego: {ie} adversario: {adv if adv is not None else '-'}{extra}")
            return orden

    load_original = joblib.load

    def load(f, *a, **k):
        nombre = str(f)
        if nombre.endswith("10_control_idx_test_init.pkl"):
            return AdversarioYCercanos()
        if nombre.endswith("0_scene_idx.pkl"):
            return Identidad()
        return load_original(f, *a, **k)
    joblib.load = load

    # --- perdida de colision original: solo pares con el ego ---------------------
    colision_original = seu.HEURISTIC_FUNC["agent_collision"]

    def colision_con_ego(sim_scene, dt, num_disks, buffer_dist, decay_rate, excluded_agents, **kw):
        cfg = colision_original(sim_scene, dt, num_disks, buffer_dist, decay_rate, excluded_agents, **kw)
        nombres = [a.name for a in sim_scene.agents]
        if "ego" in nombres:
            cfg["params"]["excluded_agents"] = [i for i, n in enumerate(nombres) if n != "ego"]
        return cfg
    seu.HEURISTIC_FUNC["agent_collision"] = colision_con_ego

    # --- guiado nuevo: el adversario intercepta al ego ----------------------------
    class InterceptaEgoLoss(gl.GuidanceLoss):
        """Distancia minima entre el plan del adversario y el del ego (en el horizonte).

        Minimo exacto (el gradiente va al instante de maxima aproximacion); el minimo
        "suave" con logsumexp lo sesgaba varios metros a la baja y anulaba la perdida.
        Por debajo de `radio` (centros a ~1 m: ya hay contacto) no empuja mas: el ultimo
        tramo lo hace la perdida de colision original.
        """
        def __init__(self, ego, adversario, radio=1.0):
            super().__init__()
            self.ego, self.adversario = int(ego), int(adversario)
            self.radio = radio

        def forward(self, x, data_batch, agt_mask=None):
            pos_w, _ = gl.transform_agents_to_world(x[..., :2], x[..., 3:4], data_batch["world_from_agent"])
            B = pos_w.size(0)
            ego_traj = pos_w[self.ego].detach()                              # (N, T, 2)
            dist = torch.norm(pos_w - ego_traj.unsqueeze(0), dim=-1)          # (B, N, T)
            dmin = dist.min(dim=-1).values
            perdida = torch.relu(dmin - self.radio)                           # (B, N)
            solo_adv = torch.zeros(B, 1, device=perdida.device, dtype=perdida.dtype)
            if 0 <= self.adversario < B:
                solo_adv[self.adversario] = 1.0
            perdida = perdida * solo_adv
            return perdida[agt_mask] if agt_mask is not None else perdida
    gl.GUIDANCE_FUNC_MAP["intercepta_ego"] = InterceptaEgoLoss

    class EvitaOtrosLoss(gl.GuidanceLoss):
        """El adversario no debe chocar con nadie MAS que con el ego.

        Cada coche se aproxima con 3 discos a lo largo (radio = medio ancho). Penaliza
        la invasion de la distancia de seguridad entre los discos del adversario y los
        de cualquier otro agente que no sea el ego. Solo actua de cerca, como la perdida
        de colision original, pero con peso positivo (repulsion).
        """
        def __init__(self, ego, adversario, margen=0.5, discos=3):
            super().__init__()
            self.ego, self.adversario = int(ego), int(adversario)
            self.margen, self.discos = margen, discos

        def _discos(self, pos, yaw, extent):
            largo, ancho = extent[:, 0], extent[:, 1]
            r = ancho / 2
            mitad = torch.clamp(largo / 2 - r, min=0.0)
            s = torch.linspace(-1, 1, self.discos, device=pos.device)              # (D,)
            off = mitad[:, None] * s[None]                                         # (B, D)
            u = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)              # (B, N, T, 2)
            return pos.unsqueeze(-2) + off[:, None, None, :, None] * u.unsqueeze(-2), r   # (B,N,T,D,2)

        def forward(self, x, data_batch, agt_mask=None):
            pos_w, yaw_w = gl.transform_agents_to_world(x[..., :2], x[..., 3:4], data_batch["world_from_agent"])
            B = pos_w.size(0)
            perdida = torch.zeros(B, pos_w.size(1), device=pos_w.device, dtype=pos_w.dtype)
            a = self.adversario
            if 0 <= a < B:
                discos, r = self._discos(pos_w, yaw_w[..., 0], data_batch["extent"])
                otros = [k for k in range(B) if k not in (a, self.ego)]
                if otros:
                    da = discos[a].unsqueeze(0)                                    # (1,N,T,D,2)
                    do = discos[otros].detach()                                    # (K,N,T,D,2)
                    dist = torch.cdist(da.reshape(-1, self.discos, 2).expand(len(otros), -1, -1, -1).reshape(-1, self.discos, 2),
                                       do.reshape(-1, self.discos, 2)).amin(dim=(-1, -2))
                    dist = dist.view(len(otros), pos_w.size(1), pos_w.size(2))     # (K, N, T)
                    umbral = (r[a] + r[otros] + self.margen).view(-1, 1, 1)
                    validos = torch.isfinite(dist)
                    invasion = torch.where(validos, torch.relu(umbral - dist), torch.zeros_like(dist))
                    perdida[a] = invasion.sum(dim=(0, 2))
            return perdida[agt_mask] if agt_mask is not None else perdida
    gl.GUIDANCE_FUNC_MAP["evita_otros"] = EvitaOtrosLoss
    # scene_editor crea una metrica por cada guiado y falla con nombres que no conoce
    from tbsim.utils import guidance_metrics as gmet
    gmet.GUIDANCE_NAME_TO_METRICS.setdefault("intercepta_ego", {})
    gmet.GUIDANCE_NAME_TO_METRICS.setdefault("evita_otros", {})

    def heuristica_intercepta(sim_scene, dt, **kw):
        nombres = [a.name for a in sim_scene.agents]
        if "ego" not in nombres or estado["adversario"] is None:
            return None
        return {"name": "intercepta_ego",
                "params": {"ego": nombres.index("ego"), "adversario": estado["adversario"]},
                "agents": None}
    seu.HEURISTIC_FUNC["intercepta_ego"] = heuristica_intercepta

    def heuristica_evita(sim_scene, dt, **kw):
        cfg = heuristica_intercepta(sim_scene, dt)
        if cfg is not None:
            cfg["name"] = "evita_otros"
        return cfg
    seu.HEURISTIC_FUNC["evita_otros"] = heuristica_evita
    return estado


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
    ap.add_argument("--objetivo_ego", action="store_true",
                    help="accidentes con el ego: se controlan los agentes mas cercanos (sin el ego) y el "
                         "guiado de colision apunta solo al ego; sin limite de 100 escenas")
    ap.add_argument("--w_ego", type=float, default=1.0,
                    help="peso del guiado 'intercepta_ego' (solo con --objetivo_ego)")
    ap.add_argument("--w_evita", type=float, default=1.0,
                    help="peso del guiado 'evita_otros': el adversario esquiva a todos menos al ego (0 = sin el)")
    ap.add_argument("--semilla", type=int, default=0,
                    help="semilla del sorteo del adversario (otra semilla = otro adversario en la escena)")
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
    if args.objetivo_ego:
        nombre += "_ego" + (f"{txt(args.w_ego)}" if args.w_ego != 1.0 else "")
        if args.w_evita != 1.0:
            nombre += f"_ev{txt(args.w_evita)}"
        if args.semilla:
            nombre += f"_s{args.semilla}"
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
    fuera = [] if args.objetivo_ego else [i for i in escenas if i >= n_max]
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
        if args.objetivo_ego and not any(h.get("name") == "intercepta_ego" for h in self.edits.heuristic_config):
            # va en tercer lugar: scene_editor fija el peso de los dos primeros con
            # --w_collision y --w_offroad
            self.edits.heuristic_config.append({"name": "intercepta_ego", "weight": args.w_ego,
                                                "params": {}, "agents": None})
            if args.w_evita > 0:
                self.edits.heuristic_config.append({"name": "evita_otros", "weight": args.w_evita,
                                                    "params": {}, "agents": None})
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
    if args.objetivo_ego:
        parchear_objetivo_ego(semilla=args.semilla)
        print("[evaluacion] objetivo: accidentes con el ego (ego sin controlar, guiado solo hacia el ego)")
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
