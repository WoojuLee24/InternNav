"""Default config (single source of truth) for qwenvl_train train+eval experiments.

Separated out of ``runner.py`` so that runner is a pure engine. This file defines
the config — and nothing about *how* it runs:
  - the ``Params`` dataclass (every knob + baseline default values), and
  - the eval-config builders (``build_eval_cfg`` / Habitat / h1).
Experiment configs in ``batch_size/`` import this module and override only the
knobs they change. ``runner.py`` imports ``Params`` from here and uses this file
as its default ``--config`` (a bare baseline run).

train and eval read the SAME ``Params`` object, so they can never drift on
``num_history`` / ``resize_*`` / ``predict_step_num`` / ``num_future_steps``.

Eval target selection (matches the existing reference configs):

    TRAIN_EVAL_TARGET=h200   # default; == habitat_dual_system_mini_h200_cfg.py (8-GPU)
    TRAIN_EVAL_TARGET=5090   #          == habitat_dual_system_mini_5090_cfg.py
    TRAIN_EVAL_TARGET=h1     # Isaac Sim / VLN-PE (run manually)

``runner.train_and_eval`` sets this automatically from ``--machine``.
Baseline values mirror scripts/train/qwenvl_train/batch_size/b4_eff128_base.sh.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional


# --------------------------------------------------------------------------- #
# Single source of truth
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    """All knobs for one train+eval experiment.

    Fields are grouped: (a) shared train<->eval params, (b) train-only params,
    (c) infra. Baseline values mirror
    scripts/train/qwenvl_train/batch_size/b4_eff128_base.sh exactly.
    """

    # ---- shared between train and eval (must match!) ----
    num_history: int = 8
    resize_w: int = 384
    resize_h: int = 384
    predict_step_num: int = 32
    num_future_steps: int = 4

    # ---- train-only ----
    sample_step: int = 4
    lr: float = 1e-4
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_pixels: int = 313600
    min_pixels: int = 3136
    val_ratio: float = 0.1
    num_train_epochs: float = 3.0
    weight_decay: float = 0.0
    warmup_ratio: float = 0.003
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_min_lr: float = 1e-05
    save_total_limit: int = 1
    save_only_model: bool = True  # skip DeepSpeed optimizer states (~32G/ckpt); set False to enable training resume
    logging_steps: int = 1
    model_max_length: int = 8192
    dataloader_num_workers: int = 4
    data_augmentation: bool = True
    pixel_goal_only: bool = True
    data_flatten: bool = False
    tune_mm_vision: bool = False
    tune_mm_mlp: bool = False
    tune_mm_llm: bool = False
    gradient_checkpointing: bool = True
    system1: str = "nextdit_async"

    # ---- BEV visual input (bev=False => plain trainer + fpv eval, unchanged) ----
    bev: bool = False              # master toggle for the BEV pipeline (train + eval)
    # -- train + eval (S1 mode is train-synced) --
    bev_s1_mode: str = "bev"       # 'fpv' | 'bev' | 'fpv_bev'   (S1 train mode, used when bev=True)
    bev_image_type: str = "rgb"    # 'rgb' | 'occ'              (train)
    bev_depth_source: str = "gt"   # 'gt' | 'dav2'   (train + eval)
    bev_dav2_max_depth: float = 10.0               # dav2 metric depth cap (metres)
    bev_z_min: float = -0.2                        # height filter lower bound (metres)
    bev_z_max: float = 2.5                         # height filter upper bound (metres)
    # -- eval only --
    bev_s2_mode_eval: str = "fpv"     # eval-time S2 mode (S2 BEV is not trained; train always fpv)
    bev_visual_provider: str = "bev_image"  # 'fpv' | 'bev_image' | 'bev_feature'
    bev_cam_pitch_deg: float = 0.0          # Habitat base camera is horizontal
    bev_depth_scale: float = 1.0            # evaluator hands metric depth to the provider
    debug_dir: str = None                   # BEV debug image output dir (None = disabled)
    bev_s2_always: bool = False             # True → pass BEV to S2 every step; False → only on LOOKDOWN steps

    # ---- unified image provider (view x type x mode x combine, independent of `bev`) ----
    image_provider: bool = False   # master toggle; mutually exclusive with `bev` (different trainer/eval classes)
    s1_image_view: str = "fpv"        # 'fpv' | 'bev'
    s1_image_type: str = "rgb"        # 'rgb' | 'depth' | 'panorama' (panorama not implemented)
    s1_image_mode: str = "raw"        # value-processing; vocabulary depends on (view,type) — see unified_image_provider.py
    s1_combine_mode: str = "none"     # 'none' | 'replace' | 'concat' (concat only supported via view='bev')
    s2_image_view: str = "fpv"
    s2_image_type: str = "rgb"
    s2_image_mode: str = "raw"
    s2_combine_mode: str = "none"
    s2_source_view: str = "lookdown"  # 'lookdown' | 'fpv' — which image is fed to get_s2_extra in the else branch
    depth_adapter_mode: str = "repeat"  # 'repeat' (no params) | 'conv' (learnable 1x1) — only used for 1ch depth images

    # ---- smoke test ----
    max_steps: int = -1  # -1 = unset (train num_train_epochs as usual); >0 = stop after N steps (also disables mid-train eval/save/load-best, which a run this short can't satisfy)
    eval_max_episodes: Optional[int] = None  # None = unset (eval full split as usual); >0 = cap episodes for a quick eval-rollout smoke test

    # ---- wandb ----
    use_wandb: bool = True  # False -> no wandb run created/resumed for this eval (runner.py sets this False for --max-steps/--debugpy smoke runs)

    # ---- Isaac Sim (h1 eval only) ----
    headless: bool = False  # True -> no Isaac Sim GUI window (unchanged default: GUI shown)
    flash_collision: str = "stop"  # 'stop' | 'reset' | 'none' (-> None: no detection). Unchanged default: 'stop'.

    # ---- data / model paths ----
    vln_datasets: str = "r2r_125cm_0_30%30,r2r_60cm_15_15%30"
    data_root: str = "/home/irteam/git/InternNav/data/InternData-N1-v0.5-mini/vln_ce"
    system2_ckpt: str = "/home/irteam/data-vol2/checkpoints/InternVLA-N1-System2"
    deepspeed: str = "scripts/train/qwenvl_train/zero2.json"

    # ---- distributed launcher ----
    nnodes: int = 1
    nproc_per_node: int = 8
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 0  # 0 -> picked at launch (mirrors $((RANDOM%101+20001)))

    # ---- derived ----
    @property
    def trainer(self) -> str:
        """Trainer entry script. BEV / unified image provider swap in their own
        provider trainer (base file untouched); at most one is active per run."""
        if self.bev:
            return "internnav/trainer/internvla_n1_bev_provider_trainer.py"
        if self.image_provider:
            return "internnav/trainer/internvla_n1_unified_provider_trainer.py"
        return "internnav/trainer/internvla_n1_trainer.py"

    @property
    def per_device_eval_batch_size(self) -> int:
        return self.batch_size * 2

    @property
    def val_interval_steps(self) -> int:
        # mirrors $((100 * 4 / batch_size)) in the shell scripts
        return 100 * 4 // self.batch_size

    def train_argv(self, output_dir: str, run_name: str) -> List[str]:
        """Flags passed to internnav/trainer/internvla_n1_trainer.py.

        Order/values mirror the batch_size/*.sh scripts so the runs are identical.
        """
        b = lambda x: "True" if x else "False"  # noqa: E731  (shell passed True/False strings)
        # BEV flags are peeled off by internvla_n1_bev_provider_trainer.py before the
        # base HfArgumentParser sees them; placed first, mirroring the bev/*.sh scripts.
        bev_argv = []
        if self.bev:
            bev_argv = [
                "--bev_s1_mode", self.bev_s1_mode,
                "--bev_image_type", self.bev_image_type,
                "--bev_depth_source", self.bev_depth_source,
                "--bev_z_min", str(self.bev_z_min),
                "--bev_z_max", str(self.bev_z_max),
            ]
            if self.debug_dir:
                bev_argv += ["--debug_dir", self.debug_dir]
        image_argv = []
        if self.image_provider:
            image_argv = [
                "--s1_image_view", self.s1_image_view,
                "--s1_image_type", self.s1_image_type,
                "--s1_image_mode", self.s1_image_mode,
                "--s1_combine_mode", self.s1_combine_mode,
                "--bev_depth_source", self.bev_depth_source,
                "--bev_z_min", str(self.bev_z_min),
                "--bev_z_max", str(self.bev_z_max),
                "--depth_adapter_mode", self.depth_adapter_mode,
            ]
            if self.debug_dir:
                image_argv += ["--debug_dir", self.debug_dir]
        # S2 training image injection is a DataArguments concern (dataset-level,
        # not model-class-level) — passed regardless of which trainer script runs.
        s2_argv = []
        if self.s2_combine_mode != "none":
            s2_argv = [
                "--s2_image_view", self.s2_image_view,
                "--s2_image_type", self.s2_image_type,
                "--s2_image_mode", self.s2_image_mode,
                "--s2_combine_mode", self.s2_combine_mode,
            ]
        smoke = self.max_steps > 0
        argv = bev_argv + image_argv + s2_argv + [
            "--deepspeed", self.deepspeed,
            "--model_name_or_path", self.system2_ckpt,
            "--vln_dataset_use", self.vln_datasets,
            "--data_root", self.data_root,
            "--data_flatten", b(self.data_flatten),
            "--tune_mm_vision", b(self.tune_mm_vision),
            "--tune_mm_mlp", b(self.tune_mm_mlp),
            "--tune_mm_llm", b(self.tune_mm_llm),
            "--bf16",
            "--num_history", str(self.num_history),
            "--data_augmentation", b(self.data_augmentation),
            "--resize_h", str(self.resize_h),
            "--resize_w", str(self.resize_w),
            "--sample_step", str(self.sample_step),
            "--num_future_steps", str(self.num_future_steps),
            "--predict_step_num", str(self.predict_step_num),
            "--pixel_goal_only", b(self.pixel_goal_only),
            "--system1", self.system1,
            "--output_dir", output_dir,
            "--num_train_epochs", str(self.num_train_epochs),
            "--per_device_train_batch_size", str(self.batch_size),
            "--per_device_eval_batch_size", str(self.per_device_eval_batch_size),
            "--gradient_accumulation_steps", str(self.grad_accum_steps),
            "--max_pixels", str(self.max_pixels),
            "--min_pixels", str(self.min_pixels),
            "--val_ratio", str(self.val_ratio),
            "--eval_strategy", "no" if smoke else "steps",
            "--eval_steps", str(self.val_interval_steps),
            "--save_strategy", "no" if smoke else "steps",
            "--save_steps", str(self.val_interval_steps),
            "--save_total_limit", str(self.save_total_limit),
            "--save_only_model", "True" if self.save_only_model else "False",
            "--metric_for_best_model", "eval_loss",
            "--greater_is_better", "False",
            "--load_best_model_at_end", b(not smoke),
            "--learning_rate", _fmt_num(self.lr),
            "--weight_decay", _fmt_num(self.weight_decay),
            "--warmup_ratio", _fmt_num(self.warmup_ratio),
            "--max_grad_norm", _fmt_num(self.max_grad_norm),
            "--lr_scheduler_type", self.lr_scheduler_type,
            "--lr_scheduler_kwargs", json.dumps({"min_lr": self.lr_scheduler_min_lr}),
            "--logging_steps", str(self.logging_steps),
            "--model_max_length", str(self.model_max_length),
            "--gradient_checkpointing", b(self.gradient_checkpointing),
            "--dataloader_num_workers", str(self.dataloader_num_workers),
            "--run_name", run_name,
            "--report_to", "none" if (smoke or not self.use_wandb) else "wandb",
        ]
        if smoke:
            argv += ["--max_steps", str(self.max_steps)]
        return argv


def _fmt_num(x: float) -> str:
    """Format like the shell literals: 1e-4, 0, 0.003, 1, 1e-05."""
    if x == 0:
        return "0"
    if x == int(x):
        return str(int(x))
    # keep scientific notation compact (1e-4, 1e-05) as in the scripts
    return repr(x)


# --------------------------------------------------------------------------- #
# Eval config builders (consumed by scripts/eval/eval.py via --config)
# --------------------------------------------------------------------------- #
# Machine-specific train infra. Applied by runner.py from --machine.
# Only paths / launcher knobs differ; all model hyperparams stay in Params.
TRAIN_MACHINE = {
    "h200": {
        "data_root": "/home/irteam/git/InternNav/data/InternData-N1-v0.5-mini/vln_ce",
        "system2_ckpt": "/home/irteam/data-vol2/checkpoints/InternVLA-N1-System2",
        "nproc_per_node": 8,
        "checkpoints_root": "/home/irteam/data-vol2/checkpoints",
    },
    "5090": {
        "data_root": "/ws/src/InternNav/data/InternData-N1-v0.5-mini/vln_ce",
        "system2_ckpt": "/ws/src/InternNav/checkpoints/InternVLA-N1-System2",
        "nproc_per_node": 1,
        "checkpoints_root": "/ws/src/InternNav/checkpoints",
        "batch_size": 2,
        "grad_accum_steps": 4,
        "max_pixels": 150528,  # 336*336*4/3 — fits 24 GB VRAM
        "debug_dir": "./logs/5090_debug",  # visualization output dir for the BEV evaluator (also used by the BEV trainer when bev=True)
    },
}

# --------------------------------------------------------------------------- #
# Machine-specific eval infra (mirrors the existing reference configs), keyed by
# the same --machine values as build_eval_cfg (h200/5090 = Habitat, h1 = Isaac Sim).
# num_history / resize_* / predict_step_num are NOT here — they come from the
# training Params so train and eval stay in lockstep. Only the host/runtime
# settings differ per machine. build_habitat_eval_cfg() consumes the full dict
# (h200/5090 only); build_h1_eval_cfg() and runner.py's checkpoint fallback only
# read "model_path", so the "h1" entry only needs that key.
EVAL_MACHINE = {
    # scripts/eval/configs/habitat_dual_system_mini_h200_cfg.py (H200, 8-GPU)
    "h200": {
        "model_path": "/home/irteam/git/InternNav/checkpoints/InternVLA-N1-w-NavDP",
        "max_new_tokens": 1024,
        "config_path": "scripts/eval/configs/vln_r2r_mini.yaml",
        "output_path": "./logs/habitat/test_dual_system_mini",
        "use_wandb": True,
        "wandb_project": "huggingface",
        "extra_eval": {},
    },
    # scripts/eval/configs/habitat_dual_system_mini_5090_cfg.py (RTX 5090)
    "5090": {
        "model_path": "/ws/src/InternNav/checkpoints/InternVLA-N1-DualVLN",
        "max_new_tokens": 256,
        "config_path": "scripts/eval/configs/vln_r2r_mini_5090.yaml",
        "output_path": "./logs/habitat/test_dual_system",
        "use_wandb": False,
        "wandb_project": "internnav",
        "extra_eval": {"wandb_run_name": "habitat_dual_system_mini_single"},
        "debug_dir": "./logs/5090_debug",  # visualization output dir for the BEV evaluator (also used by the BEV trainer when bev=True)
    },
    # Isaac Sim / VLN-PE (run manually); see build_h1_eval_cfg
    "h1": {
        "model_path": "/ws/src/InternNav/checkpoints/InternVLA-N1-DualVLN",
    },
}


def build_habitat_eval_cfg(p: Params, machine: str = "h200"):
    """Habitat (VLN-CE) eval config = machine infra (h200|5090) + train-synced params.

    Mirrors scripts/eval/configs/habitat_dual_system_mini_{machine}_cfg.py. The old
    batch_size scripts pointed eval at habitat_dual_system_mini_cfg.py, which does
    not exist on disk — this is the working, parameter-matched replacement.

    `predict_step_num` is set explicitly to keep train==eval under ablation; with the
    baseline value 32 it equals the evaluator's hard-coded default, so behavior is
    identical to the reference configs that omit the key.
    """
    if machine not in ("h200", "5090"):
        raise ValueError(f"unknown habitat machine: {machine!r} (expected 'h200' or '5090')")
    m = EVAL_MACHINE[machine]

    from internnav.configs.agent import AgentCfg
    from internnav.configs.evaluator import EnvCfg, EvalCfg

    model_settings = {
        "mode": "dual_system",
        "model_path": m["model_path"],  # overridden by eval.py --model_path
        "num_history": p.num_history,
        "resize_w": p.resize_w,
        "resize_h": p.resize_h,
        "predict_step_num": p.predict_step_num,
        "max_new_tokens": m["max_new_tokens"],
        "vis_debug": False,
        "vis_debug_path": "./logs/habitat/vis_debug",
    }
    eval_type = "habitat_vln"
    output_path = m["output_path"]

    if p.bev:
        # Mirrors habitat_dual_system_mini_{machine}_bev_cfg.py. Importing the BEV
        # evaluator module registers Evaluator 'habitat_vln_bev' (import side effect).
        import internnav.habitat_extensions.vln.habitat_vln_evaluator_bev  # noqa: F401
        eval_type = "habitat_vln_bev"
        output_path = output_path + "_bev"
        model_settings.update({
            "visual_provider": p.bev_visual_provider,  # 'fpv' | 'bev_image' | 'bev_feature'
            "bev_s1_mode": p.bev_s1_mode,        # train-synced
            "bev_s2_mode": p.bev_s2_mode_eval,   # S2 BEV not trained -> fpv by default
            "bev_cam_pitch_deg": p.bev_cam_pitch_deg,  # Habitat base camera is horizontal
            "bev_depth_scale": p.bev_depth_scale,      # evaluator hands metric depth to the provider
            "bev_image_type": p.bev_image_type,        # 'rgb' | 'occ' — train-synced
            "bev_depth_source": p.bev_depth_source,    # 'gt' | 'dav2' — train-synced
            "bev_dav2_max_depth": p.bev_dav2_max_depth,
            "bev_z_min": p.bev_z_min,
            "bev_z_max": p.bev_z_max,
            "debug_dir": p.debug_dir or os.environ.get("BEV_DEBUG_DIR"),
            "bev_s2_always": p.bev_s2_always,
        })
    elif p.image_provider:
        import internnav.habitat_extensions.vln.habitat_vln_evaluator_unified  # noqa: F401
        eval_type = "habitat_vln_unified"
        output_path = output_path + "_image_provider"
        model_settings.update({
            "visual_provider": "unified_image",
            "s1_image_view": p.s1_image_view,
            "s1_image_type": p.s1_image_type,
            "s1_image_mode": p.s1_image_mode,
            "s1_combine_mode": p.s1_combine_mode,
            "s2_image_view": p.s2_image_view,
            "s2_image_type": p.s2_image_type,
            "s2_image_mode": p.s2_image_mode,
            "s2_combine_mode": p.s2_combine_mode,
            "s2_source_view": p.s2_source_view,
            "bev_cam_pitch_deg": p.bev_cam_pitch_deg,
            "bev_depth_scale": p.bev_depth_scale,
            "bev_depth_source": p.bev_depth_source,
            "bev_dav2_max_depth": p.bev_dav2_max_depth,
            "bev_z_min": p.bev_z_min,
            "bev_z_max": p.bev_z_max,
            "depth_adapter_mode": p.depth_adapter_mode,
            "debug_dir": p.debug_dir or os.environ.get("BEV_DEBUG_DIR"),
        })

    return EvalCfg(
        agent=AgentCfg(
            model_name="internvla_n1",
            model_settings=model_settings,
        ),
        env=EnvCfg(
            env_type="habitat",
            env_settings={"config_path": m["config_path"]},
        ),
        eval_type=eval_type,
        eval_settings={
            "output_path": output_path,
            "save_video": False,
            "epoch": 0,
            "max_steps_per_episode": 500,
            "max_episodes": p.eval_max_episodes or (int(os.environ["EVAL_MAX_EPISODES"]) if os.environ.get("EVAL_MAX_EPISODES") else None),
            "port": "2333",
            "dist_url": "env://",
            "use_wandb": m["use_wandb"] and p.use_wandb,
            "wandb_project": m["wandb_project"],
            **m["extra_eval"],
        },
    )


def build_h1_eval_cfg(p: Params, model_path: str = EVAL_MACHINE["h1"]["model_path"]):
    """h1 (Isaac Sim / InternUtopia, VLN-PE) eval config, params synced from `p`.

    Based on scripts/eval/configs/h1_internvla_n1_async_cfg.py. Requires the Isaac
    Sim environment, so it is normally launched manually (not from train_and_eval).
    """
    from internnav.configs.agent import AgentCfg
    from internnav.configs.evaluator import EnvCfg, EvalCfg, EvalDatasetCfg, SceneCfg, TaskCfg

    model_name = "internvla_n1"
    model_settings = {
        "env_num": 1,
        "sim_num": 1,
        "model_path": model_path,  # overridden by eval.py --model_path
        "camera_intrinsic": [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
        "width": 640,
        "height": 480,
        "hfov": 79,
        "resize_w": p.resize_w,
        "resize_h": p.resize_h,
        "max_new_tokens": 1024,
        "num_frames": 32,
        "num_history": p.num_history,
        "num_future_steps": p.num_future_steps,
        "device": "cuda:0",
        "predict_step_nums": p.predict_step_num,
        "continuous_traj": True,
        "infer_mode": "partial_async",
        "vis_debug": False,
        "vis_debug_path": "./logs/test_n1/vis_debug",
    }

    if p.image_provider:
        if p.s2_image_view == 'bev_ld':
            # Unlike habitat_vln_evaluator_unified.py (which physically re-points the
            # camera down and computes a genuine lookdown-pitch BEV — see its
            # _LOOKDOWN_PITCH_OFFSET_DEG), the h1/Isaac path has no lookdown camera:
            # internvla_n1_agent.py's look_down turn sends a no-op action ([-1]) and
            # reuses the same forward-facing rgb/depth, and internutopia_env.py has no
            # camera-pitch control at all. So 'bev_ld' here would silently compute the
            # exact same BEV as 'bev' (same frame, same fixed rig pitch) while claiming
            # to be a lookdown view. Fail loudly instead until h1 gets a real lookdown
            # camera; use s2_image_view='bev' in the meantime.
            raise ValueError(
                "s2_image_view='bev_ld' is not supported on the h1/Isaac path: there is "
                "no lookdown camera implemented for h1 (see internvla_n1_agent.py's "
                "look_down handling and internutopia_env.py). Use s2_image_view='bev' "
                "instead, or run this config on habitat where lookdown is implemented."
            )
        # Mirrors build_habitat_eval_cfg's `elif p.image_provider:` branch. Importing
        # the agent module registers Agent 'internvla_n1_unified' (import side effect).
        import internnav.agent.internvla_n1_agent_unified  # noqa: F401

        model_name = "internvla_n1_unified"
        model_settings.update({
            "visual_provider": "unified_image",
            "s1_image_view": p.s1_image_view,
            "s1_image_type": p.s1_image_type,
            "s1_image_mode": p.s1_image_mode,
            "s1_combine_mode": p.s1_combine_mode,
            "s2_image_view": p.s2_image_view,
            "s2_image_type": p.s2_image_type,
            "s2_image_mode": p.s2_image_mode,
            "s2_combine_mode": p.s2_combine_mode,
            "bev_depth_source": p.bev_depth_source,
            "bev_dav2_max_depth": p.bev_dav2_max_depth,
            "bev_z_min": p.bev_z_min,
            "bev_z_max": p.bev_z_max,
            "depth_adapter_mode": p.depth_adapter_mode,
            # H1 camera hardware constants (fixed rig, NOT train-synced Params —
            # h1_internvla_n1_async_bev_cfg.py's values, reused unchanged):
            "bev_cam_height": 1.25,
            "bev_cam_pitch_deg": 30.0,
            "bev_fx": 585.0,
            "bev_fy": 585.0,
            "bev_cx": 320.0,
            "bev_cy": 240.0,
            "bev_ref_width": 640,
            "bev_ref_height": 480,
            "bev_depth_scale": 10.0,  # S2 obs depth [0,1] x 10 = metres; S1 depth already metric
            "debug_dir": p.debug_dir or os.environ.get("BEV_DEBUG_DIR"),
        })

    # same env-var relay as EVAL_HEADLESS/EVAL_MAX_EPISODES above (eval.py subprocess
    # re-imports this experiment config fresh, so a runner.py CLI flag set on a
    # different process's Params object can't reach p.flash_collision here directly).
    _flash_collision = os.environ.get("EVAL_FLASH_COLLISION") or p.flash_collision
    _flash_collision = None if _flash_collision == "none" else _flash_collision

    return EvalCfg(
        agent=AgentCfg(
            server_port=8023,
            model_name=model_name,
            ckpt_path="",
            model_settings=model_settings,
        ),
        env=EnvCfg(
            env_type="internutopia",
            env_settings={
                "use_fabric": False,
                # eval.py runs as a fresh subprocess that re-imports this experiment config
                # from scratch, so a runner.py --headless flag (set on a different process's
                # Params object) can't reach p.headless here directly — same env-var relay
                # runner.py already uses for EVAL_MAX_EPISODES/BEV_DEBUG_DIR below.
                "headless": p.headless or os.environ.get("EVAL_HEADLESS") == "1",
                # smoke-test cap (mirrors build_habitat_eval_cfg's eval_settings["max_episodes"]);
                # None => unset => today's full-split behavior, unchanged.
                "max_episodes": p.eval_max_episodes or (
                    int(os.environ["EVAL_MAX_EPISODES"]) if os.environ.get("EVAL_MAX_EPISODES") else None
                ),
            },
        ),
        task=TaskCfg(
            task_name="test_n1",
            task_settings={"env_num": 1, "use_distributed": False, "proc_num": 1, "max_step": 1000},
            scene=SceneCfg(
                scene_type="mp3d",
                scene_data_dir="/ws/src/InternNav/data/InternData-N1-v0.5-mini/scene_data/mp3d_pe",
            ),
            robot_name="h1",
            robot_flash=True,
            robot_platform_size=0.3,
            flash_collision=_flash_collision,
            robot_usd_path="/ws/src/InternNav/data/InternData-N1-v0.5-mini/Embodiments/vln-pe/h1/h1_internvla.usd",
            camera_resolution=[640, 480],
            camera_prim_path="torso_link/h1_1_25_down_30",
            one_step_stand_still=True,
        ),
        dataset=EvalDatasetCfg(
            dataset_type="mp3d",
            dataset_settings={
                "base_data_dir": "/ws/src/InternNav/data/InternData-N1-v0.5-mini/vln_pe/raw_data/r2r",
                "split_data_types": ["val_unseen"],
                "filter_stairs": True,
            },
        ),
        eval_type="vln_distributed",
        eval_settings={
            "save_to_json": True,
            "vis_output": False,
            "show_rgb": False,
            "use_agent_server": False,
            "use_wandb": p.use_wandb,
        },
    )


def build_eval_cfg(p: Params, target: str = "h200"):
    """target: 'h200' or '5090' (Habitat machines) or 'h1' (Isaac Sim).

    'habitat' is accepted as an alias for 'h200' (the H200 8-GPU default).
    """
    if target in ("habitat", "h200"):
        return build_habitat_eval_cfg(p, machine="h200")
    if target == "5090":
        return build_habitat_eval_cfg(p, machine="5090")
    if target == "h1":
        return build_h1_eval_cfg(p)
    raise ValueError(f"unknown eval target: {target!r} (expected 'h200', '5090' or 'h1')")


def make_eval_cfg(params: Params):
    """eval_cfg for the env-selected target (used by experiment configs + this file)."""
    return build_eval_cfg(params, os.environ.get("TRAIN_EVAL_TARGET", "h200"))


# --------------------------------------------------------------------------- #
# This file is itself the default experiment config (= b4 baseline).
# `runner.py` reads PARAMS/eval_cfg here when --config is omitted.
# --------------------------------------------------------------------------- #
EXP_NAME = "batch_size/b4_eff128_base"
PARAMS = Params()  # baseline: batch_size=4, grad_accum_steps=4 (eff 128)
eval_cfg = make_eval_cfg(PARAMS)
