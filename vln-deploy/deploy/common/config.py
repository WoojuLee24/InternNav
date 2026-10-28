"""Deployment configuration — the single source of truth every deployed module reads.

Precedence:  manager CLI flags  >  VLN_* environment variables  >  these defaults.

The data-path port numbers MUST stay identical to
`quantization_kit.async_zmq.zmq_proto.DEFAULT_PORTS` so the deployed modules keep
interoperating with the existing tooling (orchestrator / camera_publisher / prompt_cli /
visualizer). The two *_health ports are deploy-only additions (out-of-band REP beacons).
"""
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parents[1]
ROOT = DEPLOY_DIR.parent                                  # repo root (imports rely on it)
QK = ROOT / "quantization_kit"
IS_JETSON = platform.machine() == "aarch64"
LLAMA_BUILD = QK / "llama.cpp" / ("build" if IS_JETSON else "build_x86")   # arch-aware, like run.sh
CONVENV_PY = QK / ("convenv_aarch64" if IS_JETSON else "convenv_x86") / "bin" / "python"  # [fix] Thor/5090 자동 분기

PORTS = {
    "s2_in": 5601,      # manager/orch PUSH -> S2 PULL : rgb (+lookdown_rgb) + instruction; mode=infer/accumulate/reset
    "s2_out": 5602,     # S2 PUB -> manager/orch SUB   : kind=latent (latent+goal_rgb+pixel) | kind=action (discrete)
    "s1_in": 5603,      # manager/orch PUSH -> S1 PULL : latent + rgb_mem (goal) + rgb_cur
    "s1_out": 5604,     # S1 PUSH -> manager/orch PULL : discrete actions [:4] + waypoints + timing
    "cam": 5605,        # camera module PUB -> orchestrator + viewer SUB : live RGB frames + t_cap
    "prompt": 5606,     # manager/prompt_cli PUSH -> orchestrator PULL  : live instruction updates
    "viz": 5607,        # orchestrator PUB -> viewer SUB                : action/pixel/waypoint overlay
    "s2_runner": 5610,  # S2 service REQ <-> C++ llama-s2-runner REP (private to the S2 module)
    "s1_health": 5614,  # deploy-only: S1 HealthBeacon REP (liveness/readiness, out-of-band)
    "s2_health": 5615,  # deploy-only: S2 HealthBeacon REP
    "cam_health": 5616, # deploy-only: camera module HealthBeacon REP
    "viz_health": 5617, # deploy-only: viewer module HealthBeacon REP
}


def _env(name, default, cast=str):
    v = os.environ.get(name)
    if v is None:
        return default
    if cast is bool:
        return v.strip().lower() in ("1", "true", "yes", "on")
    return cast(v)


@dataclass
class DeployConfig:
    # -- where the modules live (point at another machine to consume a remote module) --
    s1_host: str = "127.0.0.1"          # VLN_S1_HOST
    s2_host: str = "127.0.0.1"          # VLN_S2_HOST
    cam_host: str = "127.0.0.1"         # VLN_CAM_HOST  (camera module / publisher box)
    ports: dict = field(default_factory=lambda: dict(PORTS))   # VLN_PORT_<NAME>

    # -- service knobs (manager start/test/demo defaults; each maps to a service CLI flag) --
    device: str = "cuda:0"              # VLN_DEVICE
    s2_model: str = "q4"                # VLN_S2_MODEL   q4 = Q4_K_M-latent (validated) | f16 = reference
    s2_resize: int = 336                # VLN_S2_RESIZE  112-aligned (224/336/448); 448 = HF-parity
    num_history: int = 8                # VLN_NUM_HISTORY
    kv_reuse: bool = True               # VLN_KV_REUSE   multi-turn KV (fast S2; same default as run.sh async)
    s1_steps: int = 10                  # VLN_S1_STEPS   diffusion steps (10 = original agent; 5 ~= 2x faster, waypoints ~13 mm off)
    trt_s1: bool = True                 # VLN_TRT_S1     native TRT FP16 S1 (off -> PyTorch fp32)
    s2_mps_pct: int = 80                # VLN_S2_MPS_PCT MPS SM cap for S2 (0 = no MPS; needs a daemon)
    s1_mps_pct: int = 0                 # VLN_S1_MPS_PCT MPS SM cap for S1 (0 = uncapped MPS client)
    s2_backend: str = "gguf"            # VLN_S2_BACKEND gguf (llama-s2-runner) | hf (transformers)
    s1_backend: str = "lite"            # VLN_S1_BACKEND lite (S1LiteModel + TRT) | hf (transformers)
    hf_model_path: str = ""             # VLN_HF_MODEL   [hf] checkpoint dir ("" = checkpoints/InternVLA-N1-DualVLN)
    hf_quant: str = "bf16"              # VLN_HF_QUANT   [hf S2] bf16 | nf4 | int8
    cam_device: int = 0                 # VLN_CAM_DEVICE /dev/videoN index for the camera module
    viz_view: str = "auto"              # VLN_VIZ_VIEW   viewer backend: auto|tk|mjpeg|save
    mjpeg_port: int = 8089              # VLN_MJPEG_PORT --view mjpeg HTTP port
    # camera model for waypoint projection (see common/camproj.py — repo conventions):
    cam_hfov_deg: float = 90.0          # VLN_CAM_HFOV   HFOV of the published SQUARE frame.
    #   90° = the habitat/VLN-CE camera the model was trained on (also keeps ground points
    #   >=1.26 m in-frame at the default height). The BEV sensor's fx=585@640x480 would be
    #   44.6° after the square crop — recalibrate per real camera.
    cam_height_m: float = 1.25          # VLN_CAM_HEIGHT repo default camera height (unproject_depth)
    cam_pitch_deg: float = 0.0          # VLN_CAM_PITCH  +down (repo convention); 0 = level camera
    instruction: str = "Walk forward along the hallway and stop near the table."

    # -- manager bookkeeping --
    log_dir: Path = DEPLOY_DIR / "logs"
    run_dir: Path = DEPLOY_DIR / "run"  # pidfiles

    @classmethod
    def from_env(cls):
        cfg = cls(
            s1_host=_env("VLN_S1_HOST", "127.0.0.1"),
            s2_host=_env("VLN_S2_HOST", "127.0.0.1"),
            cam_host=_env("VLN_CAM_HOST", "127.0.0.1"),
            device=_env("VLN_DEVICE", "cuda:0"),
            s2_model=_env("VLN_S2_MODEL", "q4"),
            s2_resize=_env("VLN_S2_RESIZE", 336, int),
            num_history=_env("VLN_NUM_HISTORY", 8, int),
            kv_reuse=_env("VLN_KV_REUSE", True, bool),
            s1_steps=_env("VLN_S1_STEPS", 10, int),
            trt_s1=_env("VLN_TRT_S1", True, bool),
            s2_mps_pct=_env("VLN_S2_MPS_PCT", 80, int),
            s1_mps_pct=_env("VLN_S1_MPS_PCT", 0, int),
            s2_backend=_env("VLN_S2_BACKEND", "gguf"),
            s1_backend=_env("VLN_S1_BACKEND", "lite"),
            hf_model_path=_env("VLN_HF_MODEL", ""),
            hf_quant=_env("VLN_HF_QUANT", "bf16"),
            cam_device=_env("VLN_CAM_DEVICE", 0, int),
            viz_view=_env("VLN_VIZ_VIEW", "auto"),
            mjpeg_port=_env("VLN_MJPEG_PORT", 8089, int),
            cam_hfov_deg=_env("VLN_CAM_HFOV", 90.0, float),
            cam_height_m=_env("VLN_CAM_HEIGHT", 1.25, float),
            cam_pitch_deg=_env("VLN_CAM_PITCH", 0.0, float),
        )
        for name in cfg.ports:
            cfg.ports[name] = _env(f"VLN_PORT_{name.upper()}", cfg.ports[name], int)
        return cfg
