"""
habitat_dual_system_mini_5090_cfg.py 기반.
validate_depth_to_bev_habitat.py에서 model_settings의 카메라 파라미터를 읽어
yaml sensor config 기본값을 override한다.

VLN-CE 기본값 (vln_r2r_mini.yaml 기준):
  hfov=79°, W=640, H=480 → fx=fy≈388.35, cx=319.5, cy=239.5
  cam_height=1.25m (rgb_sensor.position[1])
  cam_pitch_deg=0.0 (수평)

카메라 파라미터 우선순위:
  yaml sensor config (기본) → model_settings (override)
"""

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

# ── 카메라 파라미터 설정 ──────────────────────────────────────────────
# VLN-CE 기본값과 동일하게 명시. 다른 값으로 수정하면 자동으로 override됨.
# cam_height: 지면에서 카메라까지 높이 [m] (yaml position[1]=1.25)
# cam_pitch_deg: 양수=하향, 0=수평, 음수=상향
# camera_intrinsic: fx=fy=320/tan(hfov/2 rad), hfov=79° → tan(39.5°)≈0.8243 → fx≈388.35
CAMERA_HEIGHT = 1.25
CAMERA_PITCH_DEG = 0.0
CAMERA_INTRINSIC = [
    [388.35, 0.0,    319.5],
    [0.0,    388.35, 239.5],
    [0.0,    0.0,    1.0  ],
]
# ──────────────────────────────────────────────────────────────────────

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",
            'model_path': "/ws/src/InternNav/checkpoints/InternVLA-N1-w-NavDP",
            "num_history": 8,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 1024,
            "vis_debug": False,
            "vis_debug_path": "./logs/habitat/vis_debug",
            # Camera parameters for validate_depth_to_bev_habitat.py
            'cam_height': CAMERA_HEIGHT,
            'cam_pitch_deg': CAMERA_PITCH_DEG,
            'camera_intrinsic': CAMERA_INTRINSIC,
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': 'scripts/eval/configs/vln_r2r_mini.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat/test_dual_system",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2333",
        "dist_url": "env://",
        "use_wandb": False,
        "wandb_project": "internnav",
        "wandb_run_name": "habitat_custom_cam",
    },
)
