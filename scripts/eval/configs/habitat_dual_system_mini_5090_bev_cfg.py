"""Habitat dual-system eval with BEV visual input (RTX 5090 mini split).

Same as habitat_dual_system_mini_5090_cfg.py except:
  - eval_type='habitat_vln_bev'  → HabitatVLNEvaluatorBEV (new file, parent untouched)
  - model_settings gains the visual_provider / bev_* keys

Set visual_provider='fpv' to reproduce the original behaviour exactly.
bev_fx/fy/cx/cy, bev_ref_*, bev_cam_height are auto-filled from the habitat
sensor config when omitted; S1/S2 pitch defaults to bev_cam_pitch_deg + 60°
(look-down frames). bev_z_min/z_max default to the training values (-0.2/2.5).
"""

# Registers Evaluator 'habitat_vln_bev' (import side effect — do not remove)
import internnav.habitat_extensions.vln.habitat_vln_evaluator_bev  # noqa: F401  # isort: skip

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",  # inference mode: dual_system or system2
            'model_path': "/ws/src/InternNav/checkpoints/InternVLA-N1-w-NavDP",
            "num_history": 4,
            "resize_w": 256,  # image resize width
            "resize_h": 256,  # image resize height
            "max_new_tokens": 256,  # maximum number of tokens for generation
            "vis_debug": False,  # If vis_debug=True, save debug videos per episode
            "vis_debug_path": "./logs/habitat/vis_debug",
            # ---------------- BEV visual input ----------------
            "visual_provider": "bev_image",  # 'fpv' | 'bev_image' | 'bev_feature'
            "bev_s1": True,   # replace NavDP images_dp with BEV frames
            "bev_s2": True,   # append BEV image to the LLM look-down turn
            "bev_cam_pitch_deg": 0.0,  # Habitat base camera is horizontal
            "bev_depth_scale": 1.0,    # evaluator hands metric depth to the provider
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': 'scripts/eval/configs/vln_r2r_mini_5090.yaml',
        },
    ),
    eval_type='habitat_vln_bev',
    eval_settings={
        "output_path": "./logs/habitat/test_dual_system_bev",  # output directory for logs/results
        "save_video": False,  # whether to save videos
        "epoch": 0,  # epoch number for logging
        "max_steps_per_episode": 500,  # maximum steps per episode
        # distributed settings
        "port": "2333",  # communication port
        "dist_url": "env://",  # url for distributed setup
        # wandb settings
        "use_wandb": False,
        "wandb_project": "internnav",
        "wandb_run_name": "habitat_dual_system_mini_bev",
    },
)
