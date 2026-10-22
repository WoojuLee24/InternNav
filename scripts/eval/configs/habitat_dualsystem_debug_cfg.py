"""
habitat_dualsystem_debug_cfg.py

Corrected config for Habitat VLN-CE dual-system evaluation.
Uses InternVLA-N1-DualVLN checkpoint (best available, 64.3% SR published).
Fixed YAML with correct scene_datasets paths.

Run:
    cd /workspace/InternNav-dev
    python scripts/eval/eval.py --config scripts/eval/configs/habitat_dualsystem_debug_cfg.py
"""

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",
            "model_path": "/checkpoints/InternVLA-N1-DualVLN",
            "num_history": 3,   # reduced from 8 to fit RTX 3090 24GB; 8 needs 32+ GB
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 128,
            "vis_debug": False,
            "vis_debug_path": "./logs/habitat_debug/vis_debug",
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            # Fixed YAML: uses absolute paths for scene_datasets and data
            'config_path': '/workspace/ChronoNav/scripts/vln_eval/vln_r2r_debug.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat_debug/dualsystem_dualvln",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2334",
        "dist_url": "env://",
        "use_wandb": False,
    },
)
