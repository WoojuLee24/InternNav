"""
habitat_dualsystem_s2only_cfg.py

System2-only mode: Uses Qwen2.5-VL text output + ShortestPathFollower.
Separate from dual_system to compare strategies.
Uses InternVLA-N1-DualVLN as S2 backbone (Qwen2.5-VL component).

Published: SR=60.9% (InternVLA-N1 S2 + SPF)
"""

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "system2",
            "model_path": "/checkpoints/InternVLA-N1-DualVLN",
            "num_history": 8,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 128,
            "vis_debug": False,
            "vis_debug_path": "./logs/habitat_debug/vis_debug_s2",
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': '/workspace/ChronoNav/scripts/vln_eval/vln_r2r_debug.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat_debug/s2only_dualvln",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2334",
        "dist_url": "env://",
        "use_wandb": False,
    },
)
