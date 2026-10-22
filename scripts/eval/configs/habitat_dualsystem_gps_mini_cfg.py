"""
habitat_dualsystem_gps_mini_cfg.py — GPS-stop enabled mini eval (10 episodes).

Adds GPS-based automatic stopping: if distance_to_goal < 3.0m after any
navigation step (and step_id >= 5), issue STOP immediately.

This is the A/B comparison vs habitat_dualsystem_mini_debug_cfg.py (baseline).
Expected: SR should jump from ~33% baseline to ~50-67%.

Same 10 episodes as baseline for fair comparison.
"""

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",
            "model_path": "/checkpoints/InternVLA-N1-DualVLN",
            "num_history": 3,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 128,
            "vis_debug": False,
            "vis_debug_path": "./logs/habitat_debug/vis_debug_gps_mini",
            "attn_implementation": "eager",
            "load_in_8bit": True,
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': '/workspace/ChronoNav/scripts/vln_eval/vln_r2r_mini_debug.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat_debug/dualsystem_gps_mini",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2334",
        "dist_url": "env://",
        "use_wandb": False,
    },
)
