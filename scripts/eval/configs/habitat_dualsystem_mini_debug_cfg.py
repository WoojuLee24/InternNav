"""
habitat_dualsystem_mini_debug_cfg.py — 10-episode smoke test.
Verifies full pipeline works end-to-end before launching full eval.
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
            "vis_debug": True,
            "vis_debug_path": "./logs/habitat_debug/vis_debug_mini",
            "load_in_8bit": True,   # 7B bfloat16=14GB > 12GB VRAM; 8-bit~7GB fits
            "attn_implementation": "eager",
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
        "output_path": "./logs/habitat_debug/mini_dualsystem_dualvln",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2334",
        "dist_url": "env://",
        "use_wandb": False,
    },
)
