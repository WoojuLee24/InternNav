"""
habitat_s2only_4bit_mini_cfg.py — S2-only VLM eval, 4-bit NF4, 10 episodes.

Uses mode='system2' (VLM text output + action queue, no NavDP trajectory).
Avoids memory_encoder dtype issue with bitsandbytes on 12GB GPU.
Expected baseline: ~60-65% SR on full dataset (published 60.9%).
"""
from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "system2",
            "model_path": "/checkpoints/InternVLA-N1-DualVLN",
            "num_history": 3,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 64,
            "vis_debug": False,
            "vis_debug_path": "./logs/vln_ce_mini/vis_s2only_4bit",
            "attn_implementation": "eager",
            "load_in_4bit": True,
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
        "output_path": "./logs/vln_ce_mini/s2only_4bit",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2334",
        "dist_url": "env://",
        "use_wandb": False,
    },
)
