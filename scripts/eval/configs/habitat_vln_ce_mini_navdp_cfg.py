"""
habitat_vln_ce_mini_navdp_cfg.py — VLN-CE mini eval for NavDP baseline (10 episodes).
Uses DualVLN checkpoint as NavDP base for comparison.
"""
from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "single_system",
            "model_path": "/checkpoints/InternVLA-N1-DualVLN",
            "num_history": 3,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 64,
            "vis_debug": False,
            "vis_debug_path": "./logs/vln_ce_mini/vis_navdp",
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
        "output_path": "./logs/vln_ce_mini/navdp_baseline",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2335",
        "dist_url": "env://",
        "use_wandb": False,
    },
)
