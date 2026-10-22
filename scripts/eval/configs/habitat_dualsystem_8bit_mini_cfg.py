"""
habitat_dualsystem_8bit_mini_cfg.py — 8-bit quantized DualVLN, 10-episode smoke test.

INT8 reduces 7B model from ~14GB → ~7GB, fits on RTX 3080 Ti (12GB).
Uses vln_r2r_mini_debug.yaml (10 episodes) for quick VLN-CE baseline.
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
            "max_new_tokens": 64,
            "vis_debug": False,
            "vis_debug_path": "./logs/vln_ce_mini/vis_dualvln_8bit",
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
        "output_path": "./logs/vln_ce_mini/dualvln_8bit",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2334",
        "dist_url": "env://",
        "use_wandb": False,
    },
)
