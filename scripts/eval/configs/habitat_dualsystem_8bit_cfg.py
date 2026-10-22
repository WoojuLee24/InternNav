"""
habitat_dualsystem_8bit_cfg.py

8-bit quantized dual-system evaluation.
bitsandbytes INT8 reduces model from ~14GB → ~7GB,
allowing num_history=8 (full context) on 24GB RTX 3090.

The evaluator uses the HabitatVLNEvaluator but we override model loading.
"""

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",
            "model_path": "/checkpoints/InternVLA-N1-DualVLN",
            "num_history": 8,           # full history (enabled by 8-bit)
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 128,
            "vis_debug": False,
            "vis_debug_path": "./logs/habitat_debug/vis_debug_8bit",
            "attn_implementation": "eager",
            "load_in_8bit": True,       # enable 8-bit quantization
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
        "output_path": "./logs/habitat_debug/dualsystem_dualvln_8bit",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2334",
        "dist_url": "env://",
        "use_wandb": False,
    },
)
