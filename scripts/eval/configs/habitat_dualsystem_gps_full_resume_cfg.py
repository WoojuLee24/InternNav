"""
habitat_dualsystem_gps_full_resume_cfg.py — Resume of GPS-stop val_unseen eval.

Resumes from episode 715/1839 using the filtered remaining-episodes JSON.
The evaluator's resume_from_output_path() pre-fills metrics from the 714
already-completed episodes in progress_rank0.json, then processes the 1125
remaining episodes and appends their results. Final metrics cover all 1839.

Partial result: 714 ep / SR=79.0% / SPL=0.641 (from full_improved_eval_v10.log)
Bug fixed: IndexError when model outputs garbled text with <2 digit tokens.
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
            "vis_debug_path": "./logs/habitat_debug/vis_debug_gps_full",
            "attn_implementation": "eager",
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': '/workspace/ChronoNav/scripts/vln_eval/vln_r2r_remaining.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat_debug/dualsystem_gps_full",  # same as original → resume picks up 714 results
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2334",
        "dist_url": "env://",
        "use_wandb": False,
    },
)
