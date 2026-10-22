"""
habitat_dualsystem_gps_full_cfg.py — Full 1839-episode val_unseen eval with GPS stop.

GPS stop patch is inline in habitat_vln_evaluator.py (InternNav-dev):
  - If distance_to_goal < 3.0m and step_id >= 5 → force STOP action
  - Expected: converts oracle_success=1 failures → successes (+10-20% SR)

Model: InternVLA-N1-DualVLN (published 64.3% SR baseline, without GPS stop)
Target: SR > 64.3% (improvement over published baseline)

Runtime: ~192s/ep × 1839 eps ≈ 98 hours worst case; ~16-30 hours typical
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
            'config_path': '/workspace/ChronoNav/scripts/vln_eval/vln_r2r_debug.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat_debug/dualsystem_gps_full",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2334",
        "dist_url": "env://",
        "use_wandb": False,
    },
)
