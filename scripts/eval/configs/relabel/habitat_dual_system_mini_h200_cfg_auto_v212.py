"""생성물 — build_relabel_dataset.py --labels auto_v212 --emit yaml.

원본: scripts/eval/configs/habitat_dual_system_mini_h200_cfg.py (config_path 한 줄만 다름).
"""

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",  # inference mode: dual_system or system2
            'model_path': "/home/irteam/git/InternNav/checkpoints/InternVLA-N1-w-NavDP",
            "num_history": 8,
            "resize_w": 384,  # image resize width
            "resize_h": 384,  # image resize height
            "max_new_tokens": 1024,  # maximum number of tokens for generation
            "vis_debug": False,  # If vis_debug=True, save debug videos per episode
            "vis_debug_path": "./logs/habitat/vis_debug",
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': 'scripts/eval/configs/relabel/vln_r2r_mini_auto_v212.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat/test_dual_system_mini",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2333",
        "dist_url": "env://",
        "use_wandb": True,
        "wandb_project": "huggingface",
    },
)
