"""
habitat_vln_ce_mini_nexusnav_cfg.py — VLN-CE mini eval for NexusNav-V3 (10 episodes).
DualVLN + LatentReasoner (4 ThoughtCells) + ActionTokenVocabulary + Memory modules.
"""
from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='nexusnav_v3',
        model_settings={
            "mode": "dual_system",
            "model_path": "/checkpoints/InternVLA-N1-DualVLN",
            "num_history": 3,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 128,
            "vis_debug": False,
            "vis_debug_path": "./logs/vln_ce_mini/vis_nexusnav",
            "attn_implementation": "eager",
            "load_in_8bit": True,
            "use_kv_cache": True,
            "vision_cache_tau": 0.92,
            "use_latent_reasoner": True,
            "use_action_tokens": True,
            "use_memory": True,
            "thought_dim": 256,
            "n_thought_steps": 4,
            "action_vocab_size": 256,
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
        "output_path": "./logs/vln_ce_mini/nexusnav_v3",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2337",
        "dist_url": "env://",
        "use_wandb": False,
    },
)
