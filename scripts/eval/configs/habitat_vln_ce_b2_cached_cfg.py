"""
habitat_vln_ce_b2_cached_cfg.py — VisionEmbeddingCache A/B: CACHED arm.

DualVLN S2-only (4-bit) with VisionEmbeddingCache active (τ=0.92).
50 episodes, multi-scene (val_unseen). Emit timing_rank0.json.
Compare against b2_uncached_cfg.py for paper latency table.
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
            "attn_implementation": "eager",
            "load_in_4bit": True,
            "use_kv_cache": True,
            "vision_cache_tau": 0.92,  # VisionEmbeddingCache active
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': '/workspace/ChronoNav/scripts/vln_eval/vln_r2r_mini50.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/vln_ce_b2/cached",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2338",
        "dist_url": "env://",
        "use_wandb": False,
    },
)
