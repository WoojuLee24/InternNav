"""
habitat_vln_ce_b2_uncached_cfg.py — VisionEmbeddingCache A/B: UNCACHED arm.

DualVLN S2-only (4-bit) WITHOUT VisionEmbeddingCache.
50 episodes, multi-scene (val_unseen). Emit timing_rank0.json.
Compare against b2_cached_cfg.py for paper latency table.

NOTE: The completed s2only_4bit run (n=12, single-scene) showed s2_ms=1517ms
with cache_hit_rate=0.0. This config replicates that on 50 multi-scene episodes
for a statistically robust A/B comparison.
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
            # vision_cache_tau intentionally absent — no cache
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': '/workspace/ChronoNav/scripts/vln_eval/vln_r2r_mini50.yaml',
            'max_episodes': 50,
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/vln_ce_b2/uncached",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2339",
        "dist_url": "env://",
        "use_wandb": False,
    },
)
