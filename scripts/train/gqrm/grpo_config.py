"""
GRPOConfig — Group Relative Policy Optimization configuration for NexusNav GQRM RL.

C8 contribution: replaces PPO with GRPO (DeepSeek-R1 style), eliminating the value
network and using group-normalized rewards for more stable, sample-efficient training.

Usage:
    from scripts.train.gqrm.grpo_config import GRPOConfig
    cfg = GRPOConfig(base_model_path="/checkpoints/InternVLA-N1-DualVLN")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["GRPOConfig"]


@dataclass
class GRPOConfig:
    """
    Complete configuration for GRPO-GQRM training of NexusNav.

    Key difference from PPO:
      - No value network (critic) — eliminates ~3.5B param overhead
      - Group-normalize rewards within K-rollout batch (mean/std normalization)
      - Bradley-Terry ranking optional but not required (binary reward suffices)
    """

    # ── Model ──────────────────────────────────────────────────────────
    base_model_path: str = "/checkpoints/InternVLA-N1-DualVLN"
    navprm_model_path: str = ""
    """Path to NavPRM checkpoint. Empty string disables NavPRM (outcome reward only)."""

    lora_r: int = 32
    """LoRA rank for policy updates. Matches NexusNavModelConfig default."""
    lora_alpha: int = 64
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"

    # ── GRPO rollout specifics ──────────────────────────────────────────
    n_rollouts_per_state: int = 8
    """K rollouts generated from each starting state. Larger K = more stable gradients."""
    max_episode_steps: int = 30
    """Maximum navigation steps per rollout. R2R avg path = 6 steps × 5 forward = 30."""
    group_size: int = 8
    """Must equal n_rollouts_per_state. Enforced at runtime."""

    temperature: float = 0.9
    """Sampling temperature for rollout generation. Slightly high to encourage exploration."""
    top_p: float = 0.95
    """Nucleus sampling threshold during rollouts."""

    # ── Reward ─────────────────────────────────────────────────────────
    goal_radius_m: float = 3.0
    """Episode success if agent ends within this radius of goal (R2R standard SR@3m)."""
    outcome_reward_success: float = 1.0
    """Terminal reward for reaching goal."""
    outcome_reward_fail: float = -0.1
    """Terminal reward for failing. Small negative to prevent reward hacking."""

    navprm_weight: float = 0.3
    """
    Weight for step-level NavPRM reward relative to terminal outcome reward.
    0.0 = pure outcome RL (GQRM baseline)
    0.3 = recommended mix (step rewards accelerate convergence without dominating)
    1.0 = pure step-level (not recommended, outcome reward can vanish)
    """
    step_reward_decay: float = 0.99
    """Discount factor for step rewards. Applied as gamma^(T-t) to terminal reward credit."""

    progress_reward_weight: float = 0.05
    """
    Small shaping reward: distance-to-goal decrease per step.
    Helps early training before NavPRM is fine-tuned. Set 0.0 to disable.
    """

    # ── Training ───────────────────────────────────────────────────────
    learning_rate: float = 1e-6
    """Conservative LR for RL fine-tuning. 10× lower than SFT to prevent catastrophic forgetting."""
    warmup_ratio: float = 0.05
    lr_scheduler: str = "cosine"

    n_epochs: int = 3
    """Number of full passes over the training starting-state pool."""
    batch_size: int = 1
    """Episodes per GPU step. Always 1 for on-policy GRPO (generate → update → next state)."""
    gradient_accumulation: int = 8
    """Effective batch = batch_size × gradient_accumulation × n_gpus = 8 on A100×8."""
    max_grad_norm: float = 1.0
    max_steps: int = 2000
    """Total GRPO policy gradient steps. 2000 steps × 8 rollouts = 16K nav episodes."""

    # ── GRPO reward normalization ───────────────────────────────────────
    reward_normalize: bool = True
    """
    Group-normalize rewards: r_norm = (r - mean(group)) / (std(group) + eps).
    This is the core GRPO innovation vs PPO — no value network needed.
    """
    reward_eps: float = 1e-8
    """Stability epsilon for reward normalization denominator."""
    reward_clip: float = 5.0
    """Clip normalized rewards to [-clip, +clip] to prevent policy collapse."""

    # ── KL regularization ──────────────────────────────────────────────
    kl_coef: float = 0.04
    """
    KL divergence penalty from reference policy (frozen SFT model).
    Prevents RL from drifting too far from the pre-trained distribution.
    L_total = L_policy - kl_coef * KL(π || π_ref)
    """
    kl_update_steps: int = 100
    """How often to recompute reference KL baseline (expensive, so infrequent)."""

    # ── Infrastructure ─────────────────────────────────────────────────
    deepspeed_config: str = "scripts/train/qwenvl_train/zero3_nvme_offload.json"
    output_dir: str = "/workspace/InternNav/data/checkpoints/nexusnav_gqrm"
    log_every: int = 10
    save_every: int = 100
    eval_every: int = 200
    """Run val_unseen evaluation every N steps (expensive — keep infrequent)."""

    seed: int = 42
    bf16: bool = True
    dataloader_num_workers: int = 2

    # ── Environment ─────────────────────────────────────────────────────
    env_type: str = "habitat"
    """'habitat' for R2R-CE discrete, 'internutopia' for VLN-PE continuous."""
    habitat_config: str = "scripts/eval/configs/h1_internvla_n1_async_cfg_orig.py"
    train_split: str = "train"
    val_split: str = "val_unseen"
    max_train_episodes: int = 10739
    """Full R2R train split size."""

    # ── NavPRM candidate selection ───────────────────────────────────────
    navprm_k_candidates: int = 2
    """Number of S2 output candidates to generate and score at inference time."""
    navprm_max_latency_ms: float = 15.0
    """Target latency budget for NavPRM scoring at inference (keeps S2 ≤78ms total)."""

    # ── Derived properties ──────────────────────────────────────────────
    def __post_init__(self) -> None:
        if self.group_size != self.n_rollouts_per_state:
            raise ValueError(
                f"group_size ({self.group_size}) must equal "
                f"n_rollouts_per_state ({self.n_rollouts_per_state})"
            )
        if self.navprm_weight < 0.0 or self.navprm_weight > 1.0:
            raise ValueError(f"navprm_weight must be in [0, 1], got {self.navprm_weight}")

    @property
    def use_navprm(self) -> bool:
        return bool(self.navprm_model_path)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation

    def to_dict(self) -> dict:
        """Serialize to dict for logging / checkpoint metadata."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "GRPOConfig":
        """Load config from YAML file."""
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
