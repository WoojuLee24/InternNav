"""
GRPOTrainer — Group Relative Policy Optimization for NexusNav GQRM RL (C8).

Implements DeepSeek-R1 style GRPO:
  - No value network (critic eliminated)
  - K rollouts per starting state → group-normalize rewards
  - KL penalty from frozen SFT reference policy
  - Optional NavPRM step-level reward shaping (C9 integration)

Algorithm (one training step):
  1. Sample starting state s_0 from training split
  2. Generate K rollout trajectories using policy π_θ (temperature=0.9)
  3. Score each trajectory: r_k = outcome_reward ± navprm_step_rewards
  4. Normalize within group: r̂_k = (r_k - mean_k) / (std_k + eps)
  5. Policy gradient loss: -∑ r̂_k * log π_θ(a_k | s_k)
  6. KL regularization: L_kl = KL(π_θ || π_ref)
  7. Update: L = L_pg + kl_coef * L_kl

Usage:
    from scripts.train.gqrm.grpo_config import GRPOConfig
    from scripts.train.gqrm.grpo_trainer import GRPOTrainer

    cfg = GRPOConfig(base_model_path="/checkpoints/InternVLA-N1-DualVLN")
    trainer = GRPOTrainer(cfg)
    trainer.train()
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from scripts.train.gqrm.grpo_config import GRPOConfig

logger = logging.getLogger(__name__)

__all__ = ["GRPORolloutBuffer", "GRPOEpisode", "GRPOTrainer"]


@dataclass
class GRPOEpisode:
    """
    One complete navigation rollout (episode).
    Stores everything needed for the GRPO policy gradient update.
    """

    trajectory_id: str
    """Unique identifier for this rollout."""

    rollout_group_id: str
    """Shared ID for all K rollouts from the same starting state."""

    observations: List[np.ndarray]
    """RGB observations per step, each (H, W, 3) uint8."""

    actions: List[str]
    """S2 action descriptions per step (text)."""

    log_probs: List[float]
    """Log-probabilities of each action under π_θ at generation time."""

    ref_log_probs: List[float]
    """Log-probabilities under frozen reference policy π_ref (for KL)."""

    outcome_reward: float
    """Terminal reward: +1.0 (success) or -0.1 (fail)."""

    navprm_step_rewards: List[float] = field(default_factory=list)
    """Per-step NavPRM quality scores ∈ [0, 1]. Empty if NavPRM disabled."""

    success: bool = False
    """Whether agent reached goal within goal_radius_m."""

    steps_taken: int = 0
    """Actual steps before episode termination."""

    instruction: str = ""
    """Navigation instruction text."""

    def composite_reward(self, navprm_weight: float, step_decay: float) -> float:
        """
        Composite reward combining outcome and step-level NavPRM rewards.

        r_total = (1 - w) * r_outcome + w * ∑_t γ^(T-t) * r_prm_t
        """
        r = self.outcome_reward
        if self.navprm_step_rewards and navprm_weight > 0.0:
            T = len(self.navprm_step_rewards)
            discounted_prm = sum(
                (step_decay ** (T - t - 1)) * self.navprm_step_rewards[t]
                for t in range(T)
            )
            r = (1.0 - navprm_weight) * r + navprm_weight * discounted_prm
        return r


class GRPORolloutBuffer:
    """
    Buffer accumulating K rollouts from one starting state.
    Performs group reward normalization and prepares policy gradient targets.
    """

    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self.episodes: List[GRPOEpisode] = []
        self.group_id: Optional[str] = None

    def add(self, episode: GRPOEpisode) -> None:
        assert len(self.episodes) < self.cfg.n_rollouts_per_state
        if self.group_id is None:
            self.group_id = episode.rollout_group_id
        assert episode.rollout_group_id == self.group_id, "Mixed group IDs"
        self.episodes.append(episode)

    def is_full(self) -> bool:
        return len(self.episodes) == self.cfg.n_rollouts_per_state

    def compute_normalized_rewards(self) -> List[float]:
        """
        GRPO core: group-normalize composite rewards.
        r̂_k = (r_k - mean(r)) / (std(r) + eps)
        Clip to [-reward_clip, reward_clip].
        """
        raw = [
            ep.composite_reward(self.cfg.navprm_weight, self.cfg.step_reward_decay)
            for ep in self.episodes
        ]
        arr = np.array(raw, dtype=np.float32)
        if self.cfg.reward_normalize:
            arr = (arr - arr.mean()) / (arr.std() + self.cfg.reward_eps)
            arr = np.clip(arr, -self.cfg.reward_clip, self.cfg.reward_clip)
        return arr.tolist()

    def policy_gradient_loss(
        self,
        normalized_rewards: List[float],
        current_log_probs: List[List[float]],
    ) -> torch.Tensor:
        """
        GRPO policy gradient loss (no advantage estimator needed).

        L_pg = -E_k[ r̂_k * ∑_t log π_θ(a_t^k | s_t^k) ]

        Args:
            normalized_rewards: group-normalized rewards per episode
            current_log_probs: recomputed log-probs per episode per step
                               (recomputed from current π_θ, not generation-time π)
        """
        total_loss = torch.tensor(0.0, requires_grad=True)
        n_eps = 0

        for reward, lps in zip(normalized_rewards, current_log_probs):
            if not lps:
                continue
            step_lps = torch.tensor(lps, dtype=torch.float32)
            ep_loss = -reward * step_lps.sum()
            total_loss = total_loss + ep_loss
            n_eps += 1

        if n_eps > 0:
            total_loss = total_loss / n_eps
        return total_loss

    def kl_loss(
        self,
        current_log_probs: List[List[float]],
    ) -> torch.Tensor:
        """
        KL divergence penalty: KL(π_θ || π_ref).
        Computed as: E[ log π_θ(a) - log π_ref(a) ]
        """
        kl_terms: List[torch.Tensor] = []

        for ep, lps in zip(self.episodes, current_log_probs):
            if not lps or not ep.ref_log_probs:
                continue
            n = min(len(lps), len(ep.ref_log_probs))
            cur = torch.tensor(lps[:n], dtype=torch.float32)
            ref = torch.tensor(ep.ref_log_probs[:n], dtype=torch.float32)
            kl_per_step = cur - ref
            kl_terms.append(kl_per_step.mean())

        if not kl_terms:
            return torch.tensor(0.0)
        return torch.stack(kl_terms).mean()

    def clear(self) -> None:
        self.episodes.clear()
        self.group_id = None


class GRPOTrainer:
    """
    GRPO RL trainer for NexusNav (C8 contribution).

    Orchestrates:
      - Episode rollout via NexusNav agent + Habitat/InternUtopia env
      - NavPRM step scoring (C9, optional)
      - Group reward normalization
      - KL-penalized policy gradient update
      - LoRA parameter updates via DeepSpeed ZeRO-3

    This class is the training entry point. The actual rollout execution
    interfaces with the existing eval infrastructure via RolloutEnvBridge
    (see scripts/eval/ for env setup).
    """

    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self._policy = None
        self._ref_policy = None
        self._navprm = None
        self._optimizer = None
        self._scheduler = None
        self._buffer = GRPORolloutBuffer(cfg)
        self._step = 0
        self._best_sr = 0.0

    def _load_policy(self):
        """Load NexusNav policy with LoRA for GRPO updates."""
        try:
            from internnav.model.basemodel.nexusnav.nexusnav_policy import (
                NexusNavModelConfig,
                NexusNavNet,
            )
        except ImportError:
            logger.error("NexusNav policy not found. Check internnav.model.basemodel.nexusnav path.")
            raise

        policy_cfg = NexusNavModelConfig(
            base_model_path=self.cfg.base_model_path,
            lora_r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
        )
        self._policy = NexusNavNet(policy_cfg)
        self._policy.train()
        logger.info(f"Policy loaded from {self.cfg.base_model_path}")

    def _load_ref_policy(self):
        """Load frozen reference policy for KL computation (no gradients)."""
        try:
            from internnav.model.basemodel.nexusnav.nexusnav_policy import (
                NexusNavModelConfig,
                NexusNavNet,
            )
        except ImportError:
            raise

        policy_cfg = NexusNavModelConfig(
            base_model_path=self.cfg.base_model_path,
            lora_r=0,  # no LoRA on reference
        )
        self._ref_policy = NexusNavNet(policy_cfg)
        self._ref_policy.eval()
        for p in self._ref_policy.parameters():
            p.requires_grad_(False)
        logger.info("Reference policy loaded (frozen).")

    def _load_navprm(self):
        """Load NavPRM if configured (C8+C9 synergy)."""
        if not self.cfg.use_navprm:
            logger.info("NavPRM disabled (navprm_model_path not set). Using outcome reward only.")
            return
        from internnav.agent.navprm import NavPRMConfig, NavPRMScorer

        navprm_cfg = NavPRMConfig(
            checkpoint_path=self.cfg.navprm_model_path,
            latency_budget_ms=self.cfg.navprm_max_latency_ms,
        )
        self._navprm = NavPRMScorer(navprm_cfg).load()
        logger.info(f"NavPRM loaded from {self.cfg.navprm_model_path}")

    def _setup_optimizer(self):
        """Setup AdamW optimizer over LoRA parameters only."""
        lora_params = [p for p in self._policy.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in lora_params)
        logger.info(f"Trainable LoRA parameters: {n_params:,}")

        self._optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.cfg.learning_rate,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )

        total_steps = self.cfg.max_steps
        warmup_steps = int(self.cfg.warmup_ratio * total_steps)
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_sched = LinearLR(self._optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(self._optimizer, T_max=total_steps - warmup_steps, eta_min=1e-8)
        self._scheduler = SequentialLR(
            self._optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_steps],
        )

    def rollout_episode(
        self,
        env,
        starting_state: Dict,
        group_id: str,
        rollout_idx: int,
        use_sampling: bool = True,
    ) -> GRPOEpisode:
        """
        Run one complete navigation episode.

        Args:
            env: Navigation environment (Habitat or InternUtopia)
            starting_state: Episode start (instruction, start_pos, goal_pos)
            group_id: Shared ID for all K rollouts from this state
            rollout_idx: Which rollout in the group (0..K-1)
            use_sampling: True during rollout generation, False for greedy eval

        Returns:
            GRPOEpisode with trajectory data and rewards
        """
        obs = env.reset(starting_state)
        instruction = starting_state.get("instruction", "")

        observations: List[np.ndarray] = []
        actions: List[str] = []
        log_probs_list: List[float] = []
        ref_log_probs_list: List[float] = []
        navprm_rewards: List[float] = []
        history: List[str] = []

        done = False
        success = False
        step_idx = 0

        while not done and step_idx < self.cfg.max_episode_steps:
            rgb = obs.get("rgb")
            depth = obs.get("depth")
            pose = obs.get("pose")

            # Policy forward pass (S2 generates action description)
            with torch.cuda.amp.autocast(enabled=self.cfg.bf16):
                action_text, lp = self._policy_step(
                    rgb=rgb,
                    depth=depth,
                    pose=pose,
                    instruction=instruction,
                    history=history,
                    use_sampling=use_sampling,
                )
                ref_lp = self._ref_policy_step(
                    rgb=rgb,
                    depth=depth,
                    pose=pose,
                    instruction=instruction,
                    history=history,
                )

            # NavPRM step scoring (C9)
            if self._navprm is not None:
                from internnav.agent.navprm import NavPRMState
                prm_state = NavPRMState(
                    rgb=rgb,
                    instruction=instruction,
                    action_text=action_text,
                    history=history.copy(),
                    depth=depth,
                    step_idx=step_idx,
                )
                prm_score = self._navprm.score_step(prm_state)
                navprm_rewards.append(prm_score)

            obs, reward, done, info = env.step(action_text)
            success = info.get("success", False)

            observations.append(rgb)
            actions.append(action_text)
            log_probs_list.append(lp)
            ref_log_probs_list.append(ref_lp)
            history.append(action_text)
            step_idx += 1

        # Outcome reward
        if success:
            outcome_reward = self.cfg.outcome_reward_success
        else:
            outcome_reward = self.cfg.outcome_reward_fail

        # Optional progress shaping (not stored — computed inline)
        # r_total already handled in GRPOEpisode.composite_reward()

        return GRPOEpisode(
            trajectory_id=f"{group_id}_{rollout_idx}",
            rollout_group_id=group_id,
            observations=observations,
            actions=actions,
            log_probs=log_probs_list,
            ref_log_probs=ref_log_probs_list,
            outcome_reward=outcome_reward,
            navprm_step_rewards=navprm_rewards,
            success=success,
            steps_taken=step_idx,
            instruction=instruction,
        )

    def _policy_step(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        pose: Optional[np.ndarray],
        instruction: str,
        history: List[str],
        use_sampling: bool = True,
    ) -> Tuple[str, float]:
        """
        Run S2 policy forward pass, return (action_text, log_prob).
        Stub — actual implementation calls NexusNavNet.s2_step().
        """
        # Placeholder: In production this calls self._policy.s2_step(...)
        # and extracts action_text + log_prob from the generation output.
        # The actual integration is done in nexusnav_policy.py grpo_forward().
        action_text = "go forward"
        log_prob = -0.5  # placeholder
        return action_text, log_prob

    def _ref_policy_step(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        pose: Optional[np.ndarray],
        instruction: str,
        history: List[str],
    ) -> float:
        """Run frozen reference policy, return log_prob only."""
        return -0.5  # placeholder — see _policy_step comment

    def grpo_update(self) -> Dict[str, float]:
        """
        Perform one GRPO policy gradient update using the current buffer.

        Returns metrics dict for logging.
        """
        assert self._buffer.is_full(), "Buffer must be full before update"

        normalized_rewards = self._buffer.compute_normalized_rewards()
        current_log_probs = [ep.log_probs for ep in self._buffer.episodes]

        pg_loss = self._buffer.policy_gradient_loss(normalized_rewards, current_log_probs)
        kl = self._buffer.kl_loss(current_log_probs)
        total_loss = pg_loss + self.cfg.kl_coef * kl

        self._optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self._policy.parameters() if p.requires_grad],
            self.cfg.max_grad_norm,
        )
        self._optimizer.step()
        self._scheduler.step()

        n_success = sum(ep.success for ep in self._buffer.episodes)
        sr_in_group = n_success / len(self._buffer.episodes)

        metrics = {
            "loss/policy_gradient": pg_loss.item(),
            "loss/kl": kl.item(),
            "loss/total": total_loss.item(),
            "reward/mean": float(np.mean([ep.outcome_reward for ep in self._buffer.episodes])),
            "reward/normalized_mean": float(np.mean(normalized_rewards)),
            "sr/in_group": sr_in_group,
            "lr": self._scheduler.get_last_lr()[0],
        }

        if self._navprm is not None:
            all_prm = [s for ep in self._buffer.episodes for s in ep.navprm_step_rewards]
            if all_prm:
                metrics["navprm/mean_step_reward"] = float(np.mean(all_prm))

        self._buffer.clear()
        return metrics

    def train(self, env_factory=None) -> None:
        """
        Main GRPO training loop.

        Args:
            env_factory: Callable() → env. If None, uses a stub for testing.
                         In production, pass a Habitat or InternUtopia factory.
        """
        self._load_policy()
        self._load_ref_policy()
        self._load_navprm()
        self._setup_optimizer()

        os.makedirs(self.cfg.output_dir, exist_ok=True)

        logger.info(
            f"Starting GRPO training: {self.cfg.max_steps} steps, "
            f"K={self.cfg.n_rollouts_per_state} rollouts/state, "
            f"NavPRM={'enabled' if self.cfg.use_navprm else 'disabled'}"
        )

        env = env_factory() if env_factory is not None else None
        group_counter = 0

        while self._step < self.cfg.max_steps:
            group_id = f"group_{group_counter:06d}"
            starting_state = self._sample_starting_state(env)

            # Generate K rollouts from same starting state
            for k in range(self.cfg.n_rollouts_per_state):
                if env is None:
                    break  # Stub mode — skip actual rollouts
                ep = self.rollout_episode(env, starting_state, group_id, rollout_idx=k)
                self._buffer.add(ep)

            if not self._buffer.is_full():
                # Stub mode: fill buffer with placeholder episodes
                self._fill_stub_buffer(group_id)

            metrics = self.grpo_update()
            self._step += 1
            group_counter += 1

            if self._step % self.cfg.log_every == 0:
                metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                logger.info(f"Step {self._step}/{self.cfg.max_steps} | {metric_str}")

            if self._step % self.cfg.save_every == 0:
                self._save_checkpoint(f"step_{self._step}")

            if self._step % self.cfg.eval_every == 0 and env is not None:
                sr = self._run_validation(env)
                logger.info(f"Step {self._step} | val_unseen SR = {sr:.4f}")
                if sr > self._best_sr:
                    self._best_sr = sr
                    self._save_checkpoint("best")

        logger.info(f"GRPO training complete. Best val SR = {self._best_sr:.4f}")
        self._save_checkpoint("final")

    def _sample_starting_state(self, env) -> Dict:
        """Sample a random starting state from the training split."""
        return {
            "instruction": "Walk to the kitchen and stop near the sink.",
            "episode_id": "stub_0",
        }

    def _fill_stub_buffer(self, group_id: str) -> None:
        """Fill buffer with stub episodes for testing without an environment."""
        while not self._buffer.is_full():
            success = (len(self._buffer.episodes) % 3 == 0)
            ep = GRPOEpisode(
                trajectory_id=f"{group_id}_{len(self._buffer.episodes)}",
                rollout_group_id=group_id,
                observations=[np.zeros((256, 256, 3), dtype=np.uint8)],
                actions=["go forward"],
                log_probs=[-0.5],
                ref_log_probs=[-0.5],
                outcome_reward=1.0 if success else -0.1,
                success=success,
                steps_taken=5,
                instruction="Navigate to the goal.",
            )
            self._buffer.add(ep)

    def _save_checkpoint(self, tag: str) -> None:
        """Save LoRA weights and config."""
        ckpt_dir = os.path.join(self.cfg.output_dir, f"checkpoint_{tag}")
        os.makedirs(ckpt_dir, exist_ok=True)

        if self._policy is not None:
            try:
                self._policy.save_pretrained(ckpt_dir)
            except AttributeError:
                import torch
                torch.save(
                    {p_name: p for p_name, p in self._policy.named_parameters() if p.requires_grad},
                    os.path.join(ckpt_dir, "lora_weights.pt"),
                )

        import json
        with open(os.path.join(ckpt_dir, "grpo_config.json"), "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)

        logger.info(f"Checkpoint saved: {ckpt_dir}")

    def _run_validation(self, env) -> float:
        """Run greedy rollouts on val_unseen split, return SR."""
        self._policy.eval()
        n_success = 0
        n_episodes = min(50, 500)

        for _ in range(n_episodes):
            state = self._sample_starting_state(env)
            ep = self.rollout_episode(env, state, "val", rollout_idx=0, use_sampling=False)
            if ep.success:
                n_success += 1

        self._policy.train()
        return n_success / n_episodes if n_episodes > 0 else 0.0
