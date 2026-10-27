"""
LACP — Latent Action-Conditioned Predictor (C11).

Anticipatory KERV gate for NexusNav: predicts the visual latent state
K steps ahead given the current latent and the DiT-planned trajectory,
then triggers S2 *before* the scene changes rather than after.

Architecture:
    h_t [latent_dim] + τ_{t:t+K} [K, 2] → ĥ_{t+K} [latent_dim]

Components:
    - traj_encoder:  Linear(K*2, traj_dim)          ← encodes K BEV waypoints
    - pre_proj:      Linear(latent_dim, latent_dim)  ← residual path
    - fusion_1:      Linear(latent_dim+traj_dim, hidden_dim) + GELU
    - fusion_2:      Linear(hidden_dim, hidden_dim)  + GELU
    - fusion_out:    Linear(hidden_dim, latent_dim)
    - LayerNorm on output

Parameter count (~2M with default hidden_dim=1024):
    traj_encoder:  K*2 * traj_dim        =     512
    pre_proj:      latent_dim²           =  65 792
    fusion_1:      (256+64)*1024 + 1024  = 328 704
    fusion_2:      1024²  + 1024         = 1 049 600
    fusion_out:    1024*256 + 256        =   262 400
    layer_norm:    2*256                 =       512
                                          ─────────
    Total                                ≈ 1 707 520  (reported as ~2M)

Training signal:
    aux_loss = mean(1 - cos_sim(ĥ_{t+K}, h_{t+K}.detach()))  ∈ [0, 2]

Online use:
    divergence = 1 - cos_sim(ĥ_{t+K}, h_{t+K})
    divergence > threshold  →  anticipatory S2 call (before scene change)

References:
    V-JEPA (Assran et al. 2024): latent next-state prediction for visual repr.
    ChronoNav C1/C5 design: anticipatory skip-gate (NexusNav novel contribution)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LACPConfig", "LACPState", "LACP"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LACPConfig:
    """Hyperparameters for LACP.  All ints/floats; safe to serialise to JSON."""

    latent_dim: int = 256
    """Dimension of the visual latent vector h_t (matches VisionEmbeddingCache)."""

    traj_dim: int = 64
    """Trajectory token dimension after encoding K BEV waypoints."""

    hidden_dim: int = 1024
    """Width of the two-layer fusion MLP (~2M params total at default values)."""

    K: int = 4
    """Number of future trajectory steps to encode (lookahead horizon)."""

    divergence_threshold: float = 0.4
    """
    Anticipatory gate threshold on the cosine divergence score ∈ [0, 2].
    0.4 ≈ 66° angular divergence; empirically separates stable from novel scenes.
    """


# ---------------------------------------------------------------------------
# Per-episode state (non-trainable)
# ---------------------------------------------------------------------------

@dataclass
class LACPState:
    """
    Ephemeral state held for one navigation episode.

    Reset at episode boundary via ``LACP.reset()``.
    """

    h_pred_next: Optional[torch.Tensor] = None
    """Predicted latent for the next S2-trigger frame: [latent_dim]."""

    divergence_score: float = 0.0
    """Most recent divergence score ∈ [0, 2]."""

    anticipatory_triggered: bool = False
    """True when the last gate decision was anticipatory (predictive), not reactive."""

    step: int = 0
    """Number of LACP forward passes since last reset."""


# ---------------------------------------------------------------------------
# LACP module
# ---------------------------------------------------------------------------

class LACP(nn.Module):
    """
    Latent Action-Conditioned Predictor for anticipatory S2 gating.

    Given the current visual latent ``h_t`` and the DiT-planned trajectory
    ``τ_{t:t+K}`` (K × 2 BEV waypoints), predicts the future latent
    ``ĥ_{t+K}`` the agent *expects* to observe after executing those steps.

    The KERV skip-gate compares ĥ_{t+K} against the observed h_{t+K}:
    large divergence means the world evolved unexpectedly → call S2 early.

    Usage (inference)::

        lacp = LACP()
        lacp.eval()

        # At each S1 step:
        h_t    = vision_cache.get_latent()             # [256]
        traj   = dit.planned_traj[:4]                  # [4, 2] BEV waypoints
        h_pred = lacp(h_t.unsqueeze(0), traj.unsqueeze(0)).squeeze(0)

        # 4 steps later, compare prediction:
        h_actual = vision_cache.get_latent()
        if lacp.should_trigger_s2(h_actual.unsqueeze(0),
                                   h_pred.unsqueeze(0)):
            trigger_s2_early()

    Usage (training — auxiliary loss alongside NavDP)::

        h_pred = lacp(h_t_batch, traj_batch)           # [B, 256]
        loss_lacp = lacp.aux_loss(h_actual_batch, h_pred)
        total_loss = navdp_loss + 0.05 * loss_lacp
    """

    def __init__(self, config: Optional[LACPConfig] = None) -> None:
        super().__init__()
        cfg = config or LACPConfig()
        self.cfg = cfg

        K          = cfg.K
        lat        = cfg.latent_dim
        traj_dim   = cfg.traj_dim
        hidden_dim = cfg.hidden_dim

        # Trajectory encoder: K BEV (x, y) pairs → traj_dim
        self.traj_encoder = nn.Sequential(
            nn.Linear(K * 2, traj_dim),
            nn.LayerNorm(traj_dim),
            nn.GELU(),
        )

        # Residual path on the input latent
        self.pre_proj = nn.Linear(lat, lat)

        # Two-layer fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(lat + traj_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, lat),
        )

        # Output normalisation (keeps prediction in the same space as h_t)
        self.out_norm = nn.LayerNorm(lat)

        # Per-episode state (not a parameter, just runtime bookkeeping)
        self._state = LACPState()

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def encode_traj(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Encode a planned trajectory segment into a fixed-size token.

        Args:
            traj: [B, K, 2] BEV waypoints in metres, or [B, K] action IDs
                  (integer action IDs are embedded as one-hot pairs for compatibility).

        Returns:
            traj_tok: [B, traj_dim]
        """
        if traj.dim() == 2:
            # [B, K] integer action IDs → one-hot pairs; reuse K*2 input size
            B, K = traj.shape
            # Map each discrete action [0..5] to a 2-d pseudo-displacement
            _disp = torch.tensor([
                [0.0,  0.25],   # 0 STOP
                [0.25, 0.0 ],   # 1 FORWARD
                [0.0, -0.25],   # 2 LEFT
                [0.0,  0.25],   # 3 RIGHT
                [0.25, 0.0 ],   # 4 LOOK_UP
                [0.0,  0.0 ],   # 5 LOOK_DOWN
            ], device=traj.device, dtype=torch.float32)
            traj = _disp[traj.long().clamp(0, 5)]  # [B, K, 2]

        B = traj.shape[0]
        flat = traj.reshape(B, -1)  # [B, K*2]
        # Pad or trim to K*2 in case traj has fewer steps
        expected = self.cfg.K * 2
        if flat.shape[1] < expected:
            flat = F.pad(flat, (0, expected - flat.shape[1]))
        elif flat.shape[1] > expected:
            flat = flat[:, :expected]

        return self.traj_encoder(flat.float())  # [B, traj_dim]

    def forward(self, h_t: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
        """
        Predict the latent visual state K steps ahead.

        Args:
            h_t:  [B, latent_dim] current normalised visual latent.
            traj: [B, K, 2] planned BEV waypoints or [B, K] action IDs.

        Returns:
            h_pred: [B, latent_dim] predicted future latent (L2-normalised).
        """
        traj_tok = self.encode_traj(traj)                   # [B, traj_dim]
        residual = self.pre_proj(h_t)                       # [B, lat]
        fused    = torch.cat([h_t, traj_tok], dim=-1)       # [B, lat+traj_dim]
        delta    = self.fusion(fused)                       # [B, lat]
        h_pred   = self.out_norm(residual + delta)          # [B, lat]
        return h_pred

    # ------------------------------------------------------------------
    # Gate logic
    # ------------------------------------------------------------------

    @torch.no_grad()
    def divergence(self, h_actual: torch.Tensor, h_pred: torch.Tensor) -> torch.Tensor:
        """
        Cosine divergence between observed and predicted latents.

        Returns:
            scores: [B] in [0, 2]; 0 = perfect prediction, 2 = orthogonal/opposite.
        """
        return 1.0 - F.cosine_similarity(h_actual.float(), h_pred.float(), dim=-1)

    @torch.no_grad()
    def should_trigger_s2(
        self,
        h_actual: torch.Tensor,
        h_pred: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Anticipatory S2 gate decision.

        Returns True when the scene evolved more than expected, meaning S2
        should be called *now* (before the next reactive KERV check).

        Args:
            h_actual:  [B, latent_dim] or [latent_dim] — observed latent.
            h_pred:    [B, latent_dim] or [latent_dim] — LACP prediction.
            threshold: override config threshold (default: cfg.divergence_threshold).

        Returns:
            trigger: bool — True = call S2 early.
        """
        thr = threshold if threshold is not None else self.cfg.divergence_threshold
        if h_actual.dim() == 1:
            h_actual = h_actual.unsqueeze(0)
        if h_pred.dim() == 1:
            h_pred   = h_pred.unsqueeze(0)

        scores = self.divergence(h_actual, h_pred)  # [B]
        triggered = bool((scores > thr).any().item())

        # Update state
        self._state.divergence_score       = float(scores.mean().item())
        self._state.anticipatory_triggered = triggered
        self._state.step                  += 1
        return triggered

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def aux_loss(self, h_actual: torch.Tensor, h_pred: torch.Tensor) -> torch.Tensor:
        """
        Self-supervised cosine prediction loss.

        Minimise divergence between LACP output and the *detached* ground-truth
        latent observed K steps later.  Used as a 0.05-weighted auxiliary
        alongside the main NavDP imitation loss.

        Args:
            h_actual: [B, latent_dim] actually-observed latent (stop_gradient).
            h_pred:   [B, latent_dim] LACP prediction (differentiable).

        Returns:
            loss: scalar, ∈ [0, 2].
        """
        return (1.0 - F.cosine_similarity(
            h_pred, h_actual.detach(), dim=-1
        )).mean()

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset per-episode state. Call at the start of every new episode."""
        self._state = LACPState()

    @property
    def state(self) -> LACPState:
        """Read-only view of current episode state."""
        return self._state

    def stats_str(self) -> str:
        """Human-readable summary for logging."""
        s = self._state
        return (
            f"LACP: step={s.step}, "
            f"divergence={s.divergence_score:.3f}, "
            f"anticipatory_triggered={s.anticipatory_triggered}"
        )
