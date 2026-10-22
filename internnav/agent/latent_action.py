"""
Latent Action Model — Action Token Vocabulary for System 2.

From ChronoNav Gate 7 / paper Section "S2 Action Tokens":
Instead of S2 producing long text output (subgoal description ~100 tokens),
it outputs 1-3 "action tokens" from a fixed vocabulary of learned embeddings.

Each action token maps to: (subgoal_delta_x, subgoal_delta_y, subgoal_delta_theta)
expressed in the robot's local frame.

Benefits:
- Text decoding reduced from ~100 tokens to 1-3 tokens → ~30× decode speedup
- Vocabulary is finite → KERV-style rejection is exact (not heuristic)
- Tokens are learned to cover the distribution of subgoals in training data

Architecture:
    VQ-VAE style: continuous actions → nearest codebook entry → discrete token
    At inference: S2 decodes 1-3 token IDs → lookup in codebook → (dx, dy, dtheta)

Token vocab size: 256 (covers all navigation primitives: forward/left/right/stop variants)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionTokenVocabulary(nn.Module):
    """
    Fixed-size codebook of navigation subgoal action embeddings.

    Maps discrete token IDs to continuous (dx, dy, dtheta) subgoal targets.
    Trained via VQ-style commitment loss on navigation trajectories.

    Special tokens:
        0: STOP — terminal action (agent should stop)
        1: FORWARD — straight ahead 1.0m
        2..N: learned navigation tokens
    """

    STOP_TOKEN = 0
    FORWARD_TOKEN = 1

    def __init__(self, vocab_size: int = 256, action_dim: int = 3, embed_dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.action_dim = action_dim  # (dx, dy, dtheta)
        self.embed_dim = embed_dim

        # Codebook: vocab_size × embed_dim learned representations
        self.codebook = nn.Embedding(vocab_size, embed_dim)
        # Action encoder: action_dim → embed_dim (inverse of decoder, for VQ commitment)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )
        # Action decoder: embed_dim → action_dim
        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),  # bound actions to [-1, 1] (scaled at use time)
        )
        # Action scales: max displacement and rotation per token
        self.register_buffer("action_scale", torch.tensor([2.0, 2.0, 3.14]))

        # Initialize special tokens
        self._init_special_tokens()

    def _init_special_tokens(self):
        with torch.no_grad():
            # STOP token: zero displacement
            self.codebook.weight[self.STOP_TOKEN].zero_()
            # FORWARD token: unit forward displacement (dx=1, dy=0, dtheta=0)
            fwd = torch.zeros(self.embed_dim)
            fwd[0] = 1.0
            self.codebook.weight[self.FORWARD_TOKEN] = fwd

    def token_to_action(self, token_id: int) -> torch.Tensor:
        """Map a single token ID to (dx, dy, dtheta) in robot frame."""
        emb = self.codebook(torch.tensor(token_id))
        action = self.action_decoder(emb.unsqueeze(0)).squeeze(0)
        return action * self.action_scale

    def tokens_to_actions(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map batch of token IDs [N] to actions [N, 3]."""
        embs = self.codebook(token_ids)
        actions = self.action_decoder(embs)
        return actions * self.action_scale

    def action_to_nearest_token(self, action: torch.Tensor) -> int:
        """
        Quantize a continuous action to the nearest token ID.
        Used during training to build pseudo-labels.
        """
        action_norm = action / self.action_scale
        # Encode action to embedding space
        encoded = action_norm.unsqueeze(0)  # [1, action_dim]
        # Compute distances to all codebook entries via learned decoder inverse
        # Approximation: find embedding closest to decoded(x) == action
        all_decoded = self.action_decoder(self.codebook.weight)  # [V, action_dim]
        dists = ((all_decoded - action_norm.unsqueeze(0)) ** 2).sum(-1)
        return dists.argmin().item()

    def vq_loss(
        self,
        action: torch.Tensor,
        token_id: Optional[torch.Tensor] = None,
        commitment_weight: float = 0.25,
    ) -> Tuple[torch.Tensor, int]:
        """
        VQ-VAE training loss for the action token vocabulary.

        Args:
            action: [B, 3] continuous ground-truth actions
            token_id: [B] pre-assigned token IDs (or None to use nearest)
            commitment_weight: β from VQ-VAE paper

        Returns:
            (loss, nearest_token_id)
        """
        action_norm = action / self.action_scale.to(action.device)

        if token_id is None:
            # Find nearest codebook entry
            with torch.no_grad():
                all_decoded = self.action_decoder(self.codebook.weight)
                dists = torch.cdist(action_norm, all_decoded)
                token_id = dists.argmin(-1)

        # Encode action into embedding space
        encoded = self.action_encoder(action_norm)

        # Embedding and decoded action for nearest token
        emb = self.codebook(token_id)
        decoded = self.action_decoder(emb)

        # Reconstruction loss: decoded token ≈ original action
        recon_loss = F.mse_loss(decoded, action_norm)

        # Commitment loss: encoded action ≈ codebook entry (straight-through)
        commit_loss = F.mse_loss(encoded, emb.detach())

        return recon_loss + commitment_weight * commit_loss, token_id.tolist()

    def save_vocab(self, path: str):
        torch.save({"codebook": self.codebook.state_dict(),
                    "decoder": self.action_decoder.state_dict()}, path)

    def load_vocab(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.codebook.load_state_dict(ckpt["codebook"])
        self.action_decoder.load_state_dict(ckpt["decoder"])


class LatentActionDecoder(nn.Module):
    """
    Decode S2 token sequence into a planned trajectory.

    S2 outputs 1-3 action token IDs → each decoded to a waypoint (dx, dy, dtheta).
    The waypoints form a coarse subgoal path for S1 to follow.

    Token format: [token_id_1, token_id_2, ..., STOP_or_EOS]
    """

    def __init__(self, vocab: ActionTokenVocabulary, max_tokens: int = 3):
        super().__init__()
        self.vocab = vocab
        self.max_tokens = max_tokens

    @torch.no_grad()
    def decode_plan(self, token_ids: List[int]) -> List[torch.Tensor]:
        """
        Convert a list of token IDs into a sequence of waypoints.

        Args:
            token_ids: list of integer token IDs from S2 output

        Returns:
            List of (dx, dy, dtheta) tensors, one per token (STOP excluded)
        """
        waypoints = []
        for tid in token_ids[:self.max_tokens]:
            if tid == ActionTokenVocabulary.STOP_TOKEN:
                break
            action = self.vocab.token_to_action(tid)
            waypoints.append(action)
        return waypoints

    def decode_cumulative_trajectory(
        self, token_ids: List[int]
    ) -> torch.Tensor:
        """
        Decode tokens into a cumulative trajectory in local frame.

        Returns [N, 3] where each row is cumulative (x, y, theta) from start.
        """
        waypoints = self.decode_plan(token_ids)
        if not waypoints:
            return torch.zeros(1, 3)

        traj = torch.stack(waypoints)  # [N, 3]
        # Cumulative sum for position (x, y); heading is local delta
        traj_cum = traj.cumsum(0)
        return traj_cum


def build_default_vocab(
    device: str = "cpu",
    init_from_primitives: bool = True,
) -> ActionTokenVocabulary:
    """
    Build default action token vocabulary with canonical navigation primitives.
    If init_from_primitives=True, first 16 tokens are hand-initialized
    to cover basic navigation moves.
    """
    vocab = ActionTokenVocabulary(vocab_size=256, embed_dim=64).to(device)

    if init_from_primitives:
        primitives = [
            # (dx, dy, dtheta) in robot frame
            (0.0, 0.0, 0.0),    # 0: STOP
            (1.0, 0.0, 0.0),    # 1: FORWARD 1m
            (0.5, 0.0, 0.0),    # 2: FORWARD 0.5m
            (2.0, 0.0, 0.0),    # 3: FORWARD 2m
            (0.0, 0.0, 0.79),   # 4: TURN LEFT 45°
            (0.0, 0.0, 1.57),   # 5: TURN LEFT 90°
            (0.0, 0.0, -0.79),  # 6: TURN RIGHT 45°
            (0.0, 0.0, -1.57),  # 7: TURN RIGHT 90°
            (1.0, 0.0, 0.39),   # 8: FORWARD+LEFT
            (1.0, 0.0, -0.39),  # 9: FORWARD+RIGHT
            (0.5, 0.0, 1.57),   # 10: SHORT+TURN_LEFT
            (0.5, 0.0, -1.57),  # 11: SHORT+TURN_RIGHT
            (1.5, 0.0, 0.0),    # 12: FORWARD 1.5m
            (0.0, 0.0, 3.14),   # 13: U-TURN
            (0.3, 0.0, 0.0),    # 14: CREEP FORWARD
            (3.0, 0.0, 0.0),    # 15: FORWARD 3m
        ]
        with torch.no_grad():
            for i, (dx, dy, dth) in enumerate(primitives):
                action = torch.tensor([dx, dy, dth]) / vocab.action_scale
                # Set embedding to match this action via inverse
                # Simple: use zeros + set decoder input directly
                fwd = torch.zeros(vocab.embed_dim)
                fwd[0] = action[0]; fwd[1] = action[1]; fwd[2] = action[2]
                vocab.codebook.weight[i] = fwd

    return vocab
