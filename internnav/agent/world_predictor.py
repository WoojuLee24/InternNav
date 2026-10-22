"""
World Prediction Module for System 2 (S2).

From Prophet paper (2511.20633):
- Predict future RGB frames given subgoal action sequence
- Use predicted frames to validate subgoal reachability BEFORE committing
- Lightweight: MobileNetV3-Small feature predictor (not full video diffusion)
- Predicted frames serve as additional context for S2 instruction grounding

Architecture:
    current_frame [3,H,W] + delta_pose [3] → predicted_frame_features [256]
    predicted_features → reachability_score ∈ [0,1]

Usage:
    wp = WorldPredictor()
    score, pred_features = wp.predict_reachability(rgb_frame, delta_pose)
    if score > 0.5:
        commit_subgoal(subgoal)
    else:
        wp.update(rgb_frame, delta_pose, reward=0)  # negative feedback
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FeaturePredictor(nn.Module):
    """
    Lightweight next-frame feature predictor.
    Input: current_features [B, 256] + delta_pose [B, 3]
    Output: predicted_next_features [B, 256]
    """

    def __init__(self, feature_dim: int = 256, pose_dim: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim + pose_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, feature_dim),
        )
        self.reachability_head = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, current_features: torch.Tensor, delta_pose: torch.Tensor):
        x = torch.cat([current_features, delta_pose], dim=-1)
        predicted = self.encoder(x)
        # Reachability: compare predicted future to current (large delta = harder to reach)
        combined = torch.cat([current_features, predicted], dim=-1)
        score = self.reachability_head(combined)
        return predicted, score


class _FrameEncoder(nn.Module):
    """
    Compact frame encoder: downsample + linear projection to 256-d.
    Avoids full CNN forward for speed (just average pooling + linear).
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.proj = nn.Linear(3 * 8 * 8, feature_dim)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        x = self.pool(frame.float() / 255.0)
        x = x.flatten(1)
        return self.proj(x)


class WorldPredictor:
    """
    Lightweight world model for S2 subgoal validation.

    The predictor estimates whether a proposed subgoal (delta_pose) is
    reachable from the current frame, WITHOUT running the full VLM.

    Threshold: reachability_score > 0.5 → accept subgoal

    Training: updated online via Prophet-style reward signals from S1 outcomes.
    """

    def __init__(self, feature_dim: int = 256, device: str = "cpu", learning_rate: float = 1e-4):
        self.feature_dim = feature_dim
        self.device = device
        self.learning_rate = learning_rate
        self._encoder = _FrameEncoder(feature_dim).to(device)
        self._predictor = _FeaturePredictor(feature_dim).to(device)
        self._optimizer = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._predictor.parameters()),
            lr=learning_rate)
        self._replay: List = []
        self._stats = {"accepted": 0, "rejected": 0, "updates": 0}

    @torch.no_grad()
    def predict_reachability(
        self,
        rgb_frame: torch.Tensor,
        delta_pose: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        """
        Predict if delta_pose is reachable from rgb_frame.

        Args:
            rgb_frame: [3,H,W] or [1,3,H,W] uint8 or float tensor
            delta_pose: [3] (dx, dy, dtheta) in robot frame

        Returns:
            (reachability_score: float in [0,1], predicted_features: [256])
        """
        self._encoder.eval()
        self._predictor.eval()

        if rgb_frame.device.type != self.device:
            rgb_frame = rgb_frame.to(self.device)

        features = self._encoder(rgb_frame)
        delta = delta_pose.float().to(self.device).unsqueeze(0)
        pred_features, score = self._predictor(features, delta)

        score_val = score.item()
        if score_val > 0.5:
            self._stats["accepted"] += 1
        else:
            self._stats["rejected"] += 1

        return score_val, pred_features.squeeze(0)

    def update(
        self,
        rgb_frame: torch.Tensor,
        delta_pose: torch.Tensor,
        reward: float,
        max_replay: int = 64,
    ):
        """
        Online update via Prophet-style reward.

        Args:
            rgb_frame: frame at the time of the subgoal decision
            delta_pose: the proposed delta that was executed
            reward: 1.0 = reached goal, 0.0 = failed/collision
        """
        self._replay.append((rgb_frame.cpu(), delta_pose.cpu(), reward))
        if len(self._replay) > max_replay:
            self._replay = self._replay[-max_replay:]

        if len(self._replay) < 8:
            return  # wait for enough examples

        # Mini-batch update (last 8)
        batch = self._replay[-8:]
        frames = torch.stack([b[0] for b in batch]).to(self.device)
        deltas = torch.stack([b[1].float() for b in batch]).to(self.device)
        rewards = torch.tensor([b[2] for b in batch], device=self.device).float().unsqueeze(1)

        self._encoder.train()
        self._predictor.train()
        self._optimizer.zero_grad()

        features = self._encoder(frames)
        _, scores = self._predictor(features, deltas)
        loss = F.binary_cross_entropy(scores, rewards)
        loss.backward()
        self._optimizer.step()
        self._stats["updates"] += 1

    def stats_str(self) -> str:
        total = self._stats["accepted"] + self._stats["rejected"]
        accept_rate = self._stats["accepted"] / total if total > 0 else 0.0
        return (f"WorldPredictor: accept={accept_rate:.1%} ({self._stats['accepted']}/{total}), "
                f"online_updates={self._stats['updates']}")

    def save(self, path: str):
        torch.save({
            "encoder": self._encoder.state_dict(),
            "predictor": self._predictor.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self._encoder.load_state_dict(ckpt["encoder"])
        self._predictor.load_state_dict(ckpt["predictor"])
