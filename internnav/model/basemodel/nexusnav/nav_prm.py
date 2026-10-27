"""NavPRM: Navigation Process Reward Model for NexusNav.

NexusNav contribution C9. A 300M-parameter model that scores intermediate
navigation steps, enabling K=2 best-of-N candidate selection at inference.
First process reward model for VLN (CVPR 2026 novelty: HIGHEST — zero VLN PRM papers).

STATUS: ARCHITECTURE STUB — requires Naver Cloud A100x4 cluster + 8h training.
To activate:
  1. Provide A100x4 environment
  2. Run: python scripts/train/nav_prm_train.py --model nav_prm_300m
  3. Set PRM_CHECKPOINT in NexusNavModelConfig
  4. Set PRM_K_CANDIDATES = 2 for inference

RESOURCE REQUIREMENTS:
  - GPU: 4x A100 80GB (300M model fits on 1, faster training on 4)
  - Training time: ~8h
  - Data: VLN-CE R2R train split with step-level success annotations
  - Architecture: BERT-base (110M) + nav-specific head (190M projection + MLP)
"""

import torch
import torch.nn as nn

# === ACTIVATION BLOCK — uncomment when A100x4 available ===
# from transformers import BertModel, BertConfig
#
# PRM_HIDDEN = 768         # BERT-base hidden dim
# PRM_NAV_DIM = 1024       # nav feature projection dim
# PRM_SCORE_DIM = 256      # scoring head dim
#
# class NavPRM(nn.Module):
#     """300M navigation process reward model.
#
#     Scores the quality of a navigation step given:
#     - Current observation (visual features, 768-dim)
#     - Instruction tokens (BERT-encoded)
#     - Predicted next waypoint (3-dim: dx, dy, dtheta)
#     - Step index (position in trajectory)
#
#     Output: scalar score in [0, 1] representing step quality.
#     """
#     def __init__(self):
#         super().__init__()
#         self.text_encoder = BertModel(BertConfig())   # 110M params
#         self.vis_proj = nn.Linear(768, PRM_NAV_DIM)
#         self.wp_proj = nn.Linear(3, PRM_NAV_DIM)
#         self.step_embed = nn.Embedding(100, 64)
#         self.fusion = nn.Sequential(
#             nn.Linear(PRM_NAV_DIM * 2 + PRM_HIDDEN + 64, PRM_SCORE_DIM * 4),
#             nn.GELU(),
#             nn.Linear(PRM_SCORE_DIM * 4, PRM_SCORE_DIM),
#             nn.GELU(),
#             nn.Linear(PRM_SCORE_DIM, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, vis_feats, input_ids, attention_mask, waypoint, step_idx):
#         text_out = self.text_encoder(input_ids, attention_mask).pooler_output  # (B, 768)
#         vis_out = self.vis_proj(vis_feats.mean(dim=1))                          # (B, PRM_NAV_DIM)
#         wp_out = self.wp_proj(waypoint)                                          # (B, PRM_NAV_DIM)
#         step_out = self.step_embed(step_idx)                                     # (B, 64)
#         fused = torch.cat([vis_out, wp_out, text_out, step_out], dim=-1)
#         return self.fusion(fused).squeeze(-1)                                    # (B,)
#
# class NavPRMSelector:
#     """K=2 candidate selection using NavPRM scores at inference."""
#     def __init__(self, prm: "NavPRM", k_candidates: int = 2):
#         self.prm = prm
#         self.k_candidates = k_candidates
#
#     @torch.no_grad()
#     def select_best(self, candidates, vis_feats, input_ids, attention_mask, step_idx):
#         scores = []
#         for cand in candidates:
#             wp_tensor = torch.tensor(cand['waypoint']).unsqueeze(0)
#             score = self.prm(vis_feats, input_ids, attention_mask, wp_tensor, step_idx)
#             scores.append(score.item())
#         best_idx = scores.index(max(scores))
#         return candidates[best_idx], scores
# === END ACTIVATION BLOCK ===


class NavPRMStub:
    """Placeholder — raises informative error if called without activation."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "NavPRM (C9) requires Naver A100x4 cluster. "
            "See RESOURCE REQUIREMENTS at top of nav_prm.py to activate."
        )
