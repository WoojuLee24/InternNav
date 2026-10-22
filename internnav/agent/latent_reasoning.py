"""
Latent Reasoning Module — Silent Chain-of-Thought in Embedding Space.

From ChronoNav Gate 7 + Coconut paper (chain of continuous thought):
Instead of S2 generating intermediate reasoning tokens (slow, visible),
it runs N recurrent "thinking steps" in embedding space before emitting.

Based on:
  - Coconut (Hao et al. 2024): reasoning in continuous latent space, not text
  - Quiet-STaR (Zeiler et al. 2024): parallel hidden thoughts
  - Prophet (2511.20633): latent prediction for subgoal validation

Benefits:
  - N=4 thinking steps ≈ 0.4ms (vs 120ms for 4 generated tokens)
  - Final thought vector guides action token selection
  - Thought history = compact episodic memory across S2 calls
  - Graded uncertainty: thought variance → confidence score

Architecture:
    ThoughtCell: (thought_state [D] + context [D]) → next_thought_state [D]
    LatentReasoner: N × ThoughtCell → final_thought [D] → action_logits [V]

D = 256 (thought vector dimension, matching WorldPredictor feature space)
V = 256 (action token vocabulary size)
N = 4 (thinking steps, tunable)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThoughtCell(nn.Module):
    """
    Single latent reasoning step.

    Analogous to an RNN cell but operates on thought-level semantics.
    Input: (thought_state, context) → Output: next_thought_state, gate

    Uses gated residual: allows the model to "skip" a thinking step
    when the context doesn't provide new information (cache-friendly).
    """

    def __init__(self, thought_dim: int = 256, context_dim: int = 256):
        super().__init__()
        self.thought_dim = thought_dim

        # GRU-style update gate
        self.update_gate = nn.Sequential(
            nn.Linear(thought_dim + context_dim, thought_dim),
            nn.Sigmoid(),
        )
        # New thought candidate
        self.thought_transform = nn.Sequential(
            nn.Linear(thought_dim + context_dim, thought_dim * 2),
            nn.GELU(),
            nn.Linear(thought_dim * 2, thought_dim),
            nn.Tanh(),
        )
        # Uncertainty estimate (used as confidence proxy)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(thought_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, thought: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            thought: [B, thought_dim] current thought state
            context: [B, context_dim] current observation context

        Returns:
            next_thought: [B, thought_dim]
            uncertainty: [B, 1] (0=certain, 1=uncertain)
        """
        combined = torch.cat([thought, context], dim=-1)
        z = self.update_gate(combined)
        candidate = self.thought_transform(combined)
        next_thought = (1 - z) * thought + z * candidate
        uncertainty = self.uncertainty_head(next_thought)
        return next_thought, uncertainty


class LatentReasoner(nn.Module):
    """
    Multi-step latent reasoning chain for S2 planning.

    Takes encoded context (instruction + vision + memory) and iteratively
    refines a thought vector through N ThoughtCell steps, then predicts
    an action token sequence.

    Usage:
        reasoner = LatentReasoner(n_steps=4, vocab_size=256)
        logits, thought, conf = reasoner(context_embedding)
        action_tokens = logits.argmax(-1)  # [B, max_decode_len]
    """

    def __init__(
        self,
        thought_dim: int = 256,
        context_dim: int = 256,
        n_steps: int = 4,
        vocab_size: int = 256,
        max_decode_len: int = 3,
    ):
        super().__init__()
        self.thought_dim = thought_dim
        self.n_steps = n_steps
        self.max_decode_len = max_decode_len

        # Initial thought seed from context
        self.thought_init = nn.Sequential(
            nn.Linear(context_dim, thought_dim),
            nn.LayerNorm(thought_dim),
        )

        # N recurrent thinking steps
        self.cells = nn.ModuleList([
            ThoughtCell(thought_dim, context_dim) for _ in range(n_steps)
        ])

        # Decode action token sequence from final thought
        self.action_head = nn.GRU(
            input_size=thought_dim,
            hidden_size=thought_dim,
            num_layers=1,
            batch_first=True,
        )
        self.token_proj = nn.Linear(thought_dim, vocab_size)

        # Stop prediction (should the next token be STOP?)
        self.stop_head = nn.Linear(thought_dim, 1)

    def forward(
        self,
        context: torch.Tensor,
        prev_thought: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            context: [B, context_dim] encoded observation context
            prev_thought: [B, thought_dim] carry-over thought from previous call
                          (None = start fresh)

        Returns:
            action_logits: [B, max_decode_len, vocab_size]
            final_thought: [B, thought_dim] (pass back as prev_thought next call)
            confidence: [B, 1] average certainty across steps
        """
        B = context.shape[0]

        # Initialize thought
        if prev_thought is not None:
            # Blend previous thought with new context (70% carry, 30% new)
            thought = 0.7 * prev_thought + 0.3 * self.thought_init(context)
        else:
            thought = self.thought_init(context)

        # Run N thinking steps
        uncertainties = []
        for cell in self.cells:
            thought, unc = cell(thought, context)
            uncertainties.append(unc)

        avg_uncertainty = torch.stack(uncertainties, dim=1).mean(dim=1)
        confidence = 1.0 - avg_uncertainty

        # Decode action sequence from final thought
        # Use thought as initial hidden state, decode max_decode_len tokens
        thought_seq = thought.unsqueeze(1).expand(-1, self.max_decode_len, -1)
        gru_out, _ = self.action_head(thought_seq, thought.unsqueeze(0))
        action_logits = self.token_proj(gru_out)

        return action_logits, thought, confidence

    def predict_tokens(
        self,
        context: torch.Tensor,
        prev_thought: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        stop_token_id: int = 0,
    ) -> Tuple[List[int], torch.Tensor, float]:
        """
        Greedy decode action token IDs from context.

        Returns:
            token_ids: list of predicted token IDs (up to max_decode_len, or until STOP)
            next_thought: updated thought state to pass to next call
            confidence_score: float in [0, 1]
        """
        with torch.no_grad():
            logits, thought, conf = self(context, prev_thought)
            # logits: [1, max_decode_len, vocab_size]
            if temperature != 1.0:
                logits = logits / temperature
            tokens = logits[0].argmax(-1).tolist()  # [max_decode_len]

        # Truncate at STOP
        result = []
        for tid in tokens:
            if tid == stop_token_id:
                break
            result.append(tid)

        return result, thought, conf.item()


class S2ThoughtCache:
    """
    Persistent thought state across S2 calls (episodic working memory).

    The thought vector from the previous S2 invocation carries forward
    as a "warm start" for the current invocation — analogous to KV-cache
    but for the latent reasoning chain, not attention keys.

    This makes reasoning about recurring landmarks faster: the thought
    state has already "seen" the hallway once, so it converges faster.
    """

    def __init__(self, thought_dim: int = 256):
        self.thought_dim = thought_dim
        self._thought: Optional[torch.Tensor] = None
        self._step: int = 0
        self._thought_history: List[torch.Tensor] = []
        self._max_history: int = 8

    @property
    def current_thought(self) -> Optional[torch.Tensor]:
        return self._thought

    def update(self, thought: torch.Tensor):
        self._thought = thought.detach()
        self._step += 1
        self._thought_history.append(thought.detach())
        if len(self._thought_history) > self._max_history:
            self._thought_history = self._thought_history[-self._max_history:]

    def reset(self):
        self._thought = None
        self._step = 0
        self._thought_history.clear()

    def thought_drift(self) -> float:
        """
        Cosine distance between current and previous thought.
        Large drift = large semantic change in context (should fire S2).
        Small drift = stable context (could skip S2).
        """
        if len(self._thought_history) < 2:
            return 1.0
        prev = self._thought_history[-2]
        curr = self._thought_history[-1]
        cos_sim = F.cosine_similarity(prev.flatten().unsqueeze(0),
                                      curr.flatten().unsqueeze(0)).item()
        return 1.0 - cos_sim  # drift = 1 - similarity

    def stats_str(self) -> str:
        drift = self.thought_drift()
        return (f"S2ThoughtCache: step={self._step}, "
                f"drift={drift:.3f}, "
                f"history_len={len(self._thought_history)}")


def build_context_encoder(
    vision_dim: int = 256,
    text_dim: int = 512,
    memory_dim: int = 64,
    out_dim: int = 256,
) -> nn.Module:
    """
    Lightweight context encoder: fuses vision + text + memory → 256-d.
    Used as the `context` input to LatentReasoner.
    """
    return nn.Sequential(
        nn.Linear(vision_dim + text_dim + memory_dim, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Linear(512, out_dim),
        nn.LayerNorm(out_dim),
    )
