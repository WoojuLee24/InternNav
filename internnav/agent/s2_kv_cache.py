"""
S2-KV Cache: prefix KV reuse + O(1) bounded memory for System 2 (async VLM planner).

From ChronoNav paper Section 3.6:
- Skip re-encoding the stable prefix (system prompt + stable history) whose KV is identical
- Bounded FIFO history window (no linspace re-encoding)
- Global memory: evicted frames compress into most-recent-per-place store
- Measured: 94.5% of prefix tokens reusable on stable-history stream
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch


class S2KVCache:
    """
    Prefix KV-cache for System 2.

    Usage:
        cache = S2KVCache(max_history=8, global_memory_size=16)
        prompt, reuse_len = cache.build_prompt(instruction, new_frame_desc, past_kv)
        # pass reuse_len to model.generate() as past_key_values prefix length
        cache.update(new_frame_desc, place_id)
    """

    def __init__(self, max_history: int = 8, global_memory_size: int = 16, system_prompt: str = ""):
        self.max_history = max_history
        self.global_memory_size = global_memory_size
        self.system_prompt = system_prompt
        self._history: List[str] = []
        self._global_mem: OrderedDict = OrderedDict()
        self._last_prefix_hash: Optional[str] = None
        self._last_prefix_len: int = 0
        self._stats: Dict = {"reuse": 0, "miss": 0, "total": 0}

    def _prefix_hash(self, instruction: str) -> str:
        content = self.system_prompt + "|" + instruction + "|" + "|".join(self._history[-self.max_history:])
        return hashlib.md5(content.encode()).hexdigest()

    def build_prompt(
        self,
        instruction: str,
        new_frame_desc: str,
        past_kv=None,
    ) -> Tuple[str, int]:
        """
        Build the S2 prompt and return (prompt_text, reusable_prefix_token_count).

        The reusable_prefix_token_count can be used with HuggingFace's
        `past_key_values` caching: only tokens AFTER the prefix need to be
        computed from scratch.

        Returns:
            prompt_text: full prompt string
            reuse_prefix_len: estimated token count of reusable prefix
                              (0 = cache miss, use full prompt)
        """
        self._stats["total"] += 1

        # Build FIFO history
        history_window = self._history[-self.max_history:]
        global_ctx = self._global_memory_str()

        # Build stable prefix (system + instruction + history + global memory)
        stable_prefix = (
            f"{self.system_prompt}\n"
            f"INSTRUCTION: {instruction}\n"
            f"GLOBAL CONTEXT:\n{global_ctx}\n"
            f"RECENT HISTORY:\n" + "\n".join(history_window)
        )

        # Check if prefix hash matches last call
        prefix_hash = self._prefix_hash(instruction)
        if prefix_hash == self._last_prefix_hash and past_kv is not None:
            self._stats["reuse"] += 1
            reuse_len = self._last_prefix_len
        else:
            self._stats["miss"] += 1
            reuse_len = 0
            self._last_prefix_hash = prefix_hash
            self._last_prefix_len = len(stable_prefix.split())  # rough token estimate

        # Full prompt = stable prefix + current observation
        full_prompt = stable_prefix + f"\nCURRENT OBSERVATION:\n{new_frame_desc}\n\nPLAN:"

        return full_prompt, reuse_len

    def update(self, frame_desc: str, place_id: Optional[str] = None):
        """Add new frame to history. If place_id given, also update global memory."""
        self._history.append(frame_desc)
        # Keep only bounded window in full history
        if len(self._history) > self.max_history * 2:
            self._history = self._history[-self.max_history:]

        if place_id is not None:
            # Most-recent-per-place: overwrite existing entry, keep capacity
            if place_id in self._global_mem:
                del self._global_mem[place_id]
            self._global_mem[place_id] = frame_desc
            while len(self._global_mem) > self.global_memory_size:
                self._global_mem.popitem(last=False)  # evict oldest

    def _global_memory_str(self) -> str:
        if not self._global_mem:
            return "(empty)"
        lines = [f"  [{pid}] {desc[:80]}" for pid, desc in list(self._global_mem.items())[-8:]]
        return "\n".join(lines)

    @property
    def reuse_rate(self) -> float:
        if self._stats["total"] == 0:
            return 0.0
        return self._stats["reuse"] / self._stats["total"]

    def reset(self):
        self._history.clear()
        self._global_mem.clear()
        self._last_prefix_hash = None
        self._last_prefix_len = 0
        self._stats = {"reuse": 0, "miss": 0, "total": 0}

    def stats_str(self) -> str:
        r = self.reuse_rate
        return (f"S2KV reuse={r:.1%} ({self._stats['reuse']}/{self._stats['total']}), "
                f"history={len(self._history)}/{self.max_history}, "
                f"global_mem={len(self._global_mem)}/{self.global_memory_size}")


class VisionEmbeddingCache:
    """
    Cache vision embeddings keyed by frame fingerprint (EMA-smoothed).
    Avoids re-encoding vision tokens when the frame hasn't changed.

    Threshold τ=0.92 (same as temporal cache in ChronoNav).
    """

    def __init__(self, max_size: int = 32, similarity_threshold: float = 0.92, ema_alpha: float = 0.10):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.ema_alpha = ema_alpha
        self._cache: OrderedDict = OrderedDict()
        self._ema_hash: Optional[torch.Tensor] = None
        self._stats: Dict = {"hit": 0, "miss": 0}

    def _fingerprint(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """16:1 spatial fingerprint via average pooling."""
        if frame_tensor.dim() == 4:
            frame_tensor = frame_tensor[0]
        return torch.nn.functional.adaptive_avg_pool2d(
            frame_tensor.float(), (14, 14)).flatten()

    def get_or_compute(
        self,
        frame_tensor: torch.Tensor,
        compute_fn,
    ):
        """
        Return cached embedding if frame is similar to last, else compute.

        Args:
            frame_tensor: [C,H,W] or [1,C,H,W] image tensor
            compute_fn: callable that takes frame_tensor and returns embedding

        Returns:
            (embedding, cache_hit: bool)
        """
        fp = self._fingerprint(frame_tensor)

        if self._ema_hash is not None:
            cos_sim = torch.nn.functional.cosine_similarity(
                fp.unsqueeze(0), self._ema_hash.unsqueeze(0)).item()
            if cos_sim >= self.similarity_threshold:
                # Cache hit — return last embedding
                self._stats["hit"] += 1
                last_emb = next(reversed(self._cache.values()))
                return last_emb, True

        # Cache miss — compute
        self._stats["miss"] += 1
        embedding = compute_fn(frame_tensor)

        # Update EMA fingerprint
        if self._ema_hash is None:
            self._ema_hash = fp.clone()
        else:
            self._ema_hash = (1 - self.ema_alpha) * self._ema_hash + self.ema_alpha * fp

        # Store in LRU cache
        key = str(fp.norm().item())[:8]
        if key in self._cache:
            del self._cache[key]
        self._cache[key] = embedding
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

        return embedding, False

    @property
    def hit_rate(self) -> float:
        total = self._stats["hit"] + self._stats["miss"]
        return self._stats["hit"] / total if total > 0 else 0.0

    def reset(self):
        self._cache.clear()
        self._ema_hash = None
        self._stats = {"hit": 0, "miss": 0}
