"""Task memory: tracks instruction progress through segments. (T²-Nav / DGNav)"""
from __future__ import annotations
from typing import List, Optional


class TaskMemory:
    def __init__(self):
        self._segments: List[str] = []
        self._done: List[bool] = []
        self._ptr: int = 0
        self._landmark: Optional[str] = None

    def reset(self, instruction_segments: List[str]) -> None:
        self._segments = list(instruction_segments)
        self._done = [False] * len(instruction_segments)
        self._ptr = 0
        self._landmark = None

    def update(self, obs_text: str, instruction_segments: List[str]) -> None:
        if not self._segments:
            self.reset(instruction_segments)
        seg = self._segments[self._ptr] if self._ptr < len(self._segments) else ""
        # Mark current segment complete when key landmark words from the segment appear in obs
        seg_words = {w.lower() for w in seg.split() if len(w) > 3}
        obs_words = {w.lower() for w in obs_text.split()}
        overlap = seg_words & obs_words
        if len(overlap) >= max(1, len(seg_words) // 3):
            self._done[self._ptr] = True
            if self._ptr < len(self._segments) - 1:
                self._ptr += 1
        # Extract landmark: last noun phrase (heuristic: capitalised word or "the X")
        words = seg.split()
        for i, w in enumerate(reversed(words)):
            if w[0].isupper() or (w.lower() in ("kitchen", "door", "table", "chair",
                                                  "counter", "hallway", "stairs", "room")):
                self._landmark = w.rstrip(".,")
                break

    def current_segment(self) -> str:
        if self._ptr < len(self._segments):
            return self._segments[self._ptr]
        return ""

    def current_landmark(self) -> Optional[str]:
        return self._landmark

    def all_complete(self) -> bool:
        return all(self._done) if self._done else False

    def progress(self) -> tuple[int, int]:
        return self._ptr, len(self._segments)
