"""Body movement memory: odometry ring buffer with loop closure detection."""
from __future__ import annotations
import time
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np


class BodyMemory:
    RING_SIZE = 50

    def __init__(self):
        self._buf: deque = deque(maxlen=self.RING_SIZE)
        self._s2_call_pos: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._buf.clear()
        self._s2_call_pos = None

    def update(self, odom: Tuple[float, float, float]) -> None:
        """odom = (x, y, heading_rad)"""
        self._buf.append((float(odom[0]), float(odom[1]), float(odom[2]), time.time()))

    def mark_s2_call(self) -> None:
        if self._buf:
            self._s2_call_pos = np.array(self._buf[-1][:2])

    def displacement(self) -> float:
        """Distance traveled since last S2 call (or full buffer if no call marked)."""
        if not self._buf:
            return 0.0
        positions = np.array([s[:2] for s in self._buf])
        if self._s2_call_pos is not None:
            # Find the closest index to s2_call_pos and sum from there
            dists_from_ref = np.linalg.norm(positions - self._s2_call_pos, axis=1)
            ref_idx = int(np.argmin(dists_from_ref))
            segment = positions[ref_idx:]
        else:
            segment = positions
        if len(segment) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1)))

    def loop_detected(self, threshold: float = 0.5) -> bool:
        """True if current position is within threshold of any earlier position."""
        if len(self._buf) < 10:
            return False
        positions = np.array([s[:2] for s in self._buf])
        current = positions[-1]
        # Compare against all but last 5 steps (avoid detecting immediate vicinity)
        earlier = positions[:-5]
        if len(earlier) == 0:
            return False
        dists = np.linalg.norm(earlier - current, axis=1)
        return bool(dists.min() < threshold)

    def velocity_profile(self) -> Dict[str, float]:
        if len(self._buf) < 2:
            return {"mean": 0.0, "max": 0.0, "std": 0.0}
        positions = np.array([s[:2] for s in self._buf])
        times = np.array([s[3] for s in self._buf])
        dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        dts = np.diff(times)
        dts = np.where(dts < 1e-6, 1e-6, dts)
        speeds = dists / dts
        return {
            "mean": float(np.mean(speeds)),
            "max": float(np.max(speeds)),
            "std": float(np.std(speeds)),
        }

    def current_pose(self) -> Optional[np.ndarray]:
        if self._buf:
            return np.array(self._buf[-1][:3])
        return None

    def buffer_size(self) -> int:
        return len(self._buf)
