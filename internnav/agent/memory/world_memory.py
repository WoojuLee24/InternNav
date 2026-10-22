"""World memory: adaptive topological graph with CLIP loop closure. (DGNav / T²-Nav / OmniVLN)"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class _Node:
    node_id: int
    pose: np.ndarray          # [x, y, heading]
    emb: np.ndarray           # clip embedding (512-dim)
    timestamp: float
    room_type: str = "unknown"
    visit_count: int = 1


class WorldMemory:
    LOOP_CLOSURE_SIM = 0.85   # cosine sim threshold (DGNav)
    MERGE_DIST = 0.5          # m — merge nearby nodes in open areas

    def __init__(self, max_nodes: int = 200):
        self._nodes: Dict[int, _Node] = {}
        self._edges: List[Tuple[int, int]] = []
        self._next_id: int = 0
        self._last_id: Optional[int] = None
        self._max_nodes = max_nodes

    def reset(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._next_id = 0
        self._last_id = None

    def add_node(self, pose: np.ndarray, emb: np.ndarray,
                 room_type: str = "unknown") -> int:
        emb = np.asarray(emb, dtype=np.float32)
        pose = np.asarray(pose, dtype=np.float32)

        # Loop closure: merge if emb similarity > threshold
        if self._nodes:
            embs = np.stack([n.emb for n in self._nodes.values()])
            sims = embs @ emb / (np.linalg.norm(embs, axis=1) * np.linalg.norm(emb) + 1e-8)
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= self.LOOP_CLOSURE_SIM:
                matched_id = list(self._nodes.keys())[best_idx]
                node = self._nodes[matched_id]
                # Update pose as EMA and increment visit
                node.pose = 0.7 * node.pose + 0.3 * pose
                node.emb = node.emb + 0.1 * (emb - node.emb)
                node.emb /= np.linalg.norm(node.emb) + 1e-8
                node.visit_count += 1
                node.timestamp = time.time()
                if self._last_id is not None and matched_id != self._last_id:
                    self._edges.append((self._last_id, matched_id))
                self._last_id = matched_id
                return matched_id

        # Adaptive granularity: merge if spatially very close
        if self._nodes:
            poses = np.stack([n.pose[:2] for n in self._nodes.values()])
            dists = np.linalg.norm(poses - pose[:2], axis=1)
            if dists.min() < self.MERGE_DIST:
                close_id = list(self._nodes.keys())[int(np.argmin(dists))]
                node = self._nodes[close_id]
                node.emb = node.emb + 0.1 * (emb - node.emb)
                node.emb /= np.linalg.norm(node.emb) + 1e-8
                node.visit_count += 1
                if self._last_id is not None and close_id != self._last_id:
                    self._edges.append((self._last_id, close_id))
                self._last_id = close_id
                return close_id

        # Evict oldest node if at capacity
        if len(self._nodes) >= self._max_nodes:
            oldest = min(self._nodes, key=lambda k: self._nodes[k].timestamp)
            del self._nodes[oldest]

        nid = self._next_id
        self._next_id += 1
        self._nodes[nid] = _Node(
            node_id=nid, pose=pose.copy(), emb=emb.copy(),
            timestamp=time.time(), room_type=room_type)
        if self._last_id is not None:
            self._edges.append((self._last_id, nid))
        self._last_id = nid
        return nid

    def get_context(self, query_emb: np.ndarray, k: int = 5) -> List[_Node]:
        if not self._nodes:
            return []
        q = np.asarray(query_emb, dtype=np.float32)
        embs = np.stack([n.emb for n in self._nodes.values()])
        sims = embs @ q / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q) + 1e-8)
        top_k = int(min(k, len(self._nodes)))
        top_ids = np.argsort(sims)[-top_k:][::-1]
        keys = list(self._nodes.keys())
        return [self._nodes[keys[i]] for i in top_ids]

    def to_compact_str(self) -> str:
        """DSG-style compact context string, ≤200 tokens (OmniVLN insight)."""
        if not self._nodes:
            return "[empty world]"
        # Group by room_type
        rooms: Dict[str, List[str]] = {}
        for n in self._nodes.values():
            rt = n.room_type
            entry = f"({n.pose[0]:.1f},{n.pose[1]:.1f})"
            rooms.setdefault(rt, []).append(entry)
        parts = []
        for rt, locs in rooms.items():
            if len(locs) > 3:
                locs = locs[:3] + [f"+{len(locs)-3}more"]
            parts.append(f"{rt}:[{','.join(locs)}]")
        result = " ".join(parts)
        # Hard-truncate to ~200 tokens (≈800 chars)
        return result[:800]

    def node_count(self) -> int:
        return len(self._nodes)
