import importlib.util
import numpy as np
import pytest
import sys
import time
from pathlib import Path

# Load memory modules directly to avoid internnav.agent.__init__ pulling in
# quaternion (not installed in vlnav_env).
_MEM_DIR = Path(__file__).parents[2] / "internnav" / "agent" / "memory"

def _load(name):
    spec = importlib.util.spec_from_file_location(name, _MEM_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register so @dataclass can find __module__
    spec.loader.exec_module(mod)
    return mod

_task_mod = _load("task_memory")
_world_mod = _load("world_memory")
_body_mod = _load("body_memory")
TaskMemory = _task_mod.TaskMemory
WorldMemory = _world_mod.WorldMemory
BodyMemory = _body_mod.BodyMemory


# ── TaskMemory ─────────────────────────────────────────────────────────────

def test_task_memory_advance_segments():
    tm = TaskMemory()
    segs = ["walk to the kitchen", "pass through the doorway", "stop at the counter"]
    tm.reset(segs)
    assert tm.current_segment() == segs[0]
    assert not tm.all_complete()

    # Simulate obs matching segment 0
    tm.update("I see the kitchen ahead with the stove", segs)
    assert tm._ptr == 1
    assert not tm.all_complete()

    # Simulate obs matching segment 1
    tm.update("I am passing through the doorway now", segs)
    assert tm._ptr == 2
    assert not tm.all_complete()

    # Simulate obs matching segment 2
    tm.update("standing at the counter in the kitchen", segs)
    assert tm.all_complete()


def test_task_memory_progress():
    tm = TaskMemory()
    segs = ["go left", "enter room", "stop here"]
    tm.reset(segs)
    ptr, total = tm.progress()
    assert total == 3
    assert ptr == 0


# ── WorldMemory ────────────────────────────────────────────────────────────

def _rand_emb():
    e = np.random.randn(512).astype(np.float32)
    return e / np.linalg.norm(e)


def test_world_memory_add_nodes():
    wm = WorldMemory()
    for i in range(5):
        pose = np.array([float(i * 2), 0.0, 0.0])
        emb = _rand_emb()
        wm.add_node(pose, emb, room_type="hallway")
    assert wm.node_count() >= 1  # some may merge spatially but spread >0.5m each → 5


def test_world_memory_loop_closure():
    wm = WorldMemory()
    emb_a = _rand_emb()
    pose_a = np.array([0.0, 0.0, 0.0])
    nid_a = wm.add_node(pose_a, emb_a, room_type="kitchen")

    # Go somewhere else
    for i in range(3):
        wm.add_node(np.array([float(i + 5), 0.0, 0.0]), _rand_emb(), "hallway")

    # Return with nearly identical embedding → should merge back to node A
    nid_return = wm.add_node(np.array([0.1, 0.0, 0.0]), emb_a.copy(), "kitchen")
    assert nid_return == nid_a


def test_world_memory_get_context_returns_le_k():
    wm = WorldMemory()
    query = _rand_emb()
    for i in range(3):
        pose = np.array([float(i * 3), 0.0, 0.0])
        wm.add_node(pose, _rand_emb(), "room")
    results = wm.get_context(query, k=5)
    assert len(results) <= 5
    assert len(results) == min(5, wm.node_count())


def test_world_memory_compact_str_length():
    wm = WorldMemory()
    for i in range(10):
        wm.add_node(np.array([float(i * 2), 0.0, 0.0]), _rand_emb(), f"room_{i % 3}")
    s = wm.to_compact_str()
    assert len(s) <= 800
    assert len(s) > 0


# ── BodyMemory ─────────────────────────────────────────────────────────────

def test_body_memory_displacement():
    bm = BodyMemory()
    bm.mark_s2_call()
    # Move 1 m per step, 10 steps
    for i in range(10):
        bm.update((float(i), 0.0, 0.0))
    bm.mark_s2_call()
    for i in range(10, 20):
        bm.update((float(i), 0.0, 0.0))
    d = bm.displacement()
    # Should be ~10 m (from step 10 to 19, since mark_s2_call resets reference)
    assert 8.0 <= d <= 12.0


def test_body_memory_loop_detection():
    bm = BodyMemory()
    # Move away
    for i in range(15):
        bm.update((float(i), 0.0, 0.0))
    assert not bm.loop_detected(threshold=0.5)

    # Return to near start
    for _ in range(5):
        bm.update((0.1, 0.0, 0.0))
    assert bm.loop_detected(threshold=0.5)


def test_body_memory_velocity_profile():
    bm = BodyMemory()
    t0 = time.time()
    for i in range(10):
        bm._buf.append((float(i), 0.0, 0.0, t0 + i * 0.1))
    vp = bm.velocity_profile()
    assert "mean" in vp and "max" in vp and "std" in vp
    assert vp["mean"] > 0


def test_body_memory_buffer_size():
    bm = BodyMemory()
    for i in range(60):
        bm.update((float(i), 0.0, 0.0))
    assert bm.buffer_size() == 50  # ring buffer capped at 50
