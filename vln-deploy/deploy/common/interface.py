"""THE wire contract between the deployed modules — schemas, clients, health.

Every inter-module byte crosses ZMQ using the framing of
`quantization_kit.async_zmq.zmq_proto` (msgpack header + raw numpy frames), re-exported
here so deploy code has exactly one import point.

================================  MESSAGE SCHEMAS  ================================

S2 request   (PUSH -> s2_in :5601)
    meta   {"seq": int, "mode": "infer",              # "accumulate" (history only) | "reset"
            "instruction": str, "force_latent": bool}
    arrays {"rgb": u8 [H,W,3], "lookdown_rgb": u8 [H,W,3]}      # lookdown used on action 5

S2 reply     (PUB <- s2_out :5602)  — one per "infer" request, matched by seq
    kind=latent: meta   {"seq", "kind": "latent", "pixel": [x,y]|None, "action": None,
                         "text": str, "t_infer_ms": float}
                 arrays {"latent": f32 [1,4,3584], "goal_rgb": u8 [H,W,3]}
    kind=action: meta   {"seq", "kind": "action", "action": [int..], "pixel": None,
                         "text": str, "t_infer_ms": float}      # discrete; S1 not needed

S1 request   (PUSH -> s1_in :5603)
    meta   {"seq": int, "cache_key": int|None}
    arrays {"latent": f32 [1,4,3584], "rgb_mem": u8 [H,W,3], "rgb_cur": u8 [H,W,3]}

S1 reply     (PUSH <- s1_out :5604)  — one per request, matched by seq
    meta   {"seq": int, "actions": [int..] (<=4, STOP filtered), "t_infer_ms": float}
    arrays {"waypoints": f32 [T+1,2]}   # the DENOISED robot-facing path, metric robot
                                        # frame (x fwd, y left) — verbatim output of
                                        # vln_utils.traj_to_actions(use_discrate_action=False)
                                        # (the same trajectory the discrete actions come from)

Viz overlay  (PUB <- viz :5607, orchestrator -> viewer)
    meta   {"action", "src", "instr", "pixel": [row,col]|None, "inflight", "latent_age",
            "s1_hz", "s2_hz", "s2_text"}      # pixel is POLICY-NATIVE [row, col]
    arrays {"waypoints": f32 [T+1,2], "s1_frame": u8 [H,W,3]}   # the path PAIRED with the
                                                                # exact frame S1 computed it on

Health       (REQ <-> s1_health :5614 / s2_health :5615, JSON)   — deploy-only, out-of-band:
    {"cmd": "ping"}  ->  {"ok": true, "module": "s1"|"s2", "pid": int,
                          "ready": bool, "uptime_s": float}
    `ready` flips when the service prints its READY line (model loaded, sockets bound).
    Health NEVER touches the data path, so it is safe while an orchestrator is live.
    (Do NOT attach a second consumer to s1_out during live operation: PUSH round-robins.)

Action space: 0=STOP  1=forward  2=left  3=right  5=look_down.
===================================================================================
"""
import io
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import zmq

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantization_kit.async_zmq.zmq_proto import make_socket, recv_msg, send_msg  # noqa: E402,F401

from deploy.common.config import DeployConfig  # noqa: E402

ACTIONS = {0: "STOP", 1: "forward", 2: "left", 3: "right", 5: "look_down"}
LATENT_TOKENS, LATENT_DIM = 4, 3584
MAX_S1_ACTIONS = 4


# ======================================================================================
# results + validators (validators raise AssertionError with a precise reason)
# ======================================================================================
@dataclass
class S2Result:
    seq: int
    kind: str                      # "latent" | "action"
    latent: Optional[np.ndarray]   # f32 [1,4,3584] when kind=latent
    goal_rgb: Optional[np.ndarray]
    pixel: Optional[list]
    action: Optional[list]
    text: str
    t_infer_ms: float


@dataclass
class S1Result:
    seq: int
    actions: list
    waypoints: Optional[np.ndarray]   # DENOISED robot-facing path [T+1, 2] (metric robot frame)
    t_infer_ms: float


def validate_s2_result(r: S2Result, resize: Optional[int] = None):
    assert r is not None, "no S2 reply (timeout)"
    assert r.kind in ("latent", "action"), f"unknown kind {r.kind!r}"
    assert not r.text.startswith("ERR:"), f"S2 infer error: {r.text}"
    if r.kind == "latent":
        v = r.latent
        assert v is not None, "kind=latent but no latent array"
        assert tuple(v.shape[-2:]) == (LATENT_TOKENS, LATENT_DIM), f"latent shape {v.shape}"
        assert v.dtype == np.float32, f"latent dtype {v.dtype}"
        assert np.isfinite(v).all(), "latent has NaN/Inf"
        assert float(np.abs(v).max()) > 0, "latent is all-zero"
        g = r.goal_rgb
        assert g is not None and g.dtype == np.uint8 and g.ndim == 3 and g.shape[2] == 3, \
            f"goal_rgb bad: {None if g is None else (g.dtype, g.shape)}"
        if r.pixel is not None:
            assert len(r.pixel) == 2, f"pixel {r.pixel}"
            if resize:
                assert 0 <= r.pixel[0] <= resize and 0 <= r.pixel[1] <= resize, \
                    f"pixel {r.pixel} outside {resize}px frame"
    else:
        assert isinstance(r.action, list) and len(r.action) >= 1, f"action {r.action}"
        assert all(a in ACTIONS for a in r.action), f"action ids {r.action} outside {sorted(ACTIONS)}"
    assert r.t_infer_ms > 0, "t_infer_ms missing"


def validate_s1_result(r: S1Result):
    assert r is not None, "no S1 reply (timeout)"
    assert isinstance(r.actions, list) and len(r.actions) <= MAX_S1_ACTIONS, f"actions {r.actions}"
    assert all(a in ACTIONS for a in r.actions), f"action ids {r.actions} outside {sorted(ACTIONS)}"
    w = r.waypoints
    assert w is not None and w.ndim == 2 and w.shape[1] == 2 and w.shape[0] >= 2, \
        f"waypoints bad: {None if w is None else w.shape}"
    assert np.isfinite(w).all(), "waypoints have NaN/Inf"
    assert np.abs(w[0]).max() == 0.0, "waypoints must start at the robot origin"
    assert float(np.abs(w).max()) < 20.0, \
        f"waypoints not metric (max |xy| {np.abs(w).max():.1f} m — missing the /4 unnormalize?)"
    assert r.t_infer_ms > 0, "t_infer_ms missing"


# ======================================================================================
# module clients — how ANY consumer (manager, tests, robot code) talks to a module
# ======================================================================================
class S2Client:
    """Talks to the S2 module: reset / accumulate / infer (sync) or fire + try_recv (async)."""

    def __init__(self, cfg: DeployConfig = None, ctx=None, settle_s=0.4):
        self.cfg = cfg or DeployConfig.from_env()
        self.ctx = ctx or zmq.Context.instance()
        p, h = self.cfg.ports, self.cfg.s2_host
        self.push = make_socket(self.ctx, zmq.PUSH, p["s2_in"], bind=False, host=h)
        self.sub = make_socket(self.ctx, zmq.SUB, p["s2_out"], bind=False, host=h)
        self.poll = zmq.Poller()
        self.poll.register(self.sub, zmq.POLLIN)
        self._seq = 0
        time.sleep(settle_s)   # PUB/SUB slow-joiner: let the subscription reach the service

    def reset(self):
        send_msg(self.push, {"mode": "reset"})

    def accumulate(self, rgb):
        send_msg(self.push, {"mode": "accumulate"}, {"rgb": rgb.astype(np.uint8)})

    def fire(self, rgb, instruction, lookdown_rgb=None, force_latent=False, seq=None) -> int:
        """Fire-and-forget infer request; pair with try_recv(). Returns the seq to match."""
        seq = self._seq if seq is None else seq
        self._seq = seq + 1
        send_msg(self.push,
                 {"seq": seq, "mode": "infer", "instruction": instruction,
                  "force_latent": bool(force_latent)},
                 {"rgb": rgb.astype(np.uint8),
                  "lookdown_rgb": (lookdown_rgb if lookdown_rgb is not None else rgb).astype(np.uint8)})
        return seq

    def try_recv(self, seq=None) -> Optional[S2Result]:
        """Non-blocking: drain one S2 reply if present (optionally only accept `seq`)."""
        if not self.poll.poll(0):
            return None
        meta, arr = recv_msg(self.sub)
        if seq is not None and meta.get("seq") != seq:
            return None                                   # stale reply from an earlier request
        return S2Result(seq=meta.get("seq", -1), kind=meta.get("kind", "?"),
                        latent=arr.get("latent"), goal_rgb=arr.get("goal_rgb"),
                        pixel=meta.get("pixel"), action=meta.get("action"),
                        text=meta.get("text") or "", t_infer_ms=meta.get("t_infer_ms", 0.0))

    def infer(self, rgb, instruction, lookdown_rgb=None, force_latent=False,
              timeout_s=90.0) -> Optional[S2Result]:
        seq = self.fire(rgb, instruction, lookdown_rgb, force_latent)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            r = self.try_recv(seq)
            if r is not None:
                return r
            time.sleep(0.02)
        return None

    def close(self):
        self.push.close()
        self.sub.close()


class S1Client:
    """Talks to the S1 module: one latent + goal/current RGB -> discrete actions + waypoints."""

    def __init__(self, cfg: DeployConfig = None, ctx=None):
        self.cfg = cfg or DeployConfig.from_env()
        self.ctx = ctx or zmq.Context.instance()
        p, h = self.cfg.ports, self.cfg.s1_host
        self.push = make_socket(self.ctx, zmq.PUSH, p["s1_in"], bind=False, host=h)
        self.pull = make_socket(self.ctx, zmq.PULL, p["s1_out"], bind=False, host=h)
        self.poll = zmq.Poller()
        self.poll.register(self.pull, zmq.POLLIN)
        self._seq = 0

    def act(self, latent, rgb_mem, rgb_cur, cache_key=None, timeout_s=15.0,
            depth_mem=None, depth_cur=None, seed=None) -> Optional[S1Result]:
        """depth_mem/depth_cur (f32 metres) and seed are optional; only the HF S1 backend reads them."""
        seq = self._seq
        self._seq += 1
        arrays = {"latent": latent.astype(np.float32),
                  "rgb_mem": rgb_mem.astype(np.uint8), "rgb_cur": rgb_cur.astype(np.uint8)}
        if depth_mem is not None:
            arrays["depth_mem"] = depth_mem.astype(np.float32)
        if depth_cur is not None:
            arrays["depth_cur"] = depth_cur.astype(np.float32)
        meta = {"seq": seq, "cache_key": cache_key}
        if seed is not None:
            meta["seed"] = int(seed)
        send_msg(self.push, meta, arrays)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.poll.poll(20):
                meta, arr = recv_msg(self.pull)
                if meta.get("seq") != seq:
                    continue                              # stale reply from an earlier request
                wp = arr.get("waypoints")
                if wp is None and "trajs" in arr:         # legacy raw-fan reply -> same mean
                    wp = arr["trajs"].mean(axis=0)
                return S1Result(seq=seq, actions=meta.get("actions") or [],
                                waypoints=wp,
                                t_infer_ms=meta.get("t_infer_ms", 0.0))
        return None

    def close(self):
        self.push.close()
        self.pull.close()


class HealthClient:
    """Out-of-band liveness/readiness probe (safe during live operation)."""

    def __init__(self, host, port, ctx=None):
        self.host, self.port = host, port
        self.ctx = ctx or zmq.Context.instance()

    def ping(self, timeout_s=1.0) -> Optional[dict]:
        req = self.ctx.socket(zmq.REQ)                    # fresh REQ per ping: no stuck state machine
        req.setsockopt(zmq.LINGER, 0)
        req.connect(f"tcp://{self.host}:{self.port}")
        try:
            t0 = time.perf_counter()
            req.send_json({"cmd": "ping"})
            if not req.poll(int(timeout_s * 1000)):
                return None
            rep = req.recv_json()
            rep["rtt_ms"] = (time.perf_counter() - t0) * 1000.0
            return rep
        except zmq.ZMQError:
            return None
        finally:
            req.close()


# ======================================================================================
# health beacon — runs INSIDE each module (started before the heavy imports)
# ======================================================================================
class HealthBeacon(threading.Thread):
    """REP socket answering pings with {module, pid, ready, uptime_s}. Daemon thread."""

    def __init__(self, module: str, port: int):
        super().__init__(daemon=True, name=f"health-{module}")
        self.state = {"module": module, "pid": os.getpid(), "ready": False}
        self.port = port
        self._t0 = time.time()

    def mark_ready(self):
        self.state["ready"] = True

    def run(self):
        ctx = zmq.Context.instance()
        rep = ctx.socket(zmq.REP)
        rep.setsockopt(zmq.LINGER, 0)
        rep.bind(f"tcp://*:{self.port}")
        while True:
            try:
                rep.recv_json()
                rep.send_json({"ok": True, **self.state,
                               "uptime_s": round(time.time() - self._t0, 1)})
            except zmq.ZMQError:
                break


class _ReadyWatch(io.TextIOBase):
    """stdout tee that fires a callback when the service's READY line passes through,
    so readiness is observable on the health port without touching the service code."""

    def __init__(self, inner, token, on_ready):
        self.inner, self.token, self.on_ready = inner, token, on_ready
        self.fired = False

    def write(self, s):
        if not self.fired and self.token in s:
            self.fired = True
            self.on_ready()
        return self.inner.write(s)

    def flush(self):
        self.inner.flush()


def install_ready_watch(token: str, on_ready):
    sys.stdout = _ReadyWatch(sys.stdout, token, on_ready)
