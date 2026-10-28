"""Wire-protocol suite — msgpack+numpy framing round-trips exactly (no services, no GPU)."""
import sys
from pathlib import Path

import numpy as np
import zmq

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deploy.common.interface import LATENT_DIM, LATENT_TOKENS, recv_msg, send_msg  # noqa: E402
from deploy.tests import check, report  # noqa: E402


def _pair(ctx):
    a = ctx.socket(zmq.PUSH)
    b = ctx.socket(zmq.PULL)
    a.setsockopt(zmq.LINGER, 0)
    b.setsockopt(zmq.LINGER, 0)
    a.bind("inproc://proto-test")
    b.connect("inproc://proto-test")
    return a, b


def _roundtrip(a, b, meta, arrays):
    send_msg(a, meta, arrays)
    m, r = recv_msg(b)
    assert m == meta, f"meta changed: {m} != {meta}"
    assert set(r) == set(arrays or {}), f"array names changed: {set(r)}"
    for k, v in (arrays or {}).items():
        assert r[k].dtype == v.dtype and r[k].shape == v.shape, f"{k}: {r[k].dtype}{r[k].shape}"
        assert np.array_equal(r[k], np.ascontiguousarray(v)), f"{k}: payload bytes changed"
    return m, r


def _camproj_roundtrip():
    """Ground->pixel->ground must be exact (repo camera conventions, see common/camproj.py)."""
    from deploy.common.camproj import CamModel, pixel_to_ground, project_ground
    cam = CamModel(size=336, hfov_deg=44.6, height_m=1.25, pitch_deg=0.0)
    uv, in_front, in_frame = project_ground(np.array([[5.0, 0.5]]), cam)   # 5 m ahead, 0.5 m left
    assert in_front[0] and in_frame[0], f"5m ground point should be visible, uv={uv[0]}"
    g = pixel_to_ground(uv[0][1], uv[0][0], cam)                           # (row=v, col=u)
    assert g and abs(g[0] - 5.0) < 1e-6 and abs(g[1] - 0.5) < 1e-6, f"round-trip drifted: {g}"
    assert pixel_to_ground(0, cam.size // 2, cam) is None, "above-horizon ray must miss the ground"
    return f"5.0m/0.5m -> uv={uv[0].round(1).tolist()} -> back exact; horizon guarded"


def run(cfg=None, opts=None):
    ctx = zmq.Context()          # private context: no interference with live sockets
    a, b = _pair(ctx)
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((1, LATENT_TOKENS, LATENT_DIM)).astype(np.float32)
    rgb = rng.integers(0, 256, (336, 336, 3), dtype=np.uint8)

    results = [
        check("proto/meta-only (reset)", lambda: _roundtrip(a, b, {"mode": "reset"}, {}) and ""),
        check("proto/s2-request", lambda: _roundtrip(
            a, b,
            {"seq": 7, "mode": "infer", "instruction": "부엌으로 가서 멈춰 (unicode ok)", "force_latent": True},
            {"rgb": rgb, "lookdown_rgb": rgb}) and "unicode + 2 arrays"),
        check("proto/s1-request", lambda: _roundtrip(
            a, b, {"seq": 3, "cache_key": None},
            {"latent": latent, "rgb_mem": rgb, "rgb_cur": rgb}) and f"latent {latent.shape}"),
        check("proto/s1-reply", lambda: _roundtrip(
            a, b, {"seq": 3, "actions": [1, 2, 3], "t_infer_ms": 33.1},
            {"waypoints": rng.standard_normal((9, 2)).astype(np.float32)}) and ""),
        check("proto/non-contiguous", lambda: _roundtrip(
            a, b, {"seq": 1}, {"latent": latent[0].T}) and "transposed view survives"),
        check("proto/camproj-roundtrip", _camproj_roundtrip),
    ]
    a.close()
    b.close()
    ctx.term()
    return results


def main():
    ok = report(run(), "protocol")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
