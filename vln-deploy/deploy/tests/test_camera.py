"""Camera module suite — health, frame contract (shape/dtype/seq/age), and live rate."""
import sys
import time
from pathlib import Path

import numpy as np
import zmq

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deploy.common.config import DeployConfig  # noqa: E402
from deploy.common.interface import HealthClient, make_socket, recv_msg  # noqa: E402
from deploy.tests import check, report  # noqa: E402


def run(cfg=None, opts=None):
    cfg = cfg or DeployConfig.from_env()
    ctx = zmq.Context.instance()
    sub = make_socket(ctx, zmq.SUB, cfg.ports["cam"], bind=False, host=cfg.cam_host)
    poll = zmq.Poller()
    poll.register(sub, zmq.POLLIN)
    time.sleep(0.3)                                     # PUB/SUB slow-joiner settle
    results = []

    def _grab(timeout_s=3.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if poll.poll(50):
                return recv_msg(sub)
        raise AssertionError(f"no camera frame within {timeout_s}s on cam:{cfg.ports['cam']}")

    def t_health():
        h = HealthClient(cfg.cam_host, cfg.ports["cam_health"]).ping()
        assert h and h.get("ok"), "no health reply on cam_health"
        assert h.get("ready"), "camera not ready (device open failed?)"
        return f"pid={h['pid']} uptime={h['uptime_s']}s rtt={h['rtt_ms']:.1f}ms"

    def t_frame():
        m, a = _grab()
        rgb = a["rgb"]
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, f"rgb {rgb.dtype}{rgb.shape}"
        assert rgb.shape[0] == rgb.shape[1], f"frame not square: {rgb.shape}"
        assert rgb.shape[0] % 112 == 0, f"size {rgb.shape[0]} not 112-aligned (pixel-goal parity!)"
        assert float(rgb.std()) > 1.0, "frame is flat (lens covered / wrong device?)"
        age_ms = (time.time() - m["t_cap"]) * 1000.0
        assert age_ms < 1000.0, f"stale frame: {age_ms:.0f}ms old"
        return f"{rgb.shape} u8 seq={m['seq']} age={age_ms:.0f}ms std={rgb.std():.1f}"

    def t_stream():
        n, t0 = 0, time.time()
        first = last = None
        while time.time() - t0 < 2.0:
            if poll.poll(50):
                m, _ = recv_msg(sub)
                first = m["seq"] if first is None else first
                last = m["seq"]
                n += 1
        assert n >= 10, f"only {n} frames in 2s"
        assert last > first, f"seq not increasing ({first}->{last})"
        return f"{n / (time.time() - t0):.1f} fps observed (seq {first}->{last}, no reordering)"

    results.append(check("cam/health", t_health))
    results.append(check("cam/frame-contract", t_frame))
    results.append(check("cam/stream-rate", t_stream))
    sub.close()
    return results


def main():
    ok = report(run(), "camera")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
