"""S1 module suite — health, interface validity, and throughput of the deployed S1 service."""
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deploy.common.config import DeployConfig  # noqa: E402
from deploy.common.interface import HealthClient, S1Client, validate_s1_result  # noqa: E402
from deploy.tests import check, load_test_frame, report  # noqa: E402

N_THROUGHPUT = 15


def run(cfg=None, opts=None):
    cfg = cfg or DeployConfig.from_env()
    rng = np.random.default_rng(42)
    rgb = load_test_frame(cfg.s2_resize)
    zeros = np.zeros((1, 4, 3584), np.float32)             # same probe the service warms up with
    randl = (rng.standard_normal((1, 4, 3584)) * 0.5).astype(np.float32)
    c = S1Client(cfg)
    results = []

    def t_health():
        h = HealthClient(cfg.s1_host, cfg.ports["s1_health"]).ping()
        assert h and h.get("ok"), "no health reply on s1_health"
        assert h.get("ready"), "S1 not ready (still loading?)"
        return f"pid={h['pid']} uptime={h['uptime_s']}s rtt={h['rtt_ms']:.1f}ms"

    def t_zeros():
        r = c.act(zeros, rgb, rgb)
        validate_s1_result(r)
        reach = float(np.linalg.norm(r.waypoints[-1]))
        return (f"actions={r.actions} denoised waypoints={r.waypoints.shape} "
                f"reach={reach:.2f}m {r.t_infer_ms:.1f}ms")

    def t_random():
        r = c.act(randl, rgb, rgb)
        validate_s1_result(r)
        return f"actions={r.actions} {r.t_infer_ms:.1f}ms"

    def t_throughput():
        t0 = time.perf_counter()
        ms = []
        for _ in range(N_THROUGHPUT):
            r = c.act(randl, rgb, rgb)
            validate_s1_result(r)
            ms.append(r.t_infer_ms)
        hz = N_THROUGHPUT / (time.perf_counter() - t0)
        assert hz >= 1.0, f"S1 too slow: {hz:.1f} Hz"
        return f"{hz:.1f} Hz e2e over {N_THROUGHPUT} calls (infer {np.mean(ms):.1f}±{np.std(ms):.1f}ms)"

    results.append(check("s1/health", t_health))
    results.append(check("s1/act-zeros-latent", t_zeros))
    results.append(check("s1/act-random-latent", t_random))
    results.append(check("s1/throughput", t_throughput))
    c.close()
    return results


def main():
    ok = report(run(), "s1_module")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
