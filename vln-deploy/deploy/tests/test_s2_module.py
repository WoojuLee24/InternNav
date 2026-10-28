"""S2 module suite — health, reset/accumulate/infer, and latent contract of the deployed S2."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deploy.common.config import DeployConfig  # noqa: E402
from deploy.common.interface import HealthClient, S2Client, validate_s2_result  # noqa: E402
from deploy.tests import check, load_test_frame, report  # noqa: E402


def run(cfg=None, opts=None):
    cfg = cfg or DeployConfig.from_env()
    rgb = load_test_frame(cfg.s2_resize)
    c = S2Client(cfg)
    results = []
    state = {}

    def t_health():
        h = HealthClient(cfg.s2_host, cfg.ports["s2_health"]).ping()
        assert h and h.get("ok"), "no health reply on s2_health"
        assert h.get("ready"), "S2 not ready (still loading?)"
        return f"pid={h['pid']} uptime={h['uptime_s']}s rtt={h['rtt_ms']:.1f}ms"

    def t_infer_cold():
        c.reset()
        r = c.infer(rgb, cfg.instruction, force_latent=True, timeout_s=180.0)   # first: CUDA warm
        validate_s2_result(r, resize=cfg.s2_resize)
        state["r0"] = r
        return f"kind={r.kind} pixel={r.pixel} {r.t_infer_ms:.0f}ms  text={r.text[:48]!r}"

    def t_accumulate_then_infer():
        c.accumulate(rgb)                                   # history grows without inferring
        r = c.infer(rgb, cfg.instruction, force_latent=True, timeout_s=120.0)
        validate_s2_result(r, resize=cfg.s2_resize)
        state["r1"] = r
        return f"kind={r.kind} pixel={r.pixel} {r.t_infer_ms:.0f}ms (warm, history+1)"

    def t_latent_varies():
        import numpy as np
        r0, r1 = state.get("r0"), state.get("r1")
        assert r0 is not None and r1 is not None, "needs the two infer tests above"
        assert r0.kind == "latent" and r1.kind == "latent", "need latent replies to compare"
        a, b = r0.latent.ravel(), r1.latent.ravel()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        assert not np.array_equal(r0.latent, r1.latent), "two steps returned bit-identical latents"
        return f"step-to-step latent cos={cos:.3f} (state evolves, not frozen)"

    def t_reset():
        c.reset()                                            # must not wedge the service
        r = c.infer(rgb, cfg.instruction, force_latent=True, timeout_s=120.0)
        validate_s2_result(r, resize=cfg.s2_resize)
        return f"post-reset infer ok ({r.t_infer_ms:.0f}ms)"

    results.append(check("s2/health", t_health))
    results.append(check("s2/reset+infer(cold)", t_infer_cold))
    results.append(check("s2/accumulate+infer", t_accumulate_then_infer))
    results.append(check("s2/latent-evolves", t_latent_varies))
    results.append(check("s2/reset-recovers", t_reset))
    c.reset()
    c.close()
    return results


def main():
    ok = report(run(), "s2_module")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
