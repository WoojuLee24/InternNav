"""End-to-end suite — the dual-system handshake ACROSS both deployed modules.

  chain:    rgb --s2_in--> S2 --s2_out--> latent --s1_in--> S1 --s1_out--> actions
  overlap:  fire S2 asynchronously and PROVE S1 keeps producing actions while S2 thinks
            (the whole point of the split: the control path never blocks on S2).
"""
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deploy.common.config import DeployConfig  # noqa: E402
from deploy.common.interface import (  # noqa: E402
    S1Client,
    S2Client,
    validate_s1_result,
    validate_s2_result,
)
from deploy.tests import check, load_test_frame, report  # noqa: E402


def run(cfg=None, opts=None):
    cfg = cfg or DeployConfig.from_env()
    rgb = load_test_frame(cfg.s2_resize)
    s2, s1 = S2Client(cfg), S1Client(cfg)
    results = []
    state = {}

    def t_chain():
        s2.reset()
        r2 = s2.infer(rgb, cfg.instruction, force_latent=True, timeout_s=180.0)
        validate_s2_result(r2, resize=cfg.s2_resize)
        assert r2.kind == "latent", f"expected a latent plan for the chain, got {r2.kind}"
        r1 = s1.act(r2.latent, r2.goal_rgb, rgb)
        validate_s1_result(r1)
        state["plan"] = r2
        return (f"S2 {r2.t_infer_ms:.0f}ms -> latent{tuple(r2.latent.shape)} -> "
                f"S1 {r1.t_infer_ms:.1f}ms -> actions={r1.actions}")

    def t_overlap():
        plan = state.get("plan")
        assert plan is not None, "needs the chain test above"
        seq = s2.fire(rgb, cfg.instruction, force_latent=True)   # fire-and-FORGET
        ticks, t0 = 0, time.perf_counter()
        r2 = None
        while time.perf_counter() - t0 < 120.0:
            r2 = s2.try_recv(seq)                                # non-blocking drain
            if r2 is not None:
                break
            r1 = s1.act(plan.latent, plan.goal_rgb, rgb)         # S1 keeps steering off the last plan
            validate_s1_result(r1)
            ticks += 1
        validate_s2_result(r2, resize=cfg.s2_resize)
        hz = ticks / max(time.perf_counter() - t0, 1e-9)
        assert ticks >= 1, "S1 never got a tick in while S2 was inferring"
        state["plan"] = r2 if r2.kind == "latent" else plan
        return (f"S2 busy {r2.t_infer_ms:.0f}ms; S1 overlapped {ticks} ticks @ {hz:.1f} Hz "
                f"-> control loop never blocked")

    def t_replan():
        plan = state.get("plan")
        assert plan is not None and plan.kind == "latent", "needs a latent plan"
        r1 = s1.act(plan.latent, plan.goal_rgb, rgb)
        validate_s1_result(r1)
        return f"S1 follows the refreshed plan: actions={r1.actions} ({r1.t_infer_ms:.1f}ms)"

    results.append(check("e2e/chain S2->S1", t_chain))
    results.append(check("e2e/overlap (non-blocking)", t_overlap))
    results.append(check("e2e/replan-switch", t_replan))
    s2.reset()
    s2.close()
    s1.close()
    return results


def main():
    ok = report(run(), "e2e")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
