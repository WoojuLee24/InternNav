"""Integration test suites for the deployed modules.

Each suite module exposes  run(cfg, opts) -> list[TestResult]  and is also standalone
runnable against already-running modules:

    deploy/run_manager.sh test                    # manager: auto-start, run all, teardown
    convenv/bin/python -m deploy.tests.test_s1_module     # just this suite, services must be up
"""
import time
import traceback
from dataclasses import dataclass


@dataclass
class TestResult:
    name: str
    ok: bool
    ms: float
    detail: str = ""


def check(name, fn) -> TestResult:
    """Run one test callable; AssertionError -> FAIL with the reason, return value -> detail."""
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
        return TestResult(name, True, (time.perf_counter() - t0) * 1000.0, str(detail))
    except AssertionError as e:
        return TestResult(name, False, (time.perf_counter() - t0) * 1000.0, str(e))
    except Exception as e:
        traceback.print_exc()
        return TestResult(name, False, (time.perf_counter() - t0) * 1000.0, f"{type(e).__name__}: {e}")


def report(results, title=""):
    """Print a PASS/FAIL table; return True when everything passed."""
    if title:
        print(f"\n--- {title} ---")
    w = max((len(r.name) for r in results), default=10)
    for r in results:
        mark = "PASS" if r.ok else "FAIL"
        print(f"  [{mark}] {r.name:<{w}}  {r.ms:8.1f} ms  {r.detail}")
    n_ok = sum(r.ok for r in results)
    print(f"  {n_ok}/{len(results)} passed")
    return n_ok == len(results)


def load_test_frame(size):
    """The canonical test image (quantization_kit/dbg/test448.png), resized to `size`."""
    import numpy as np
    from PIL import Image

    from deploy.common.config import QK
    img = Image.open(QK / "dbg" / "test448.png").convert("RGB").resize((size, size))
    return np.asarray(img, dtype="uint8")
