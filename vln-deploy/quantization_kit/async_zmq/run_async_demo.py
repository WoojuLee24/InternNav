"""One-shot launcher: bring up the S2 service + S1 service as independent processes, wait
until both are loaded, then run the orchestrator in-process so its control-sequence timeline
prints straight to your terminal. Tears everything down (incl. the C++ runner) on exit.

  cd quantization_kit
  PYTHONPATH=/usr/lib/python3.12/dist-packages \
    convenv/bin/python -m quantization_kit.async_zmq.run_async_demo --seconds 8

Equivalent shortcut:  ./run.sh async
"""
import argparse
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

_QK = Path(__file__).resolve().parents[1]
_ROOT = _QK.parent
sys.path[:0] = [str(_ROOT), str(_QK)]

PY = str(_QK / f"convenv_{'aarch64' if platform.machine() == 'aarch64' else 'x86'}" / "bin" / "python")  # [fix] Thor/5090 자동 분기


def _spawn(name, module, extra, ready_token, ready_evt, extra_env=None):
    env = dict(os.environ)
    if platform.machine() == "aarch64":   # Jetson: system dist-packages holds native tensorrt/libnvinfer
        env["PYTHONPATH"] = "/usr/lib/python3.12/dist-packages" + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update(extra_env)
    p = subprocess.Popen(
        [PY, "-u", "-m", module, *extra],
        cwd=str(_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,   # own process group -> clean group kill (incl. runner child)
    )

    def _tee():
        for line in iter(p.stdout.readline, ""):
            sys.stdout.write(f"  [{name}] {line}")
            sys.stdout.flush()
            if ready_token in line:
                ready_evt.set()
        ready_evt.set()   # process exited -> unblock waiter

    threading.Thread(target=_tee, daemon=True).start()
    return p


def _setup_mps(pct):
    """If an SM cap is requested AND an MPS daemon is up, attach BOTH services to it
    (set CUDA_MPS_PIPE_DIRECTORY in this process -> inherited by both children, so S1 and S2
    are co-scheduled MPS clients) and return the extra env that caps *S2*'s SM share.
    Returns None (with a clear NOTE) when no daemon is found, so a default cap50 silently
    doing nothing never surprises you. Honours an explicit CUDA_MPS_PIPE_DIRECTORY override."""
    if not pct:
        return None
    pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    if os.path.exists(os.path.join(pipe, "control")):   # daemon's control socket
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = pipe     # both S1 + S2 become MPS clients
        print(f"[demo] MPS daemon found at {pipe} -> capping S2 to {pct}% of SMs "
              f"(higher % = faster S2 / slower S1; tune with --s2-mps-pct).", flush=True)
        return {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(pct)}
    print(f"[demo] NOTE: --s2-mps-pct {pct} requested but no MPS daemon at {pipe} (no 'control' pipe); "
          f"running WITHOUT MPS -> S1 sees the full S2 contention dip (~12.7 Hz vs ~18 Hz capped). "
          f"Start MPS as the same user, or pass --s2-mps-pct 0 to silence this.", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--instruction", default="Walk forward along the hallway and stop near the table.")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-trt-s1", action="store_true", help="run S1 in PyTorch fp32 (no TensorRT)")
    ap.add_argument("--faithful", action="store_true", help="honour S2's discrete/pixel choice (no force_latent)")
    ap.add_argument("--s1-steps", type=int, default=5, help="S1 diffusion steps (5 ~= 30 Hz)")
    ap.add_argument("--s2-period-s", type=float, default=0.0,
                    help="min wall-seconds between S2 fires. Default 0 = CONTINUOUS: S2 re-fires the moment "
                         "it finishes (freshest plans, ~0.7 Hz S2), but S1 is then ALWAYS in the contention "
                         "dip (~18 Hz flat under cap50, no 29 Hz windows). Set e.g. 3 to insert idle windows "
                         "where S1 runs full-speed (~21 Hz avg S1, but plans up to 3 s old).")
    ap.add_argument("--sys2-max-forward-step", type=int, default=1,
                    help="re-fire S2 after S1 has taken this many steps off the current plan. Default 1 = "
                         "S2 fires right after it completes (back-to-back, no idle gap). Raise it to let S1 "
                         "take more steps per S2 plan (a small fast-S1 window between fires).")
    ap.add_argument("--kv-reuse", action=argparse.BooleanOptionalAction, default=True,
                    help="S2 multi-turn KV reuse: append only the new frame (much faster S2)")
    ap.add_argument("--s2-q4", action=argparse.BooleanOptionalAction, default=True,
                    help="run S2 on the Q4_K_M-latent GGUF (DEFAULT: ~1.2x faster decode + shorter S1 "
                         "contention window, vln-ce-validated; latent cos ~0.96 vs f16). "
                         "Use --no-s2-q4 for the f16 reference model.")
    ap.add_argument("--s2-resize", type=int, default=336,
                    help="S2 input square resolution (112-aligned: 224/336/448). 448 = HF-parity reference, "
                         "336 = ~11%% faster (default). No rebuild needed.")
    ap.add_argument("--s2-mps-pct", type=int, default=80,
                    help="MPS SM cap for S2 (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE) -- the on-device S1<->S2 dial: "
                         "higher = faster S2 / slower S1. Measured (Q4, continuous): cap50 -> S1 18Hz, S2 0.62Hz | "
                         "cap70 -> 15, 0.92 | cap80 -> 13, 1.10 (DEFAULT: meets the S2>=1Hz target) | cap100 -> 7.5, 1.31. "
                         "Use 50 to favour S1, 100 to max S2. 0 = no MPS; cap30 crashes. Auto-attaches both "
                         "services to the daemon; prints a NOTE and runs un-capped if none is found.")
    ap.add_argument("--load-timeout", type=float, default=240.0)
    args = ap.parse_args()

    s2_ready = threading.Event()
    s1_ready = threading.Event()
    procs = []
    try:
        print("[demo] launching S2 service (Qwen2.5-VL GGUF + C++ ZMQ runner) ...", flush=True)
        s2_extra = (["--device", args.device] + (["--kv-reuse"] if args.kv_reuse else [])
                    + ["--s2-model", "q4" if args.s2_q4 else "f16"]
                    + ["--resize", str(args.s2_resize)])
        s2_env = _setup_mps(args.s2_mps_pct)
        procs.append(_spawn("s2", "quantization_kit.async_zmq.s2_service",
                            s2_extra, "S2_SERVICE READY", s2_ready, extra_env=s2_env))
        print("[demo] launching S1 service (lite diffusion + TensorRT) ...", flush=True)
        s1_extra = ["--device", args.device, "--s1-steps", str(args.s1_steps)] + (["--no-trt"] if args.no_trt_s1 else [])
        procs.append(_spawn("s1", "quantization_kit.async_zmq.s1_service", s1_extra,
                            "S1_SERVICE READY", s1_ready))

        t0 = time.time()
        while not (s2_ready.is_set() and s1_ready.is_set()):
            if time.time() - t0 > args.load_timeout:
                raise TimeoutError("services did not become ready in time")
            if any(p.poll() is not None for p in procs):
                raise RuntimeError("a service exited during startup (see logs above)")
            time.sleep(0.2)
        print("\n[demo] both services ready -> starting orchestrator\n", flush=True)

        from quantization_kit.async_zmq import orchestrator
        sys.argv = ["orchestrator", "--seconds", str(args.seconds), "--hz", str(args.hz),
                    "--instruction", args.instruction, "--host", "127.0.0.1",
                    "--s2-period-s", str(args.s2_period_s),
                    "--sys2-max-forward-step", str(args.sys2_max_forward_step)]
        if args.frames_dir:
            sys.argv += ["--frames-dir", args.frames_dir]
        if args.faithful:
            sys.argv += ["--faithful"]
        orchestrator.main()

    finally:
        print("\n[demo] shutting down services ...", flush=True)
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception: pass


if __name__ == "__main__":
    main()
