"""Deployment manager (master) — lifecycle + integration testing for the S1/S2 modules.

    deploy/run_manager.sh start [s1|s2|cam|viz|all] [--wait]   spawn module daemons (pidfiles run/, logs logs/)
    deploy/run_manager.sh status                       health table (out-of-band; safe while live)
    deploy/run_manager.sh test [--suite ...]           PASS/FAIL integration suites (auto-starts what it needs)
    deploy/run_manager.sh demo [--seconds 8]           the async control loop on static frames
    deploy/run_manager.sh camera [--view mjpeg]        LIVE: camera -> S2/S1 -> actions, with the 3-panel viewer
    deploy/run_manager.sh prompt "<instruction>"       change the instruction of a live camera/demo session
    deploy/run_manager.sh logs s2 [-n 40]              tail a module log
    deploy/run_manager.sh stop [s1|s2|cam|viz|all]     graceful shutdown (SIGTERM group -> SIGKILL)
    deploy/run_manager.sh clean                        force-free every deploy port + kill leftovers

The manager owns NO model state: it only speaks the deploy interface (ZMQ data ports +
out-of-band health ports) and POSIX process control. Modules on other hosts (VLN_S2_HOST=...)
are probed/tested but never spawned/stopped locally.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deploy.common.config import CONVENV_PY, DEPLOY_DIR, LLAMA_BUILD, QK, ROOT, DeployConfig  # noqa: E402
from deploy.common.interface import HealthClient  # noqa: E402

MODULES = {
    "s2": {"run_sh": DEPLOY_DIR / "s2_module" / "run.sh", "health": "s2_health",
           "host_attr": "s2_host", "data_ports": ["s2_in", "s2_out", "s2_runner"]},
    "s1": {"run_sh": DEPLOY_DIR / "s1_module" / "run.sh", "health": "s1_health",
           "host_attr": "s1_host", "data_ports": ["s1_in", "s1_out"]},
    "cam": {"run_sh": DEPLOY_DIR / "camera_module" / "run.sh", "health": "cam_health",
            "host_attr": "cam_host", "data_ports": ["cam"]},
    "viz": {"run_sh": DEPLOY_DIR / "viewer_module" / "run.sh", "health": "viz_health",
            "host_attr": None, "data_ports": []},           # pure observer: SUBs only, always local
}
ALL_MODULES = ["s2", "s1", "cam", "viz"]   # spawn/display order (S2 loads longest)
CORE = ["s2", "s1"]                        # what `start all` / `demo` bring up


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Manager:
    def __init__(self, cfg: DeployConfig):
        self.cfg = cfg
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        cfg.run_dir.mkdir(parents=True, exist_ok=True)

    # ---------- bookkeeping ----------
    def _pidfile(self, mod):
        return self.cfg.run_dir / f"{mod}.json"

    def _logfile(self, mod):
        return self.cfg.log_dir / f"{mod}.log"

    def _host(self, mod):
        attr = MODULES[mod]["host_attr"]
        return getattr(self.cfg, attr) if attr else "127.0.0.1"

    def _is_remote(self, mod):
        return self._host(mod) not in ("127.0.0.1", "localhost")

    def pid_of(self, mod):
        """Managed local pid, or None. Cleans up stale pidfiles."""
        f = self._pidfile(mod)
        if not f.exists():
            return None
        try:
            pid = json.loads(f.read_text())["pid"]
        except Exception:
            f.unlink(missing_ok=True)
            return None
        if not _alive(pid):
            f.unlink(missing_ok=True)
            return None
        return pid

    def health(self, mod, timeout_s=0.8):
        return HealthClient(self._host(mod), self.cfg.ports[MODULES[mod]["health"]]).ping(timeout_s)

    # ---------- preflight ----------
    def preflight(self, mod):
        missing = []
        if not CONVENV_PY.exists() and not os.environ.get("VLN_PY"):
            missing.append(f"python venv: {CONVENV_PY}")
        if mod in ("s1", "s2") and not (ROOT / "checkpoints" / "InternVLA-N1-DualVLN").is_dir():
            missing.append(f"checkpoint: {ROOT / 'checkpoints' / 'InternVLA-N1-DualVLN'}")
        if mod == "s2" and self.cfg.s2_backend == "gguf":
            runner = LLAMA_BUILD / "bin" / "llama-s2-runner"
            if not runner.exists():
                missing.append(f"runner: {runner}  (cmake --build {LLAMA_BUILD} --target llama-s2-runner -j)")
            text = {"q4": "InternVLA-S2-text-Q4_K_M-latent.gguf",
                    "f16": "InternVLA-S2-text-f16-latent.gguf"}[self.cfg.s2_model]
            for g in (text, "InternVLA-S2-mmproj-f16.gguf"):
                if not (QK / "gguf" / g).exists():
                    missing.append(f"gguf: {QK / 'gguf' / g}  (quantization_kit/run.sh setup)")
        if mod == "cam" and not Path(f"/dev/video{self.cfg.cam_device}").exists():
            missing.append(f"camera device: /dev/video{self.cfg.cam_device}  (--cam-device N / VLN_CAM_DEVICE)")
        for m in missing:
            print(f"  [preflight:{mod}] MISSING {m}")
        return not missing

    # ---------- service argv ----------
    def _service_args(self, mod):
        c = self.cfg
        if mod == "s2":
            args = ["--device", c.device, "--resize", str(c.s2_resize),
                    "--s2-model", c.s2_model, "--num-history", str(c.num_history)]
            if c.kv_reuse:
                args.append("--kv-reuse")
            if c.s2_backend == "hf":
                args += ["--s2-backend", "hf", "--hf-quant", c.hf_quant]
                if c.hf_model_path:
                    args += ["--hf-model-path", c.hf_model_path]
        elif mod == "s1":
            args = ["--device", c.device, "--s1-steps", str(c.s1_steps)]
            if not c.trt_s1:
                args.append("--no-trt")
            if c.s1_backend == "hf":
                args += ["--s1-backend", "hf"]
                if c.hf_model_path:
                    args += ["--hf-model-path", c.hf_model_path]
        elif mod == "cam":
            # camera size == S2 resize: keeps S2 pixel goals 1:1 on the published frame
            args = ["--device", str(c.cam_device), "--size", str(c.s2_resize)]
        else:   # viz — pure SUB observer; point it at wherever the publishers live
            args = ["--view", c.viz_view, "--mjpeg-port", str(c.mjpeg_port),
                    "--cam-host", c.cam_host, "--s2-host", c.s2_host, "--viz-host", "127.0.0.1",
                    "--cam-hfov", str(c.cam_hfov_deg), "--cam-height", str(c.cam_height_m),
                    "--cam-pitch", str(c.cam_pitch_deg)]
        return args

    def _mps_env(self, mod):
        """Mirror run_async_demo._setup_mps: cap S2's SM share iff an MPS daemon is up."""
        if mod == "s1" and self.cfg.s1_mps_pct:
            pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
            if os.path.exists(os.path.join(pipe, "control")):
                print(f"[manager] MPS daemon at {pipe} -> capping S1 to {self.cfg.s1_mps_pct}% of SMs")
                return {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(self.cfg.s1_mps_pct)}
            return None
        if mod != "s2" or not self.cfg.s2_mps_pct:
            return None
        pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
        if os.path.exists(os.path.join(pipe, "control")):
            os.environ["CUDA_MPS_PIPE_DIRECTORY"] = pipe          # children (S1 too) become MPS clients
            print(f"[manager] MPS daemon at {pipe} -> capping S2 to {self.cfg.s2_mps_pct}% of SMs")
            return {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(self.cfg.s2_mps_pct)}
        print(f"[manager] NOTE: no MPS daemon at {pipe} -> S2 runs UNCAPPED "
              f"(S1 will dip harder while S2 infers; start mps.sh or set VLN_S2_MPS_PCT=0 to silence)")
        return None

    # ---------- lifecycle ----------
    def start(self, mods, wait=True, timeout_s=300.0):
        mods = [m for m in ALL_MODULES if m in mods]
        started = []
        for mod in mods:
            if self._is_remote(mod):
                print(f"[manager] {mod}: remote @ {getattr(self.cfg, MODULES[mod]['host_attr'])} — not spawning")
                continue
            pid = self.pid_of(mod)
            if pid:
                # cam/viz are cheap: respawn them when the desired args changed (a stale
                # viewer silently keeps its OLD calibration/backend otherwise)
                stored = {}
                try:
                    stored = json.loads(self._pidfile(mod).read_text())
                except Exception:
                    pass
                if mod in ("cam", "viz") and stored.get("argv") != self._service_args(mod):
                    print(f"[manager] {mod}: args changed -> restarting "
                          f"({' '.join(stored.get('argv', []))}  ->  {' '.join(self._service_args(mod))})")
                    self.stop([mod])
                else:
                    if mod in ("s1", "s2") and stored.get("argv") not in (None, self._service_args(mod)):
                        print(f"[manager] {mod}: NOTE running with different args "
                              f"({' '.join(stored.get('argv', []))}) — stop+start to apply new ones")
                    print(f"[manager] {mod}: already running (pid {pid})")
                    continue
            if self.health(mod, 0.3):
                print(f"[manager] {mod}: something already answers on its health port (unmanaged?) — skipping")
                continue
            if not self.preflight(mod):
                raise SystemExit(f"[manager] {mod}: preflight failed")
            args = self._service_args(mod)
            env = dict(os.environ)
            env.update(self._mps_env(mod) or {})
            log = open(self._logfile(mod), "ab", buffering=0)
            p = subprocess.Popen(["bash", str(MODULES[mod]["run_sh"]), *args],
                                 cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)     # own group: clean group-kill (incl. C++ runner)
            self._pidfile(mod).write_text(json.dumps(
                {"pid": p.pid, "argv": args, "t0": time.time(), "log": str(self._logfile(mod))}))
            started.append(mod)
            print(f"[manager] {mod}: spawned pid {p.pid}  ({' '.join(args)})  log={self._logfile(mod)}")
        if wait and (started or mods):
            self.wait_ready([m for m in mods if not self._is_remote(m)], timeout_s)
        return started

    def wait_ready(self, mods, timeout_s=300.0):
        t0 = time.time()
        pending = [m for m in mods if m]
        while pending:
            for mod in list(pending):
                h = self.health(mod, 0.5)
                if h and h.get("ready"):
                    print(f"[manager] {mod}: READY after {time.time() - t0:.0f}s "
                          f"(pid {h['pid']}, health rtt {h['rtt_ms']:.1f}ms)")
                    pending.remove(mod)
                    continue
                # no beacon answer AND no live managed pid -> the local process is gone
                # (an UNMANAGED instance counts as alive as long as its beacon answers)
                if h is None and self.pid_of(mod) is None and not self._is_remote(mod):
                    self.logs(mod, 15)
                    raise SystemExit(f"[manager] {mod}: process died during startup (log tail above)")
            if time.time() - t0 > timeout_s:
                raise SystemExit(f"[manager] timeout: {pending} not ready after {timeout_s:.0f}s")
            if pending:
                time.sleep(1.0)

    def stop(self, mods):
        for mod in mods:
            pid = self.pid_of(mod)
            if pid is None:
                print(f"[manager] {mod}: not running (nothing managed to stop)")
                continue
            print(f"[manager] {mod}: SIGTERM group {pid}")
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            for _ in range(100):
                if not _alive(pid):
                    break
                time.sleep(0.1)
            if _alive(pid):
                print(f"[manager] {mod}: still alive -> SIGKILL group")
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self._pidfile(mod).unlink(missing_ok=True)
            print(f"[manager] {mod}: stopped")

    def status(self):
        print(f"{'module':<8}{'state':<12}{'pid':<9}{'ready':<7}{'uptime':<10}{'rtt':<9}endpoints")
        for mod in ALL_MODULES:
            info, ports = MODULES[mod], self.cfg.ports
            host = self._host(mod)
            pid = self.pid_of(mod)
            h = self.health(mod, 0.5)
            if self._is_remote(mod):
                state = "REMOTE"
            elif pid and h:
                state = "RUNNING"
            elif pid:
                state = "LOADING"      # process alive, beacon not answering yet (heavy import)
            elif h:
                state = "UNMANAGED"    # answers health but we hold no pidfile
            else:
                state = "STOPPED"
            up = f"{h['uptime_s']:.0f}s" if h else "-"
            rtt = f"{h['rtt_ms']:.1f}ms" if h else "-"
            eps = " ".join(f"{p}:{ports[p]}" for p in info["data_ports"] + [info["health"]])
            print(f"{mod:<8}{state:<12}{str(pid or (h or {}).get('pid', '-')):<9}"
                  f"{str(bool(h and h.get('ready'))):<7}{up:<10}{rtt:<9}{host} {eps}")

    def logs(self, mod, n=40):
        f = self._logfile(mod)
        if not f.exists():
            print(f"[manager] no log yet: {f}")
            return
        print(f"--- tail -{n} {f} ---")
        with open(f, "r", errors="replace") as fh:
            for line in deque(fh, maxlen=n):
                sys.stdout.write(line)

    def clean(self):
        for port in sorted(set(self.cfg.ports.values())):
            subprocess.run(["fuser", "-k", "-9", f"{port}/tcp"], capture_output=True)
        for pat in ("deploy.s1_module", "deploy.s2_module", "quantization_kit.async_zmq", "llama-s2-runner"):
            subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
        for mod in MODULES:
            self._pidfile(mod).unlink(missing_ok=True)
        time.sleep(1.0)
        print(f"[manager] cleaned: freed ports {sorted(set(self.cfg.ports.values()))} + killed leftovers")

    # ---------- integration testing ----------
    def test(self, suites, keep_up=False, timeout_s=300.0):
        from deploy.tests import report
        from deploy.tests import test_camera, test_e2e, test_protocol, test_s1_module, test_s2_module
        registry = {                      # suite -> (module, required deployed modules)
            "protocol": (test_protocol, set()),
            "s1": (test_s1_module, {"s1"}),
            "s2": (test_s2_module, {"s2"}),
            "e2e": (test_e2e, {"s1", "s2"}),
            "cam": (test_camera, {"cam"}),
        }
        suites = [s for s in ("protocol", "s1", "s2", "e2e", "cam") if s in suites]
        needed = set().union(*(registry[s][1] for s in suites))
        missing = [m for m in ALL_MODULES if m in needed
                   and not (self.health(m, 0.5) or {}).get("ready")]
        auto = []
        if missing:
            print(f"[manager] test needs {missing} -> auto-starting (stopped again after; --keep-up to keep)")
            auto = self.start(missing, wait=True, timeout_s=timeout_s)
        if needed:
            print("[manager] note: do not run tests while a live demo/orchestrator is consuming "
                  "s1_out (PUSH round-robins across consumers)")
        all_ok = True
        try:
            for s in suites:
                all_ok &= report(registry[s][0].run(self.cfg), s)
        finally:
            if auto and not keep_up:
                self.stop(auto)
        print(f"\n[manager] integration test: {'ALL PASS' if all_ok else 'FAILURES'} "
              f"(suites: {', '.join(suites)})")
        return all_ok

    # ---------- demo (the real control loop over the deployed modules) ----------
    def demo(self, ns):
        self._assert_no_live_session()
        missing = [m for m in CORE if not (self.health(m, 0.5) or {}).get("ready")]
        if missing:
            print(f"[manager] demo needs {missing} -> starting")
            self.start(missing, wait=True, timeout_s=ns.timeout)
        from deploy import orchestrator   # deploy-owned: raw trajs + synced-frame pairing
        sys.argv = ["orchestrator", "--seconds", str(ns.seconds), "--hz", str(ns.hz),
                    "--instruction", ns.instruction or self.cfg.instruction,
                    "--host", "127.0.0.1", "--s2-host", self.cfg.s2_host, "--s1-host", self.cfg.s1_host,
                    "--s2-period-s", str(ns.s2_period_s),
                    "--sys2-max-forward-step", str(ns.sys2_max_forward_step)]
        if ns.faithful:
            sys.argv.append("--faithful")
        orchestrator.main()
        print("\n[manager] demo done — modules left running (deploy/run_manager.sh stop)")

    def _assert_no_live_session(self):
        """The orchestrator binds the prompt port; a second live session would crash on it."""
        import zmq
        s = zmq.Context.instance().socket(zmq.PULL)
        s.setsockopt(zmq.LINGER, 0)
        try:
            s.bind(f"tcp://*:{self.cfg.ports['prompt']}")
        except zmq.ZMQError:
            raise SystemExit(f"[manager] another live camera/demo session is already running "
                             f"(prompt port {self.cfg.ports['prompt']} is busy) — Ctrl-C it first, "
                             f"or send it instructions with:  run_manager.sh prompt \"...\"")
        finally:
            s.close()

    # ---------- camera (LIVE: camera -> S2/S1 -> actions, with the 3-panel viewer) ----------
    def camera(self, ns):
        self._assert_no_live_session()
        mods = ["s2", "s1", "cam"] + ([] if ns.no_viz else ["viz"])
        missing = [m for m in mods if not (self.health(m, 0.5) or {}).get("ready")]
        if missing:
            print(f"[manager] camera needs {missing} -> starting")
            self.start(missing, wait=True, timeout_s=ns.timeout)
        view = self.cfg.viz_view
        if view == "auto":
            view = "tk" if os.environ.get("DISPLAY") else "save"
        if not ns.no_viz:
            hint = {"mjpeg": f"open  http://<this-host-ip>:{self.cfg.mjpeg_port}/  in a browser",
                    "tk": "a native window should be open (press q to close)",
                    "save": f"headless -> watch {QK / 'dbg' / 'viz_live.png'} (~2 Hz)"}[view]
            print(f"[manager] viewer ({view}): {hint}")
        print('[manager] change the instruction live:  deploy/run_manager.sh prompt "<text>"\n')
        from deploy import orchestrator   # deploy-owned: raw trajs + synced-frame pairing
        sys.argv = ["orchestrator", "--camera", "--seconds", str(ns.seconds),
                    "--instruction", ns.instruction or self.cfg.instruction,
                    "--host", "127.0.0.1", "--s2-host", self.cfg.s2_host,
                    "--s1-host", self.cfg.s1_host, "--cam-host", self.cfg.cam_host,
                    "--s2-period-s", str(ns.s2_period_s),
                    "--sys2-max-forward-step", str(ns.sys2_max_forward_step)]
        if ns.faithful:
            sys.argv.append("--faithful")
        try:
            orchestrator.main()
        finally:
            print("\n[manager] camera session done — modules left running (deploy/run_manager.sh stop)")

    def prompt(self, text):
        """Push a new instruction to the live orchestrator (camera/demo must be running)."""
        import zmq
        ctx = zmq.Context.instance()
        push = ctx.socket(zmq.PUSH)
        push.setsockopt(zmq.LINGER, 1000)          # give the queued message 1 s to flush on close
        push.connect(f"tcp://127.0.0.1:{self.cfg.ports['prompt']}")
        push.send_string(text)
        time.sleep(0.1)
        push.close()
        print(f"[manager] instruction sent: {text!r} (a live camera/demo session will pick it up)")


# ==========================================================================================
def _add_knobs(sp):
    """Service knobs shared by start/test/demo; None -> keep DeployConfig/env default."""
    sp.add_argument("--device")
    sp.add_argument("--s2-model", choices=["q4", "f16"])
    sp.add_argument("--s2-resize", type=int)
    sp.add_argument("--num-history", type=int)
    sp.add_argument("--kv-reuse", action=argparse.BooleanOptionalAction, default=None)
    sp.add_argument("--s1-steps", type=int)
    sp.add_argument("--no-trt-s1", action="store_true")
    sp.add_argument("--s2-mps-pct", type=int)
    sp.add_argument("--timeout", type=float, default=300.0, help="model-load readiness timeout (s)")


def _apply_knobs(cfg, ns):
    for src, dst in (("device", "device"), ("s2_model", "s2_model"), ("s2_resize", "s2_resize"),
                     ("num_history", "num_history"), ("kv_reuse", "kv_reuse"),
                     ("s1_steps", "s1_steps"), ("s2_mps_pct", "s2_mps_pct")):
        v = getattr(ns, src, None)
        if v is not None:
            setattr(cfg, dst, v)
    if getattr(ns, "no_trt_s1", False):
        cfg.trt_s1 = False


def _mods(arg, all_means):
    return list(all_means) if arg in (None, "all") else [arg]


def main():
    ap = argparse.ArgumentParser(prog="deploy/run_manager.sh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("start", help="spawn module daemons ('all' = the s2+s1 core)")
    sp.add_argument("module", nargs="?", choices=["s1", "s2", "cam", "viz", "all"], default="all")
    sp.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    _add_knobs(sp)

    sp = sub.add_parser("stop", help="stop managed modules ('all' = every managed module)")
    sp.add_argument("module", nargs="?", choices=["s1", "s2", "cam", "viz", "all"], default="all")

    sub.add_parser("status", help="module health table")

    sp = sub.add_parser("test", help="run integration test suites")
    sp.add_argument("--suite", nargs="+", choices=["protocol", "s1", "s2", "e2e", "cam", "all"],
                    default=["all"])
    sp.add_argument("--keep-up", action="store_true", help="leave auto-started modules running")
    _add_knobs(sp)

    sp = sub.add_parser("demo", help="run the async control loop on static frames")
    sp.add_argument("--seconds", type=float, default=8.0)
    sp.add_argument("--hz", type=float, default=30.0)
    sp.add_argument("--instruction", default=None)
    sp.add_argument("--s2-period-s", type=float, default=0.0)
    sp.add_argument("--sys2-max-forward-step", type=int, default=1)
    sp.add_argument("--faithful", action="store_true")
    _add_knobs(sp)

    sp = sub.add_parser("camera", help="LIVE: camera -> S2/S1 -> actions, with the 3-panel viewer")
    sp.add_argument("--seconds", type=float, default=0.0, help="0 = run until Ctrl-C")
    sp.add_argument("--instruction", default=None)
    sp.add_argument("--s2-period-s", type=float, default=0.0)
    sp.add_argument("--sys2-max-forward-step", type=int, default=8)
    sp.add_argument("--faithful", action="store_true")
    sp.add_argument("--no-viz", action="store_true", help="skip the viewer module")
    sp.add_argument("--view", choices=["auto", "tk", "mjpeg", "save"], default=None,
                    help="viewer backend (tk window | mjpeg browser | save png)")
    sp.add_argument("--mjpeg-port", type=int, default=None)
    sp.add_argument("--cam-device", type=int, default=None, help="/dev/videoN index")
    sp.add_argument("--cam-hfov", type=float, default=None, help="projection: HFOV of the square frame (deg)")
    sp.add_argument("--cam-height", type=float, default=None, help="projection: camera height (m)")
    sp.add_argument("--cam-pitch", type=float, default=None, help="projection: camera pitch, +down (deg)")
    _add_knobs(sp)

    sp = sub.add_parser("prompt", help="send a new instruction to a live camera/demo session")
    sp.add_argument("text", nargs="+", help="the instruction words")

    sp = sub.add_parser("logs", help="tail a module log")
    sp.add_argument("module", choices=["s1", "s2", "cam", "viz"])
    sp.add_argument("-n", type=int, default=40)

    sub.add_parser("clean", help="force-free deploy ports + kill leftover processes")

    ns = ap.parse_args()
    cfg = DeployConfig.from_env()
    _apply_knobs(cfg, ns)
    if getattr(ns, "view", None):
        cfg.viz_view = ns.view
    if getattr(ns, "mjpeg_port", None):
        cfg.mjpeg_port = ns.mjpeg_port
    if getattr(ns, "cam_device", None) is not None:
        cfg.cam_device = ns.cam_device
    for flag, attr in (("cam_hfov", "cam_hfov_deg"), ("cam_height", "cam_height_m"),
                       ("cam_pitch", "cam_pitch_deg")):
        v = getattr(ns, flag, None)
        if v is not None:
            setattr(cfg, attr, v)
    mgr = Manager(cfg)

    if ns.cmd == "start":
        mgr.start(_mods(ns.module, CORE), wait=ns.wait, timeout_s=ns.timeout)
    elif ns.cmd == "stop":
        mgr.stop(_mods(ns.module, ALL_MODULES))
    elif ns.cmd == "status":
        mgr.status()
    elif ns.cmd == "test":
        suites = list(ns.suite)
        if "all" in suites:                       # cam joins `all` only when a camera is present
            suites = ["protocol", "s1", "s2", "e2e"]
            if Path(f"/dev/video{cfg.cam_device}").exists():
                suites.append("cam")
        sys.exit(0 if mgr.test(suites, keep_up=ns.keep_up, timeout_s=ns.timeout) else 1)
    elif ns.cmd == "demo":
        mgr.demo(ns)
    elif ns.cmd == "camera":
        mgr.camera(ns)
    elif ns.cmd == "prompt":
        mgr.prompt(" ".join(ns.text))
    elif ns.cmd == "logs":
        mgr.logs(ns.module, ns.n)
    elif ns.cmd == "clean":
        mgr.clean()


if __name__ == "__main__":
    main()
