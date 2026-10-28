"""Deploy orchestrator — the single non-blocking control loop over the deployed modules.

Fork of quantization_kit/async_zmq/orchestrator.py with the deploy wire (protocol v2):

  * S1 replies carry the DENOISED robot-facing waypoint path (`waypoints` [T+1, 2],
    metric robot frame — vln_utils.traj_to_actions(use_discrate_action=False)); a legacy
    `trajs` [N, T+1, 2] reply is accepted and reduced with the same np.mean.
  * Every S1 reply is seq-matched and PAIRED with the exact camera frame that was sent
    for that inference (`s1_frame`), so the viewer's S1 panel renders the waypoints on
    the frame they were computed from — never on a newer live frame.
  * The viz overlay keeps S2's pixel goal in the policy's native [row, col] convention.

Loop per tick: drain S2 (never block) -> fire S2 if due -> S1 act on the latest plan ->
emit action + publish viz overlay -> hold the control rate.
"""
import argparse
import collections
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import zmq
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantization_kit.async_zmq.zmq_proto import DEFAULT_PORTS, make_socket, recv_msg, send_msg  # noqa: E402

ACT = {-1: "·", 0: "STOP", 1: "↑", 2: "←", 3: "→", 5: "↓"}
_QK = _ROOT / "quantization_kit"


class FrameSource:
    """Static RGB frames (demo fallback when no live camera). Real deployment uses --camera."""

    def __init__(self, frames_dir=None):
        self.imgs = []
        if frames_dir:
            for p in sorted(glob.glob(os.path.join(frames_dir, "*.jpg")) +
                            glob.glob(os.path.join(frames_dir, "*.png"))):
                self.imgs.append(np.array(Image.open(p).convert("RGB")))
        if not self.imgs:
            test = _QK / "dbg" / "test448.png"
            self.imgs = [np.array(Image.open(test).convert("RGB"))]
        self.lookdown = self.imgs[-1]

    def __call__(self, tick):
        return self.imgs[tick % len(self.imgs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--s2-host", default=None)
    ap.add_argument("--s1-host", default=None)
    ap.add_argument("--cam-host", default=None)
    ap.add_argument("--seconds", type=float, default=8.0, help="run duration; <=0 = run forever")
    ap.add_argument("--hz", type=float, default=30.0, help="target S1 / control rate")
    ap.add_argument("--instruction", default="Walk forward along the hallway and stop near the table.")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--camera", action="store_true", help="consume live frames from the camera module")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--sys2-max-forward-step", type=int, default=8)
    ap.add_argument("--s2-period-s", type=float, default=0.0)
    ap.add_argument("--faithful", action="store_true",
                    help="honour S2's natural discrete/pixel choice (no force_latent demo aid)")
    args = ap.parse_args()

    src = FrameSource(args.frames_dir)
    s2_host, s1_host = args.s2_host or args.host, args.s1_host or args.host
    ctx = zmq.Context.instance()
    s2_in = make_socket(ctx, zmq.PUSH, DEFAULT_PORTS["s2_in"], bind=False, host=s2_host)
    s2_out = make_socket(ctx, zmq.SUB, DEFAULT_PORTS["s2_out"], bind=False, host=s2_host)
    s1_in = make_socket(ctx, zmq.PUSH, DEFAULT_PORTS["s1_in"], bind=False, host=s1_host)
    s1_out = make_socket(ctx, zmq.PULL, DEFAULT_PORTS["s1_out"], bind=False, host=s1_host)
    poll_s2 = zmq.Poller(); poll_s2.register(s2_out, zmq.POLLIN)
    poll_s1 = zmq.Poller(); poll_s1.register(s1_out, zmq.POLLIN)

    cam_sub = poll_cam = None
    if args.camera:
        cam_sub = make_socket(ctx, zmq.SUB, DEFAULT_PORTS["cam"], bind=False, host=args.cam_host or args.host)
        poll_cam = zmq.Poller(); poll_cam.register(cam_sub, zmq.POLLIN)
    prompt_pull = make_socket(ctx, zmq.PULL, DEFAULT_PORTS["prompt"], bind=True)
    poll_prompt = zmq.Poller(); poll_prompt.register(prompt_pull, zmq.POLLIN)
    viz_pub = None if args.no_viz else make_socket(ctx, zmq.PUB, DEFAULT_PORTS["viz"], bind=True)
    time.sleep(0.4)   # let SUB/PUSH connections settle

    # ---- control state (single owner; no locks) ----
    instruction = args.instruction
    latent = goal_rgb = None
    latent_seq = -1
    latent_born = 0
    act_queue = collections.deque()
    s2_inflight = False
    s2_fire_tick = 0
    s2_fire_wall = -1e9
    dual_forward_step = 0
    s2_seq = s1_seq = 0
    # ---- viz state (waypoints stay PAIRED with the frame they were computed on) ----
    latest_frame = None
    waypoints = None             # DENOISED [T+1, 2] robot-facing path from the S1 service
    s1_frame = None              # the exact frame `waypoints` was computed from (synced bg)
    cur_pixel = None             # S2 pixel goal, native [row, col]
    cur_s2_text = ""

    # ---- stats ----
    t_hist = collections.deque(maxlen=int(args.hz))
    s1_ms_ema = None
    s2_fires = s2_dones = 0
    s2_ms_sum = 0.0
    overlaps = []
    last_action = -1

    def should_fire_s2(now):
        if s2_inflight:
            return False
        if now - s2_fire_wall < args.s2_period_s:
            return False
        if latent is None and not act_queue:
            return True
        return dual_forward_step >= args.sys2_max_forward_step

    dt = 1.0 / args.hz
    forever = args.seconds <= 0
    n_ticks = None if forever else int(args.seconds * args.hz)
    verbose = not args.camera and not forever
    viz_str = "off" if args.no_viz else f"PUB:{DEFAULT_PORTS['viz']}"
    print(f"\n[orch] target {args.hz:.0f} Hz | source={'CAMERA' if args.camera else 'static'} | "
          f"viz={viz_str} | prompt=PULL:{DEFAULT_PORTS['prompt']} | protocol=v2 (raw trajs + synced frame)")
    print(f"[orch] instruction={instruction!r}  ({'FAITHFUL' if args.faithful else 'force_latent demo'})", flush=True)
    if verbose:
        hdr = "  tick   Hz  |  S1 act  ms    |  S2"
        print(hdr); print("  " + "-" * (len(hdr)))

    wall0 = time.perf_counter()
    tick = 0
    try:
        while forever or tick < n_ticks:
            t0 = time.perf_counter()

            # 0a) freshest live camera frame
            if cam_sub is not None:
                while poll_cam.poll(0):
                    _, a = recv_msg(cam_sub)
                    latest_frame = a["rgb"]
                rgb = latest_frame if latest_frame is not None else src(tick)
            else:
                rgb = src(tick)
            lookdown = rgb if args.camera else src.lookdown

            # 0b) live instruction updates
            while poll_prompt.poll(0):
                try:
                    instruction = prompt_pull.recv_string()
                except Exception:
                    break
                latent = None; act_queue.clear(); dual_forward_step = 0
                cur_pixel = None
                print(f"[orch] >>> new instruction: {instruction!r}", flush=True)

            s2_note = "idle"

            # 1) drain S2 outputs (async; never blocks)
            while poll_s2.poll(0):
                meta, arr = recv_msg(s2_out)
                s2_inflight = False
                s2_dones += 1
                s2_ms_sum += meta.get("t_infer_ms", 0.0)
                overlap = tick - s2_fire_tick
                overlaps.append(overlap)
                cur_s2_text = (meta.get("text") or "")[:80]
                if meta["kind"] == "latent":
                    latent = arr["latent"]; goal_rgb = arr["goal_rgb"]
                    latent_seq = meta["seq"]; latent_born = tick
                    dual_forward_step = 0
                    act_queue.clear()
                    cur_pixel = meta.get("pixel")            # [row, col] (policy-native)
                    s2_note = f"✓ DONE #{meta['seq']} latent px={cur_pixel} {meta['t_infer_ms']:.0f}ms (ovl {overlap})"
                else:
                    act_queue.extend(meta.get("action") or [])
                    cur_pixel = None
                    s2_note = f"✓ DONE #{meta['seq']} action={meta.get('action')} {meta['t_infer_ms']:.0f}ms"

            # 2) fire S2 if due (fire-and-forget)
            if should_fire_s2(t0):
                send_msg(s2_in, {"seq": s2_seq, "mode": "infer", "instruction": instruction,
                                 "force_latent": (not args.faithful)},
                         {"rgb": rgb.astype(np.uint8), "lookdown_rgb": lookdown.astype(np.uint8)})
                s2_inflight = True; s2_fire_tick = tick; s2_fire_wall = t0; s2_fires += 1
                if s2_note == "idle":
                    s2_note = f"★ FIRE #{s2_seq} (bg)"
                s2_seq += 1
            elif s2_inflight:
                s2_note = f"running… ({tick - s2_fire_tick}t)"

            # 3) this tick's action
            s1_ms = None
            if act_queue:
                action = act_queue.popleft(); srctag = "S2"
                dual_forward_step += 1
            elif latent is not None:
                sent_frame = rgb.astype(np.uint8)            # remember EXACTLY what S1 sees
                send_msg(s1_in, {"seq": s1_seq, "cache_key": int(latent_seq)},
                         {"latent": latent, "rgb_mem": goal_rgb, "rgb_cur": sent_frame})
                action = last_action
                deadline = t0 + 0.5
                while time.perf_counter() < deadline:        # wait for OUR seq, dropping stale replies
                    if not poll_s1.poll(max(1, int((deadline - time.perf_counter()) * 1000))):
                        break
                    m1, a1 = recv_msg(s1_out)
                    if m1.get("seq") != s1_seq:
                        continue                             # an earlier timed-out reply -> skip it
                    acts = m1.get("actions") or []
                    action = acts[0] if acts else -1
                    s1_ms = m1.get("t_infer_ms")
                    wp = a1.get("waypoints")
                    if wp is None and "trajs" in a1:         # legacy raw-fan reply -> same mean
                        wp = a1["trajs"].mean(axis=0)
                    if wp is not None:                       # PAIR the path with its input frame
                        waypoints = wp
                        s1_frame = sent_frame
                    s1_ms_ema = s1_ms if s1_ms_ema is None else 0.9 * s1_ms_ema + 0.1 * s1_ms
                    break
                srctag = "S1"; s1_seq += 1
                dual_forward_step += 1
            else:
                action = -1; srctag = "--"

            last_action = action

            # 4) hold the control rate + live S1 Hz
            t_hist.append(time.perf_counter())
            hz = (len(t_hist) - 1) / (t_hist[-1] - t_hist[0]) if len(t_hist) > 1 else 0.0

            # 5) viz overlay (~15 Hz): raw trajs + the SYNCED frame they belong to
            if viz_pub is not None and tick % 2 == 0:
                ov = {"action": int(action), "src": srctag, "instr": instruction,
                      "pixel": (list(cur_pixel) if cur_pixel is not None else None),
                      "inflight": bool(s2_inflight),
                      "latent_age": (tick - latent_born) if latent is not None else -1,
                      "s1_hz": round(hz, 1),
                      "s2_hz": round(s2_dones / max(time.perf_counter() - wall0, 1e-6), 2),
                      "s2_text": cur_s2_text}
                arrays = {}
                if waypoints is not None and s1_frame is not None:
                    arrays = {"waypoints": np.asarray(waypoints, np.float32), "s1_frame": s1_frame}
                send_msg(viz_pub, ov, arrays)

            if verbose or ("DONE" in s2_note) or ("FIRE" in s2_note) or (tick % int(args.hz) == 0):
                ms_str = f"{s1_ms:4.1f}" if s1_ms is not None else "  - "
                age = f"age={tick - latent_born}" if latent is not None else ""
                print(f"  {tick:5d} {hz:5.1f} | {srctag:>2s} {ACT.get(action, '?'):>4s} {ms_str} {age:>7s} | {s2_note}",
                      flush=True)

            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
            tick += 1
    except KeyboardInterrupt:
        print("\n[orch] interrupted", flush=True)

    wall = time.perf_counter() - wall0
    print("\n=== async control summary ===")
    print(f"  wall {wall:.1f}s | {tick} control ticks")
    print(f"  S1: {s1_seq} infers @ {s1_seq / wall if wall else 0:4.1f} Hz  (mean {s1_ms_ema or 0:.1f} ms/infer)")
    if s2_dones:
        print(f"  S2: {s2_fires} fired / {s2_dones} done @ {s2_dones / wall:4.2f} Hz  (mean {s2_ms_sum / s2_dones:.0f} ms/infer)")
    for s in (s2_in, s2_out, s1_in, s1_out, prompt_pull):
        s.close()
    if cam_sub is not None:
        cam_sub.close()
    if viz_pub is not None:
        viz_pub.close()


if __name__ == "__main__":
    main()
