# deploy/ — field-deployment packaging of the InternVLA-N1 dual system

Three deployable units, wired **only** by ZMQ messages (no shared memory, no locks, no
shared process). Each unit has its own run script; the manager (master) does lifecycle,
health, integration tests, and the live demo.

```
deploy/
├── common/
│   ├── config.py        # single source of truth: hosts/ports/knobs (VLN_* env-overridable)
│   ├── interface.py     # THE wire contract: schemas + validators + S1Client/S2Client/Health*
│   └── env.sh           # shared launcher env (arch-aware build dir, Jetson dist-packages)
├── s2_module/           # System-2: Qwen2.5-VL GGUF via llama-s2-runner (~1 Hz reasoner)
│   ├── service.py       #   canonical s2_service + out-of-band HealthBeacon (REP :5615)
│   └── run.sh           #   standalone foreground launch (flags pass through)
├── s1_module/           # System-1: lite diffusion + TensorRT (~30 Hz controller)
│   ├── service.py       #   canonical s1_service + HealthBeacon (REP :5614)
│   └── run.sh
├── camera_module/       # V4L2 camera -> PUB rgb frames :5605 (+ HealthBeacon :5616)
├── viewer_module/       # live 3-panel visualizer, SUB-only observer (+ HealthBeacon :5617)
├── manager.py           # master: start/stop/status/test/demo/camera/prompt/logs/clean
├── run_manager.sh
├── package.sh           # build the fully SELF-CONTAINED bundle (copies everything)
├── tools/extract_s1_weights.py   # 16 GB DualVLN ckpt -> 0.18 GB System-1-only safetensors
└── tests/               # per-module + integration suites (also standalone runnable)
```

The S2 module wraps the canonical `quantization_kit/async_zmq/s2_service.py` unchanged.
The **S1 service, orchestrator and viewer are deploy-owned** (`s1_module/service.py`,
`orchestrator.py`, `viewer_module/visualizer.py`) — protocol v2: raw diffusion
trajectories, synced-frame pairing, [row,col] pixel convention, image-plane projection.

## Quickstart

```bash
deploy/run_manager.sh test --suite protocol   # wire-format sanity, no GPU, <5 s
deploy/run_manager.sh start --wait            # spawn S2+S1 daemons, block until models loaded
deploy/run_manager.sh status                  # health table (safe while anything is live)
deploy/run_manager.sh test --keep-up          # protocol, s1, s2, e2e (+cam when a camera exists)
deploy/run_manager.sh demo --seconds 8        # the async control loop on static frames
deploy/run_manager.sh camera --view mjpeg     # LIVE: camera -> S2/S1 -> actions + 3-panel viewer
deploy/run_manager.sh prompt "go to the door" # change a live session's instruction on the fly
deploy/run_manager.sh stop                    # graceful shutdown (kills the C++ runner too)
deploy/run_manager.sh clean                   # recovery: force-free ports + kill leftovers
```

`camera` brings up s2+s1+cam(+viz) and runs the orchestrator on live frames. Viewer
backends: `--view tk` (native window), `--view mjpeg` (browser at `http://<ip>:8089/`,
works headless/remote), `--view save` (writes `quantization_kit/dbg/viz_live.png` ~2 Hz).
The camera publishes at the S2 resolution (`--s2-resize`) so pixel goals map 1:1.

**Viewer panels** (`deploy/viewer_module/visualizer.py`, deploy-owned): LEFT = the exact
frame S2 processed + its pixel goal (policy-native **[row, col]** — the old [x,y] reading
transposed the marker and broke the paper's S1↔S2 goal-pixel agreement); CENTER = live
camera + action; RIGHT = **the frame S1 computed on (synced, not live)** with the
**denoised robot-facing waypoint path** (`traj_to_actions(use_discrate_action=False)`,
the same trajectory the discrete actions come from) **projected** through the repo camera
model (`deploy/common/camproj.py`: intrinsics-from-HFOV per `get_intrinsic_matrix`,
ground geometry per `unproject_depth`) plus a `S1-S2 goal: Δpx / Δm` agreement readout
whenever S2 emitted a pixel goal, and an **always-visible autoscaled BEV inset** of the
path (so waypoints can never disappear entirely on a miscalibrated camera). Calibrate the
projection per robot with `VLN_CAM_HEIGHT` (default 1.25 m, repo), `VLN_CAM_PITCH`
(+down, default 0), `VLN_CAM_HFOV` (default 90° = the habitat/VLN-CE training camera;
the BEV sensor's fx=585 would be 44.6° after the square crop) or the `camera` flags
`--cam-height/--cam-pitch/--cam-hfov`. The manager auto-restarts a running cam/viz
daemon when its args changed, so new calibration always takes effect.

`test` auto-starts whatever modules it needs and stops them again afterwards (unless
`--keep-up`). Logs land in `deploy/logs/<mod>.log` (`run_manager.sh logs s2 -n 60`),
pidfiles in `deploy/run/`.

Standalone module runs (foreground, e.g. one per terminal / per machine):

```bash
deploy/s2_module/run.sh --kv-reuse                 # flags pass straight to the service
deploy/s1_module/run.sh --s1-steps 5
convenv/bin/python -m deploy.tests.test_s1_module  # one suite against running modules
```

## Interface contract (summary — authoritative doc in `common/interface.py`)

| port | name | pattern | carries |
|---|---|---|---|
| 5601 | `s2_in`  | PUSH→PULL | `{seq, mode: infer/accumulate/reset, instruction, force_latent}` + `rgb`, `lookdown_rgb` (u8 HWC) |
| 5602 | `s2_out` | PUB→SUB   | `kind=latent`: `latent` f32 [1,4,3584] + `goal_rgb` + `pixel` \| `kind=action`: discrete ids |
| 5603 | `s1_in`  | PUSH→PULL | `{seq, cache_key}` + `latent` f32 [1,4,3584], `rgb_mem` (goal), `rgb_cur` |
| 5604 | `s1_out` | PUSH→PULL | `{seq, actions (≤4, STOP filtered), t_infer_ms}` + `waypoints` f32 **[T+1,2]** — the DENOISED robot-facing path, verbatim `vln_utils.traj_to_actions(use_discrate_action=False)` (metric robot frame; the same trajectory the discrete actions come from) |
| 5607 | `viz` | PUB→SUB | overlay meta (`pixel` = policy-native **[row, col]**) + `waypoints` **paired with `s1_frame`**, the exact frame S1 computed them on |
| 5610 | `s2_runner` | REQ↔REP | private: S2 service ↔ C++ `llama-s2-runner --zmq` |
| 5614/5615 | `s1_health`/`s2_health` | REQ↔REP | **deploy-only**, out-of-band: `{cmd:ping}` → `{ok, module, pid, ready, uptime_s}` |

Framing: msgpack header + raw numpy frames (`quantization_kit/async_zmq/zmq_proto.py`).
Action space: `0=STOP 1=forward 2=left 3=right 5=look_down`. Consumers should use
`S2Client`/`S1Client` from `deploy.common.interface` rather than raw sockets.

**Rules of the road**
- Health ports are always safe to probe (`status` works during a live demo). The **data**
  ports are not: never attach a second PULL to `s1_out` while an orchestrator is live
  (PUSH round-robins across consumers) — so don't run `test` during `demo`.
- One S2 request in flight at a time (the service is stateful: history + runner KV).
- `force_latent` is a demo/test aid guaranteeing a latent reply; real navigation should
  run `--faithful` and honour S2's discrete-action choice.

## Test suites (`run_manager.sh test --suite ...`)

| suite | needs | verifies |
|---|---|---|
| `protocol` | nothing | msgpack+numpy framing round-trips bit-exactly (incl. unicode, non-contiguous) |
| `s1` | S1 | health/ready · zeros+random latent → valid actions/waypoints · sustained throughput |
| `s2` | S2 | health/ready · cold+warm infer → latent [1,4,3584] finite/non-zero, pixel in-bounds · accumulate · reset recovery · state evolves |
| `e2e` | both | full chain S2→latent→S1→actions · **overlap**: S1 keeps ticking while S2 infers (the split's core promise) · replan switch |
| `cam` | camera | health/ready · frame contract (square, 112-aligned, u8, fresh, non-flat) · stream rate + seq ordering — joins `all` automatically when `/dev/video<N>` exists |

## Self-contained bundle (`deploy/package.sh`)

`deploy/package.sh [TARGET]` (default `~/vln-deploy`) builds a **fully independent runtime
folder** — every artifact is *copied*, nothing links back to this tree: code (`internnav/`,
`quantization_kit` python, `deploy/`), the `convenv` venv, the **whole `llama.cpp/`**
(source + built runners, minus `.git`), the Q4-latent + mmproj GGUFs (`--full-gguf` adds
f16-latent), the native TRT engines + `onnx/` (for rebuilds), DepthAnythingV2, and a
**System-1-only checkpoint** (`model-s1only.safetensors`, 0.18 GB extracted from the 16 GB
DualVLN — S2 runs from GGUF so its safetensors weights never load on-device; `--full-ckpt`
copies the shards instead). Total ≈ **13 GB**. The bundle mirrors the repo layout, so every
relative-path assumption in the code holds unchanged; run everything via the bundle-root
`./run.sh` (same manager CLI). Verified: bundle processes map **zero** files from the dev
tree (`grep VLN-field /proc/<pid>/maps` = 0). Remaining system deps: JetPack CUDA/TensorRT
(`/usr/lib/python3.12/dist-packages`), `/usr/bin/python3.12`.

## Distributed deployment

Modules bind `tcp://*`, so they are remote-ready as-is. Example — S2 on the x86 box,
S1 + manager on the robot:

```bash
# x86 box:
deploy/s2_module/run.sh
# robot (Jetson):
VLN_S2_HOST=<x86-ip> deploy/run_manager.sh start s1 --wait
VLN_S2_HOST=<x86-ip> deploy/run_manager.sh test --suite e2e --keep-up
VLN_S2_HOST=<x86-ip> deploy/run_manager.sh demo --seconds 0
```

The manager never spawns/stops remote modules — it only probes and tests them.

## Knobs

CLI flags on `start`/`test`/`demo` (or `VLN_*` env, see `common/config.py`):
`--s2-model q4|f16` · `--s2-resize 224/336/448` (112-aligned!) · `--kv-reuse/--no-kv-reuse` ·
`--s1-steps N` · `--no-trt-s1` · `--s2-mps-pct N` (0 = no MPS; needs the daemon: `source mps.sh`).

## Troubleshooting

- `Address already in use` in a module log → leftovers holding ports: `run_manager.sh clean`.
- `status` shows `LOADING` for minutes → check `logs s2`; GGUF load + CUDA warm can take
  1–3 min on the first run.
- S1 slow (>100 ms) while S2 infers → single-GPU contention; cap S2 with MPS
  (`--s2-mps-pct`, see `quantization_kit/async_zmq/README.md` for the measured Pareto)
  or move S2 off-device (`VLN_S2_HOST`).
