# Async dual-system over ZMQ (S1 ⟂ S2)

Run System-1 (~fast, diffusion+TRT) and System-2 (~slow, Qwen2.5-VL GGUF) as **independent
processes**, each with its own ZMQ input/output ports, so the slow S2 never blocks the fast
S1 control loop.

## Why

The in-process agent (`internnav/agent/internvla_n1_agent.py`) runs S2 in a background thread
and S1 in the main thread, but they share one `self.policy`, one GPU context, the GIL, and a
mutable `s2_output` guarded by three locks. Worse, the step loop literally blocks on S2:

```python
while self.s2_output.is_infering:   # <- S1/env loop STALLS for the whole ~1-2 s S2 inference
    time.sleep(0.5)
```

So S2's outputs and S1's outputs are *entangled* and cannot run at independent rates. This
package splits them into three processes connected only by messages — no shared memory, no
locks, and the control loop **never waits** for S2 (it always steers S1 off the latest plan).

## Topology — 4 orchestrator-facing ports + 1 private runner port

```
                 ORCHESTRATOR  (orchestrator.py — single owner of control state, no locks)
                 │  PUSH s2_in:5601      SUB s2_out:5602     PUSH s1_in:5603     PULL s1_out:5604
                 ▼                       ▲                   ▼                   ▲
   ┌─────────────────────────────────────────────┐   ┌───────────────────────────────────────┐
   │ S2 SERVICE  (s2_service.py, ~0.5-1 Hz)       │   │ S1 SERVICE  (s1_service.py, ~15-30 Hz) │
   │  PULL :5601  →  S2 brain  →  PUB :5602        │   │  PULL :5603  →  S1 (TRT)  →  PUSH :5604 │
   │  · prompt / sampled history / look-down 2-step│   │  · lite diffusion + native TRT FP16    │
   │  · pixel goal + 4×latent extraction           │   │  · latent + goal/cur RGB → discrete    │
   │  └── llama-s2-runner --zmq tcp://*:5610 ◄─REP─┘   │    actions (0/1/2/3/5)                  │
   │      (C++ fork, NEW ZMQ mode, REP socket)     │   └───────────────────────────────────────┘
   └───────────────────────────────────────────────┘
```

| port | dir | pattern | carries |
|---|---|---|---|
| 5601 `s2_in`  | orch → S2 | PUSH→PULL | frame + instruction (+ look-down frame); `mode=infer/accumulate/reset` |
| 5602 `s2_out` | S2 → orch | PUB→SUB   | `kind=latent` (pixel goal + `[1,4,3584]` latent + goal RGB) **or** `kind=action` (discrete) |
| 5603 `s1_in`  | orch → S1 | PUSH→PULL | latent + goal RGB + current RGB |
| 5604 `s1_out` | S1 → orch | PUSH→PULL | discrete actions `[:4]` + timing |
| 5610 `s2_runner` | S2 svc ↔ C++ | REQ↔REP | the runner line protocol `RESET / TEXT / LATENT / QUIT` |

PUB/SUB is used only for S2 telemetry (latency-tolerant, lets a passive monitor subscribe);
the fast S1 action path is PUSH/PULL so no message is ever dropped.

## Run

```bash
cd quantization_kit
./run.sh async                      # DEFAULTS = CONTINUOUS S2 (back-to-back, ~0.65 Hz, no idle gap) + MPS cap50 -> S1 ~16 Hz (always contended)
./run.sh async --s2-period-s 3      # throttle S2 -> idle windows where S1 runs full-speed (~21 Hz avg S1; plans up to 3 s old)
./run.sh async --s2-mps-pct 0       # opt OUT of the MPS cap (uncapped; S1 dips to ~12.7 Hz while S2 runs)
./run.sh async --faithful           # honour S2's real discrete/pixel choice (no force_latent demo aid)
./run.sh async --no-trt-s1          # S1 in PyTorch fp32 instead of TensorRT
```

Each component is independently launchable (e.g. on different GPUs/hosts):

```bash
PYTHONPATH=/usr/lib/python3.12/dist-packages convenv/bin/python -m quantization_kit.async_zmq.s2_service
PYTHONPATH=/usr/lib/python3.12/dist-packages convenv/bin/python -m quantization_kit.async_zmq.s1_service
PYTHONPATH=/usr/lib/python3.12/dist-packages convenv/bin/python -m quantization_kit.async_zmq.orchestrator --seconds 8
```

### Single-GPU contention, and how to deal with it

S1 and S2 share the Thor GPU and its unified LPDDR, so while S2 decodes the 7B LLM both fight over the
**one memory bus**. Measured: S1 ~34 ms (30 Hz) when S2 is idle, but ~80 ms (~12.4 Hz) **while S2 runs —
a ~2.4x dip**. Options (none free; the dip is bandwidth-bound):

```bash
./run.sh async --s2-period-s 3      # fire S2 rarely -> S1 at full 30 Hz between occasional dips
./run.sh async --s2-q4              # Q4 S2: shorter contention windows (briefer dips, not shallower)
```

**MPS is the on-device S1↔S2 dial (`--s2-mps-pct`).** Start the daemon as the *same user* that runs the
services (server `-force-tegra`); higher cap = more SMs to S2 = faster S2 but slower S1. **Measured Pareto
(Q4 default, continuous S2)** — you cannot get S1≥25 AND S2≥1 on one shared GPU; pick your point:

| `--s2-mps-pct` | S1 avg | **S2** | S2/infer | note |
|---|---|---|---|---|
| 50 | **18.1 Hz** | 0.62 Hz | 1447 ms | favour S1 (Q4≈f16 here: SM-throttled) |
| 70 | 15.2 Hz | 0.92 Hz | 977 ms | almost meets S2 |
| **80 (DEFAULT)** | 12.9 Hz | **1.10 Hz** | 846 ms | **first to meet S2≥1 Hz**; S1 fine for control |
| 100 (uncapped) | 7.5 Hz | 1.31 Hz | 693 ms | max S2, S1 craters |
| 30 | — | — | — | crashes (Tegra MPS fragile) |

Q4 only helps at cap≥70 (S2 becomes bandwidth-bound there; at cap50 it's SM-throttled so Q4≈f16). Throttling
cap100 to period 1 s gives 12.4 Hz / 0.99 Hz — no better than cap80, so just pick the cap.

```bash
./run.sh async                    # DEFAULT: Q4 + cap80 -> S2 1.1 Hz (meets spec) / S1 ~13 Hz
./run.sh async --s2-mps-pct 50    # favour S1 (~18 Hz) at S2 0.62 Hz
./run.sh async --s2-mps-pct 100   # max S2 (1.31 Hz) but S1 ~7.5 Hz
```

> Under MPS the C++ runner used to crash during model load (`ggml_backend_cuda_device_get_memory` →
> `cudaMemGetInfo`, fragile on Tegra MPS). Fixed: the runner is launched with **`--fit off`** (we pin
> `-ngl`/`-c` so llama.cpp's auto memory-fit is unnecessary). If a service ever crashes/hangs on startup,
> it now fails fast with a clear message — recover with **`./run.sh async-clean`** (frees ports 5601-5610
> and kills any leftover service/runner), then re-run. If MPS itself acts up, `--s2-mps-pct 0` runs without it.

**The default is now CONTINUOUS S2** (`--s2-period-s 0`): S2 re-fires the instant it finishes, so plans
are always fresh (~0.65 Hz S2, no idle gap) but S1 sits in the contention dip the whole time. Throttling
S2 with `--s2-period-s N` inserts idle windows where S1 runs full-speed — trading plan freshness for S1
rate. The S1 average is set by the S2 duty cycle (measured here, 12–20 s runs):

| scenario | S2 duty | **avg S1** | floor (S2 running) | idle (S2 off) |
|---|---|---|---|---|
| **continuous (default) + cap50** | ~95 % | **~16 Hz** | 18.1 Hz | — (no idle windows) |
| period 2 s + cap50 | ~65 % | 18.6 Hz | 18.1 Hz | 29.2 Hz |
| period 3 s + cap50 | ~36 % | **21.1 Hz** | 18.1 Hz | 29.3 Hz |
| period 3 s, no MPS  | ~29 % | 19.4 Hz | **12.7 Hz** | 30.3 Hz |

cap50 holds the dip **floor** at a steady 18.1 Hz (vs 12.7 Hz unmitigated, +43 %). With the *continuous*
default S1 lives at that floor (~16 Hz incl. bootstrap). To raise the S1 *average*, throttle S2
(`--s2-period-s 3` → ~21 Hz, the bigger lever) — or, to get fresh S2 **and** fast S1 at once, take S2 off
the GPU entirely (remote `--s2-host`, below). MPS doesn't help while S1 is idle (it even costs ~1 ms/tick).

**The only way to remove the dip entirely is to not share the GPU — run S2 on another machine:**
```bash
#   on the x86 box:  python -m quantization_kit.async_zmq.s2_service --host 0.0.0.0
#   on Thor:         python -m quantization_kit.async_zmq.s1_service
#                    python -m quantization_kit.async_zmq.orchestrator --s2-host <x86-ip>
```

## What the control trace shows

```
  tick   Hz  |  S1 act  ms    |  S2
     0   0.0 | --    ·   -     | ★ FIRE #0 (bg)          <- S2 fired, loop does NOT wait
     1  31.2 | --    ·   -     | running… (1t)           <- S1 loop keeps ticking while S2 computes
    ...
    37  28.1 | S1    → 58.0    | ✓ DONE #0 latent  (overlapped 37 S1 ticks)   <- first plan arrives
    45  23.0 | S1    → 82.5    | ★ FIRE #1 (bg)          <- re-fire; S1 keeps steering off latest latent
    59  13.0 | S1    → 53.4    | ✓ DONE #1 latent  (overlapped 14 S1 ticks)
```
- `S1 act ms` rises (~54 → ~85 ms) while S2 is *running* and falls back when S2 is idle: the two
  share one GPU, but S1 **keeps producing actions throughout** — it is never blocked.
- The old design's S1 rate *during* an S2 inference was **0 Hz** (it slept); here it stays ~11-17 Hz.

## Measured on this Jetson (f16-latent S2, TRT S1, 5 steps)

| | value | note |
|---|---|---|
| S1 uncontended (optimized) | **~33 ms (~30 Hz)** | TRT denoise + TRT memory_encoder/QFormer + GPU prep |
| S1 uncontended (baseline) | ~58 ms (~17 Hz) | TRT denoise only; eager conditioning + CPU PIL prep |
| S1 while S2 running | ~57-85 ms | single-GPU time-slicing with the 7B llama.cpp decode |
| S2 (kv-reuse, **default Q4**) | ~1.45 s (~0.66 Hz) **under contention** | native q4@448=609 ms (1.64 Hz), f16=747 ms (1.34 Hz) — but cap50+S1 throttle BOTH to ~1.45 s (contention-bound: Q4's −23% and 336's −11% are masked here). Real fix = remote S2 (`--s2-host`) → native 1.6-2 Hz. `--no-s2-q4` for f16. |
| S2↔S1 overlap | ~14-37 S1 ticks per S2 inference | **the control loop never blocks on S2** |

S1 cost breakdown (TRT FP16, 5 steps) and the optimizations that closed the gap to 30 Hz —
all **action-preserving** vs the fp32 reference (`verify_s1_opt.py`: 12/12 identical actions):

| component | ms | optimization |
|---|---|---|
| denoise loop (5 × 5.25 ms) | ~26 | the real bottleneck (lower `--s1-steps` / `num_sample_trajs` to cut further) |
| prep_images (PIL resize ×2) | ~7→~1 | **#4** GPU resize (`prep_images_gpu`, default on; `--no-gpu-prep` to revert) |
| memory_encoder + QFormer | ~5.4→~2 | **#3** TRT FP16 engines (default on; `attach_trt_denoise(trt_cond=…)`) |
| DINOv2 rgb (both frames) | ~1.3 | already TRT; a bit-exact goal-cache exists (`--rgb-cache`) but saves only ~0.3 ms |

> The split's job is decoupling (S1 keeps running while S2 thinks); the 30 Hz S1 rate itself comes
> from the TRT/GPU pipeline above. `quantization_kit/verify_s1_opt.py` reports each step's fidelity
> and speed; `verify_rgb_cache.py` proves the goal-feature cache is bit-exact.

## Files

- `zmq_proto.py` — ports + multipart (msgpack header + raw numpy) (de)serialization
- `s2_service.py` — S2 brain (reuses `InternVLAN1LlamaCppPolicy(load_s1=False, runner_endpoint=…)`)
- `s1_service.py` — S1 only (lite diffusion + `attach_trt_denoise`)
- `orchestrator.py` — the non-blocking control loop (frame source is swappable for a live camera/env)
- `run_async_demo.py` — launches all three, waits for ready, runs the orchestrator, tears down

## Code changes that enable this

- **C++**: `llama.cpp/tools/mtmd/s2-runner.cpp` + `CMakeLists.txt` — added a `--zmq tcp://*:PORT`
  REP server mode (purely additive; the stdin/stdout path is unchanged; `find_library(zmq)` so
  builds without libzmq still work). Rebuild: `cmake --build llama.cpp/build --target llama-s2-runner -j`.
- **Python**: `internnav/.../internvla_n1_policy_llamacpp.py` — added `load_s1=False` (S2-only
  process) and `runner_endpoint=…` (drive the runner over ZMQ instead of a stdin pipe). Defaults
  preserve the original pipe + in-process-S1 behaviour, so `dual` / `replay` / eval are unaffected.
