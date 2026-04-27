# Async Optimization Tracking (S1 / S2 / Joint)

Last updated: 2026-04-24
Branch: `true_async_background_thread`

## Goal

This report tracks async optimization quality and speed using a dual-system view:

- **S2 (Language Planner)**: VLM inference that outputs coordinates or discrete actions
- **S1 (Trajectory Generator)**: trajectory generation path when S2 emits coordinates
- **Joint (S1 + S2)**: end-to-end request/response performance seen by the client

The main objective is to improve speed without collapsing trajectory quality.

---

## Metric Definitions

### System 2 (S2)

- `s2_req_hz`: `s2_runs / elapsed_seconds`
- `s2_latency_ms`: `s2_time_total / s2_runs * 1000`

Interpretation:
- Higher `s2_req_hz` is better (planner processes more frames per second)
- Lower `s2_latency_ms` is better (faster planner step)

### System 1 (S1)

- `s1_req_hz`: `s1_runs / elapsed_seconds`
- `s1_latency_ms`: currently `null` (not separable in current architecture)

Interpretation:
- Higher `s1_req_hz` is better (more trajectory generations per second)
- `s1_latency_ms` requires internal instrumentation inside `agent.step()` to isolate true S1-only time

### Joint (S1 + S2)

- `joint_req_hz`: `total_requests / elapsed_seconds`
- `joint_latency_ms`: `total_action_time / total_requests * 1000`

Interpretation:
- Higher `joint_req_hz` means better end-to-end throughput
- Lower `joint_latency_ms` means better user-visible responsiveness

### Quality

- `trajectory_ratio`: `trajectories / (trajectories + discrete_actions) * 100`
- `discrete_ratio`: `discrete_actions / (trajectories + discrete_actions) * 100`
- `no_traj_count`: client-side count of "No trajectory in response"

---

## Baseline Control Variables (SYNC = baseline)

- Model: `checkpoints/InternVLA-N1-w-NavDP`
- Device: `cuda:0`
- Server script: `scripts/realworld/http_internvla_server_debug.py`
- Client script: `scripts/realworld/http_internvla_client_debug.py`
- Calibration: `scripts/realworld/calib/calib_scout.txt`
- Rosbag playback rate: `0.5`
- Same instruction template used across runs
- Container launched with `--gpus all`

### Baseline-to-Async control application matrix

| Control variable | SYNC baseline setting | ASYNC applied setting | Observed effect when applied to ASYNC |
|---|---|---|---|
| Execution mode | `/eval_dual` | `/eval_dual_async` | Enables async path; without extra safeguards quality can collapse |
| Agent state mutation | Single-threaded state updates | `ASYNC_BACKGROUND_INFERENCE = false` (no concurrent background state mutation) | Prevented async quality collapse (0% traj -> recovered trajectory-capable behavior) |
| Output arbitration | Implicit behavior | Quality-first: trajectory preferred when both trajectory and action exist | Reduced bias toward discrete-only responses in async |
| `look_down` handling | Present | Mirrored in async path | Restored parity with sync behavior for coordinate-producing path |
| Metrics reset | Per reset | Explicit metric reset on policy reset | Fair per-run statistics, no carry-over contamination |

Notes:
- SYNC is the baseline reference mode for all comparisons in this file.
- The highest-impact async control was disabling unsafe concurrent mutation of shared agent state.

---

## Debug Milestone Summary

1. **Broken async phase**
   - Symptom: trajectory quality collapse (near 0%)
   - Cause: shared agent state mutated from multiple threads

2. **Stabilized async phase (current)**
   - Fix: disable background mutation path, unify sync/async output priority, fix metric key errors
   - Result: trajectory quality recovered while maintaining async endpoint behavior

---

## Sync Reference (same codebase, controlled run)

Rosbag: `my_camera_bag_20260317_073623`

| Mode | S2 req_hz | S2 latency (ms) | S1 req_hz | S1 latency (ms) | Joint req_hz | Joint latency (ms) | Trajectory ratio | Discrete ratio | Total requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sync reference | 2.49 | 313.49 | 1.58 | N/A | 2.49 | 313.49 | 63.3% | 36.7% | 463 |

### Baseline delta view (ASYNC vs SYNC baseline on same bag `073623`)

| Metric | SYNC baseline | ASYNC | Delta |
|---|---:|---:|---:|
| S2 req_hz | 2.49 | 1.58 | -36.5% |
| S2 latency (ms) | 313.49 | 309.93 | -1.1% |
| S1 req_hz | 1.58 | 0.96 | -39.2% |
| Joint req_hz | 2.49 | 1.58 | -36.5% |
| Joint latency (ms) | 313.49 | 309.95 | -1.1% |
| Trajectory ratio | 63.3% | 60.5% | -2.8 pp |
| Discrete ratio | 36.7% | 39.5% | +2.8 pp |

---

## KV-cache implementation + A/B experiment (single rosbag)

Rosbag: `my_camera_bag_20260317_073623`

### What was fixed in implementation

In `internnav/agent/internvla_n1_agent_realworld.py`, S2 generation now explicitly uses:

- `use_cache=True` when `--kv-cache` is enabled
- `use_cache=False` when `--kv-cache` is disabled

Why this matters:
- Previously, `use_cache` was only conditionally set when flag was on.
- When flag was off, generation could still follow model defaults, making A/B less controlled.
- Now KV on/off comparison is explicit and reproducible.

### A/B setup

- Mode: `async`
- Same model/device/calibration/rosbag/rate as baseline controls
- Measurement window: ~180s after server/client/rosbag startup
- Metrics source: `/async_metrics` + client log counters

### A/B results (ASYNC, same rosbag)

| Variant | S2 req_hz | S2 latency (ms) | S1 req_hz | Joint req_hz | Joint latency (ms) | Trajectory ratio | Discrete ratio | Total requests | Client traj | Client discrete | Client no_traj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KV OFF | 0.93 | 605.32 | 0.69 | 0.93 | 605.34 | 74.0% | 26.0% | 246 | 182 | 64 | 19 |
| KV ON | 1.69 | 299.55 | 0.95 | 1.69 | 299.56 | 56.2% | 43.8% | 448 | 252 | 196 | 32 |

### Delta: KV ON vs KV OFF

| Metric | KV OFF | KV ON | Delta |
|---|---:|---:|---:|
| S2 req_hz | 0.93 | 1.69 | +81.7% |
| S2 latency (ms) | 605.32 | 299.55 | -50.5% |
| S1 req_hz | 0.69 | 0.95 | +37.7% |
| Joint req_hz | 0.93 | 1.69 | +81.7% |
| Joint latency (ms) | 605.34 | 299.56 | -50.5% |
| Trajectory ratio | 74.0% | 56.2% | -17.8 pp |
| Discrete ratio | 26.0% | 43.8% | +17.8 pp |

### Delta vs SYNC baseline table (requested comparison format)

| Metric | SYNC baseline | ASYNC (KV ON) | Delta |
|---|---:|---:|---:|
| S2 req_hz | 2.49 | 1.69 | -32.1% |
| S2 latency (ms) | 313.49 | 299.55 | -4.4% |
| S1 req_hz | 1.58 | 0.95 | -39.9% |
| Joint req_hz | 2.49 | 1.69 | -32.1% |
| Joint latency (ms) | 313.49 | 299.56 | -4.4% |
| Trajectory ratio | 63.3% | 56.2% | -7.1 pp |
| Discrete ratio | 36.7% | 43.8% | +7.1 pp |

Interpretation:
- KV ON improved speed metrics substantially vs KV OFF in this async setup.
- But it shifted policy behavior toward more discrete responses (lower trajectory ratio).
- Relative to SYNC baseline, ASYNC+KV ON still trails throughput and S1 rate, while improving end-to-end latency slightly.

---

## Temperature + KV-cache A/B Experiment

### Control Variables (Fixed across all runs)

| Parameter | Value | Description |
|-----------|-------|-------------|
| mode | async | Async execution mode |
| device | cuda:0 | GPU device |
| model_path | checkpoints/InternVLA-N1-w-NavDP | Model checkpoint |
| resize_w | 256 | Image width |
| resize_h | 256 | Image height |
| num_history | 1 | Number of history frames |
| plan_step_gap | 12 | Planning step interval (frames) |
| max_new_tokens | 80 | Max language tokens |
| rate | 0.5 | Rosbag playback rate |
| jpeg-quality | 95 | JPEG compression |
| depth-png-compress | 6 | PNG compression level |

### Independent Variables (A/B variants)

| Variant | KV-cache | Temperature | Repetition Penalty |
|---------|---------|--------------|-------------------|
| Run 1 (baseline) | ON | 1.0 | 1.0 |
| Run 2 | ON | 0.8 | 1.0 |
| Run 3 | ON | 0.8 | 1.1 |

### Experiment Commands

Run from shell (./exec.sh entered):

```bash
# Run 1: KV ON + temp=1.0 (baseline)
NO_ATTACH=1 \
KV_CACHE=1 \
TEMPERATURE=1.0 \
REPO_DIR=/workspace/InternNav \
RATE=0.5 \
./scripts/realworld/async_check_tmux.sh rosbag/my_camera_bag_20260317_073623 kv_on_temp10

# Run 2: KV ON + temp=0.8
NO_ATTACH=1 \
KV_CACHE=1 \
TEMPERATURE=0.8 \
REPO_DIR=/workspace/InternNav \
RATE=0.5 \
./scripts/realworld/async_check_tmux.sh rosbag/my_camera_bag_20260317_073623 kv_on_temp08

# Run 3: KV ON + temp=0.8 + rep_pen=1.1
NO_ATTACH=1 \
KV_CACHE=1 \
TEMPERATURE=0.8 \
REPETITION_PENALTY=1.1 \
REPO_DIR=/workspace/InternNav \
RATE=0.5 \
./scripts/realworld/async_check_tmux.sh rosbag/my_camera_bag_20260317_073623 kv_on_temp08_rep11
```

After each run (~240s), get metrics:
```bash
curl -s http://localhost:5802/async_metrics
```

Parse client output:
```bash
# From client log in test_data/async_check_*/
grep -c "Received Trajectory"
grep -c "Received Discrete"
grep -c "No trajectory"
```

### Results: fill below

| Variant | S2 req_hz | S2 latency (ms) | S1 req_hz | Joint req_hz | Joint latency (ms) | Trajectory ratio | Discrete ratio | Total requests | Client traj | Client discrete | Client no_traj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KV ON + temp=1.0 | 1.69 | 299.55 | 0.95 | 1.69 | 299.56 | 56.2% | 43.8% | 448 | 252 | 196 | 32 |
| KV ON + temp=0.8 | - | - | - | - | - | - | - | - | - | - | RUN ERROR* |
| KV ON + temp=0.8 + rep_pen=1.1 | | | | | | | | | | | | |

*RUN ERROR: Container ROS not connected to rosbag topics. Run from terminal for actual experiments.

### Delta vs Previous Baseline

| Metric | KV ON + temp=1.0 | KV ON + temp=0.8 | Delta | KV ON + temp=0.8 + rep_pen=1.1 | Delta |
|--------|-----------------|------------------|-------|-------------------------------|-------|
| S2 req_hz | 1.69 | | | | |
| S2 latency (ms) | 299.55 | | | | |
| S1 req_hz | 0.95 | | | | |
| Joint req_hz | 1.69 | | | | |
| Joint latency (ms) | 299.56 | | | | |
| Trajectory ratio | 56.2% | | | | |
| Discrete ratio | 43.8% | | | | |

### Per-rosbag results

| Rosbag | S2 req_hz | S2 latency (ms) | S1 req_hz | S1 latency (ms) | Joint req_hz | Joint latency (ms) | Trajectory ratio | Discrete ratio | Total requests | Client traj | Client discrete | Client no_traj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `my_camera_bag_20260317_061841` | 1.57 | 299.00 | 1.02 | N/A | 1.57 | 299.01 | 65.4% | 34.6% | 367 | 241 | 127 | 23 |
| `my_camera_bag_20260317_063047` | 1.74 | 280.50 | 0.65 | N/A | 1.74 | 280.51 | 37.3% | 62.7% | 407 | 152 | 255 | 39 |
| `my_camera_bag_20260317_073623` | 1.58 | 309.93 | 0.96 | N/A | 1.58 | 309.95 | 60.5% | 39.5% | 370 | 224 | 146 | 24 |

### Aggregate statistics (async, 3-bag)

- `joint_req_hz`: mean **1.63 Hz** (min 1.57, max 1.74, stdev 0.08)
- `joint_latency_ms`: mean **296.49 ms** (min 280.51, max 309.95, stdev 12.15)
- `s1_req_hz`: mean **0.88 Hz** (min 0.65, max 1.02)
- `trajectory_ratio`: mean **54.4%**
- Weighted trajectory ratio by event count: **53.89%**
- Weighted no-traj ratio (client): **6.99%**

---

## What this says in dual-system terms

- **S2 speed**: planner is the dominant latency contributor, running around 1.57-1.74 Hz in stabilized async runs.
- **S1 speed**: trajectory path is active and healthy (0.65-1.02 Hz across bags), not collapsed.
- **Joint speed**: end-to-end throughput is 1.57-1.74 Hz with ~281-310 ms latency.
- **Quality**: async quality is scene-dependent, but recovered from collapse and remains trajectory-capable.

---

## Risks and caveats

- `s1_latency_ms` is intentionally not reported as a numeric value yet, because S1 timing is not isolated from `agent.step()` internals.
- Bag-to-bag variation is significant (scene geometry and turning demand change trajectory/discrete balance).
- Enabling true concurrent background mutation without state isolation can degrade quality.

---

## Next optimization steps

1. Add internal instrumentation in agent/model path to isolate true S1 latency.
2. Implement worker-state isolation for safe true decoupled async (separate state channel or model worker).
3. Re-run same 3-bag protocol and compare:
   - `delta joint_req_hz`
   - `delta joint_latency_ms`
   - `delta trajectory_ratio`
4. Add plots (trend per bag and boxplots for S1/S2/joint metrics).
