# Realworld Sync Optimization Tracking

Last updated: 2026-04-08

This file records each optimization module attempt with before/after metrics.

## Metric definitions

- `traj_count`: number of `[Plan] Received Trajectory.` in client log.
- `no_traj_count`: number of `[Plan] No trajectory in response.` in client log.
- `discrete_count`: number of `[Plan] Received Discrete Actions:` in client log.
- `req_hz`: effective HTTP request throughput from client log (`idx` range over elapsed wall time).
- `http_avg_latency_s`: average of client `[HTTP] ... Latency:` values.
- `http_failures`: request failures/JSON decode/bad status.

Notes:
- These are sync-mode rosbag validation metrics, not full S1/S2 internal profiler stats.
- Trajectory count is a hard guardrail: optimizations must not significantly reduce trajectory responses.

## Stable baseline snapshot (mode scaffold only)

Reference commit: `c97d22d8` (mode scaffold)

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix18 | `my_camera_bag_20260310_035208` | 54 | 8 | 8 | 0.408 | 2.444 | 0 |
| fix19 | `my_camera_bag_20260310_035611` | 44 | 18 | 18 | 0.411 | 2.454 | 0 |

## Module 1 attempt: HTTP keep-alive session reuse (client)

Status: rejected (trajectory drop)

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix20 | `my_camera_bag_20260310_035208` | 38 | 26 | 26 | 0.420 | 2.391 | 0 |
| fix21 | `my_camera_bag_20260310_035611` | 36 | 28 | 28 | 0.415 | 2.395 | 0 |

Interpretation:
- Slight request-rate/latency improvement.
- Trajectory responses dropped too much, so module rejected.

## Module 2 attempt: stale-frame skip guard (client)

Status: rejected (over-filtering blocked planning)

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix22 | `my_camera_bag_20260310_035208` | 0 | 0 | 0 | 0.000 | 0.000 | 0 |
| fix23 | `my_camera_bag_20260310_035611` | 0 | 0 | 0 | 0.000 | 0.000 | 0 |

Interpretation:
- Guard was too strict for current timestamp alignment and prevented all inference requests.
- Reverted immediately.

## Module 3 attempt: telemetry thread in client (S1/S2/http counters)

Status: rejected (behavior affected)

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix24 | `my_camera_bag_20260310_035208` | 38 | 26 | 26 | 0.400 | 2.408 | 0 |
| fix25 | `my_camera_bag_20260310_035611` | 38 | 26 | 26 | 0.400 | 2.392 | 0 |

Additional measured telemetry (from injected metrics thread):
- fix24: `s1_control_hz ~8.77`, `s2_plan_hz ~0.43`, `http_hz ~0.40`, `http_avg_ms ~2417`
- fix25: `s1_control_hz ~8.74`, `s2_plan_hz ~0.53`, `http_hz ~0.40`, `http_avg_ms ~2388`

Interpretation:
- Even measurement-only thread changed runtime behavior and reduced trajectory responses vs baseline.
- Reverted immediately to keep sync baseline healthy.

## Module 4 attempt: remove unused `frame_data` deep-copy cache in planning loop

Status: mixed (not accepted)

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix26 | `my_camera_bag_20260310_035208` | 48 | 17 | 17 | 0.414 | 2.371 | 0 |
| fix27 | `my_camera_bag_20260310_035611` | 36 | 26 | 26 | 0.413 | 2.446 | 0 |

Interpretation:
- One bag improved, one bag degraded notably in trajectory count.
- Not robust across both rosbags, so not accepted.

## Module 5 attempt: remove duplicate JSON parsing in server route

Status: accepted

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix28 | `my_camera_bag_20260310_035208` | 50 | 13 | 13 | 0.420 | 2.403 | 0 |
| fix29 | `my_camera_bag_20260310_035611` | 46 | 18 | 18 | 0.418 | 2.416 | 0 |

Interpretation:
- Better or equal trajectory behavior on both rosbags versus baseline.
- Slight request-rate improvement with no server errors/failures.
- Kept as safe optimization.

## Module 6 attempt: add latest-request coalescing endpoint scaffold (`/eval_dual_latest`)

Status: rejected (sync metrics not consistently improved)

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix30 | `my_camera_bag_20260310_035208` | 46 | 18 | 18 | 0.414 | 2.385 | 0 |
| fix31 | `my_camera_bag_20260310_035611` | 44 | 19 | 20 | 0.414 | 2.413 | 0 |

Interpretation:
- No transport/server failures, but no robust trajectory gain.
- Added complexity without clear sync benefit, so rejected for now.

## Module 7 attempt: reduce debug log emission around per-request path

Status: rejected (bag-to-bag inconsistency)

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix32 | `my_camera_bag_20260310_035208` | 60 | 4 | 4 | 0.417 | 2.410 | 0 |
| fix33 | `my_camera_bag_20260310_035611` | 40 | 24 | 24 | 0.418 | 2.405 | 0 |

Interpretation:
- One bag improved a lot, the other degraded notably.
- Not robust across both benchmark bags, so rejected.

## Module 8 attempt: optimization-flag scaffold for future combinations

Status: accepted (scaffold only; no behavior change intended)

What was added:
- Server and client now accept these flags (currently scaffold-only):
  - `--kv-cache`
  - `--tensorrt`
  - `--quantization`
  - `--vision-cache`
  - `--method` (repeatable custom method tag)
- Client includes selected optimization flags in request JSON under `optimizations`.
- Server logs received optimization requests; execution remains sync-baseline behavior for now.

Validation runs:

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix34 | `my_camera_bag_20260310_035208` | 60 | 4 | 4 | 0.419 | 2.378 | 0 |
| fix35 | `my_camera_bag_20260310_035611` | 40 | 24 | 24 | 0.421 | 2.393 | 0 |

Interpretation:
- Scaffold is functional (flags observed in server logs) and does not introduce transport/runtime failures.
- Kept to enable controlled future A/B testing with explicit flags.

## Module 9 attempt: KV-cache + reduced max tokens (`--kv-cache --max-new-tokens 64`)

Status: rejected

Before (post-module8 baseline):

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix42 | `my_camera_bag_20260310_035208` | 62 | 2 | 2 | 0.418 | 2.403 | 0 |
| fix43 | `my_camera_bag_20260310_035611` | 54 | 10 | 10 | 0.419 | 2.404 | 0 |

After:

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix44 | `my_camera_bag_20260310_035208` | 59 | 4 | 4 | 0.412 | 2.443 | 0 |
| fix45 | `my_camera_bag_20260310_035611` | 44 | 20 | 20 | 0.419 | 2.407 | 0 |

Interpretation:
- No meaningful speed gain.
- Trajectory quality degraded on both bags (especially second bag).
- Rejected.

## Module 10 attempt: reduced max tokens only (`--max-new-tokens 64`)

Status: rejected (severe regression)

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix40 | `my_camera_bag_20260310_035208` | 20 | 2 | 2 | 0.167 | 6.706 | 0 |
| fix41 | `my_camera_bag_20260310_035611` | 20 | 7 | 7 | 0.184 | 5.631 | 0 |

Interpretation:
- Major slowdown and trajectory collapse.
- Rejected immediately.

## Module 11 attempt: reduce history length (`--num_history 2`)

Status: accepted

Before baseline:

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix46 | `my_camera_bag_20260310_035208` | 53 | 10 | 11 | 0.410 | 2.402 | 0 |
| fix47 | `my_camera_bag_20260310_035611` | 45 | 18 | 18 | 0.411 | 2.447 | 0 |

After (`--num_history 2`):

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix48 | `my_camera_bag_20260310_035208` | 86 | 3 | 3 | 0.568 | 1.718 | 0 |
| fix49 | `my_camera_bag_20260310_035611` | 80 | 7 | 10 | 0.583 | 1.677 | 0 |

Interpretation:
- Strong speed gain and strong trajectory improvement on both bags.
- Kept as accepted optimization and exposed via existing `--num_history` flag.

## Module 12 attempt: reduce history length further (`--num_history 1`)

Status: accepted (best so far)

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix50 | `my_camera_bag_20260310_035208` | 90 | 5 | 5 | 0.610 | 1.591 | 0 |
| fix51 | `my_camera_bag_20260310_035611` | 90 | 4 | 5 | 0.613 | 1.603 | 0 |

Interpretation:
- Additional speedup over module 11 and stable trajectory quality.
- Best current tradeoff among tested sync settings.
- Keep as optional flag configuration for further real-world validation.

## Current accepted state

- Keep behavior from `c97d22d8` + accepted module 5 optimization.
- Keep behavior from `c97d22d8` + accepted module 5 optimization + accepted module 8 flag scaffold.
- Accepted tuning options so far: `--num_history 2` and `--num_history 1`.
- Rejected optimization attempts are reverted from code.
- Guardrail remains strict: if trajectory quality drops, discard that method.
