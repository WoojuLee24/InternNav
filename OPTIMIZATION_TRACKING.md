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

## Current accepted state

- Keep behavior from `c97d22d8` (sync default with mode flags scaffold).
- Rejected optimization attempts are reverted from code.
- Guardrail remains strict: if trajectory quality drops, discard that method.
