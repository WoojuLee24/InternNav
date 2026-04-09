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

## Module 13 attempt: promote accepted setting to default (`--num_history 1` default)

Status: accepted

Validation runs with new default path:

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix52 | `my_camera_bag_20260310_035208` | 82 | 11 | 19 | 0.649 | 1.502 | 0 |
| fix53 | `my_camera_bag_20260310_035611` | 88 | 6 | 11 | 0.628 | 1.543 | 0 |

Interpretation:
- Default now reflects accepted optimization and remains stable.

## Module 14 attempt: increase S2 step gap (`--plan_step_gap 8`)

Status: accepted

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix54 | `my_camera_bag_20260310_035208` | 113 | 2 | 3 | 0.749 | 1.279 | 0 |
| fix55 | `my_camera_bag_20260310_035611` | 112 | 3 | 6 | 0.753 | 1.285 | 0 |

Interpretation:
- Significant speed gain and improved trajectory behavior on both bags.

## Module 15 attempt: increase S2 step gap further (`--plan_step_gap 12`)

Status: accepted (best current speed-quality point)

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix56 | `my_camera_bag_20260310_035208` | 125 | 4 | 4 | 0.836 | 1.150 | 0 |
| fix57 | `my_camera_bag_20260310_035611` | 126 | 2 | 3 | 0.844 | 1.153 | 0 |

Interpretation:
- Strongest speed increase so far with excellent trajectory behavior.
- Promoted to default while still overrideable via flag.

## Module 16 attempt: aggressive step gap (`--plan_step_gap 16`)

Status: accepted as optional profile (not default)

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix58 | `my_camera_bag_20260310_035208` | 138 | 1 | 1 | 0.900 | 1.068 | 0 |
| fix59 | `my_camera_bag_20260310_035611` | 120 | 12 | 23 | 0.914 | 1.038 | 0 |

Interpretation:
- Fastest throughput/latency so far.
- Second bag shows more discrete/no-traj events than module 15, so this is better as an aggressive optional profile for further real-world validation.

## Module 17 attempt: simple vision-cache reuse (`--vision-cache`)

Status: rejected (no cache benefit observed)

Before (`--vision-cache` off, same baseline settings):

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix60 | `my_camera_bag_20260310_035208` | 128 | 1 | 3 | 0.840 | 1.146 | 0 |
| fix61 | `my_camera_bag_20260310_035611` | 131 | 0 | 0 | 0.842 | 1.145 | 0 |

After (`--vision-cache` on):

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix62 | `my_camera_bag_20260310_035208` | 126 | 3 | 5 | 0.849 | 1.133 | 0 |
| fix63 | `my_camera_bag_20260310_035611` | 119 | 5 | 12 | 0.847 | 1.141 | 0 |

Cache-hit observation:
- Server log cache hits: `0` for both runs.

Interpretation:
- No effective cache hits, so no real acceleration mechanism was activated.
- Slightly worse trajectory behavior on both bags.
- Rejected as current implementation.

## Module 18 attempt: reduce encoder input resolution (`--resize_w 256 --resize_h 256`)

Status: accepted

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix64 | `my_camera_bag_20260310_035208` | 133 | 4 | 5 | 0.874 | 1.087 | 0 |
| fix65 | `my_camera_bag_20260310_035611` | 129 | 6 | 11 | 0.890 | 1.068 | 0 |

Interpretation:
- Improved throughput/latency versus module 15 defaults while preserving strong trajectory behavior.
- Promoted to default and kept as explicit flag.

## Module 19 attempt: reduce resolution further (`--resize_w 224 --resize_h 224`)

Status: rejected (over-aggressive)

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix66 | `my_camera_bag_20260310_035208` | 109 | 20 | 52 | 1.020 | 0.921 | 0 |
| fix67 | `my_camera_bag_20260310_035611` | 122 | 12 | 32 | 0.975 | 0.967 | 0 |

Interpretation:
- Speed improved further, but quality shifted too much into discrete/no-traj behavior.
- Rejected as default; keep as optional experimental profile only.

## Module 20 attempt: combine aggressive gap + accepted resize (`--plan_step_gap 16`, `--resize 256`)

Status: rejected

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix68 | `my_camera_bag_20260310_035208` | 72 | 39 | 110 | 1.158 | 0.805 | 0 |
| fix69 | `my_camera_bag_20260310_035611` | 78 | 36 | 101 | 1.137 | 0.822 | 0 |

Interpretation:
- Very high throughput, but severe behavior degradation (large shift to discrete/no-traj).
- Rejected.

## Module 21 attempt: intermediate gap + accepted resize (`--plan_step_gap 14`, `--resize 256`)

Status: rejected

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix70 | `my_camera_bag_20260310_035208` | 69 | 40 | 110 | 1.144 | 0.824 | 0 |
| fix71 | `my_camera_bag_20260310_035611` | 67 | 40 | 112 | 1.146 | 0.822 | 0 |

Interpretation:
- Similar degradation pattern as module 20.
- Rejected.

## Module 22 attempt: KV-cache with current accepted defaults (`--kv-cache`)

Status: rejected

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix72 | `my_camera_bag_20260310_035208` | 51 | 47 | 136 | 1.193 | 0.780 | 0 |
| fix73 | `my_camera_bag_20260310_035611` | 33 | 57 | 164 | 1.258 | 0.741 | 0 |

Interpretation:
- Throughput rises strongly, but trajectory quality collapses.
- Keep flag for optional demonstrations only; rejected for real navigation baseline.

## Module 23 attempt: safe TensorRT flag path (`--tensorrt`, fallback-only)

Status: accepted as experimental flag path (no TRT engine integration yet)

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix74 | `my_camera_bag_20260310_035208` | 134 | 2 | 2 | 0.872 | 1.101 | 0 |
| fix75 | `my_camera_bag_20260310_035611` | 130 | 6 | 10 | 0.887 | 1.072 | 0 |

Runtime observation:
- TensorRT package exists in runtime, but no engine path provided in current setup.
- Safe fallback path is active (`fallback to PyTorch`) and stable.

Interpretation:
- Keep as safe experiment flag path.
- Not counted as real TensorRT acceleration yet.

## Module 24 attempt: safe quantization flag path (`--quantization --quant-method dynamic`)

Status: accepted as experimental flag path (CUDA-safe skip)

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix76 | `my_camera_bag_20260310_035208` | 90 | 31 | 76 | 1.052 | 0.893 | 0 |
| fix77 | `my_camera_bag_20260310_035611` | 128 | 11 | 17 | 0.918 | 1.033 | 0 |

Runtime observation:
- CUDA-safe guard triggers (`CUDA runtime detected; skip quantization`).
- Results are variable across bags; no actual INT8 acceleration applied yet.

Interpretation:
- Keep as safe experiment flag path.
- Not counted as real quantization acceleration yet.

## Module 25 attempt: lower S2 gap (`--plan_step_gap 10`) with accepted defaults

Status: rejected

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix78 | `my_camera_bag_20260310_035208` | 36 | 55 | 158 | 1.231 | 0.755 | 0 |
| fix79 | `my_camera_bag_20260310_035611` | 64 | 44 | 113 | 1.122 | 0.834 | 0 |

Interpretation:
- Speed increases strongly but trajectory behavior collapses.
- Rejected.

## Module 26 attempt: GPU-native bitsandbytes quantization (`--quantization --quant-method bnb_8bit`)

Status: rejected (runtime incompatibility + behavior collapse)

Attempt A (`fix80/fix81`):
- Initial run failed at startup due to `.to` not supported for 8-bit bitsandbytes models.
- Patched startup path to skip `.to()` for bnb-loaded models.

Attempt B (`fix82/fix83`):

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix82 | `my_camera_bag_20260310_035208` | 0 | 52 | 2 | 1.504 | 0.462 | 0 |
| fix83 | `my_camera_bag_20260310_035611` | 0 | 52 | 10 | 1.438 | 0.497 | 0 |

Attempt C (`fix84/fix85`) after dtype-alignment retry:

| Run | Rosbag | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |
|---|---|---:|---:|---:|---:|---:|---:|
| fix84 | `my_camera_bag_20260310_035208` | 0 | 55 | 6 | 1.564 | 0.438 | 0 |
| fix85 | `my_camera_bag_20260310_035611` | 0 | 55 | 8 | 1.568 | 0.438 | 0 |

Runtime observation:
- Repeated server exceptions in inference path: `RuntimeError('self and mat2 must have the same dtype, but got Half and Char')`.

Interpretation:
- Raw throughput appears high, but model behavior is invalid (trajectory output collapses to zero).
- Rejected for real navigation use; keep flag for future low-level kernel compatibility work only.

## Speed/quality trend snapshot (selected checkpoints)

| Stage | Representative runs | req_hz (range) | http_avg_latency_s (range) | traj_count (range) |
|---|---|---:|---:|---:|
| Early stable baseline | fix46/fix47 | 0.410–0.411 | 2.402–2.447 | 45–53 |
| History tuned | fix50/fix51 | 0.610–0.613 | 1.591–1.603 | 90–90 |
| Gap tuned (best) | fix56/fix57 | 0.836–0.844 | 1.150–1.153 | 125–126 |
| Gap tuned (aggressive) | fix58/fix59 | 0.900–0.914 | 1.038–1.068 | 120–138 |
| Resolution tuned (accepted) | fix64/fix65 | 0.874–0.890 | 1.068–1.087 | 129–133 |
| Resolution tuned (too aggressive) | fix66/fix67 | 0.975–1.020 | 0.921–0.967 | 109–122 |
| Over-aggressive combos | fix68–fix73 | 1.137–1.258 | 0.741–0.824 | 33–78 |
| Low-gap revisit (rejected) | fix78/fix79 | 1.122–1.231 | 0.755–0.834 | 36–64 |
| bnb 8-bit attempts (rejected) | fix82–fix85 | 1.438–1.568 | 0.438–0.497 | 0 |

## Current accepted state

- Keep behavior from `c97d22d8` + accepted module 5 optimization.
- Keep behavior from `c97d22d8` + accepted module 5 optimization + accepted module 8 flag scaffold.
- Accepted tuning options so far: `--num_history 2`, `--num_history 1`, `--plan_step_gap 8`, `--plan_step_gap 12`.
- Accepted optional aggressive profile: `--plan_step_gap 16`.
- Accepted resolution tuning: `--resize_w 256 --resize_h 256` (default).
- Rejected but available for optional tests: `--resize_w 224 --resize_h 224`.
- Rejected but available for optional tests: `--plan_step_gap 14/16` with resize256 combo, `--kv-cache`.
- `--vision-cache` remains available as experimental flag but current implementation is rejected.
- `--tensorrt` and `--quantization` are available and safe, but currently fallback/skip paths (no real backend acceleration integrated yet).
- GPU-native `--quant-method bnb_8bit` is currently rejected due to dtype incompatibility in this model path.
- Rejected optimization attempts are reverted from code.
- Guardrail remains strict: if trajectory quality drops, discard that method.
