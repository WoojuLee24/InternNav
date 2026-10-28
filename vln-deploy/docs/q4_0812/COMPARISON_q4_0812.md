# InternVLA-N1-DualVLN 원본 vs 배포판 (5090) 비교

> **outdated (08-12 Q4 경로, 사용 안 함)** — 현재 재현 절차는 [command.md](../command.md).

원본 `InternVLA-N1-DualVLN` 대비 이 배포 번들의 **크기 / 속도 / 정확도** 정리.
모든 수치는 RTX 5090 (compute cap 12.0) 환경에서 실측하거나, 원본 트리 문서(`RUN.md`)의 검증값을 인용.

---

## 1. 경량화 (크기)

**서로 다른 두 파이프라인**:

| 구성요소 | 원본 (`InternVLA-N1-DualVLN`) | 배포 버전 | 방식 | 크기 변화 |
|---|---|---|---|---|
| **S1** (제어, diffusion) | safetensors 안 S1 부분 (~0.18 GB) | `checkpoints/InternVLA-N1-lite-DualVLN/model-s1only.safetensors` (0.18 GB) | **subset 추출** — 원본 텐서 그대로 골라냄 | 같음 (원본에서 subset) |
| **S2** (추론, Qwen2.5-VL) | safetensors 안 S2 부분 (~14.2 GiB f16) | `quantization_kit/gguf/InternVLA-S2-text-Q4_K_M-latent.gguf` (4.46 GiB) | **4-bit 양자화** (GGUF Q4_K_M) | **3.3× 작음** |

**정리**:
- **S1 = 경량화 아님, 필요한 부분만 추출.** 값은 원본과 bit-exact 동일.
- **S2 = 실제 양자화 (4-bit).** 크기 68% 감소.

**우리가 재현한 `InternVLA-N1-lite-DualVLN-reproduce`는 S1만** 담겨 있음 (원본 lite와 bit-exact 일치).

---

## 2. 속도

### S1 (제어)

| 시나리오 | latency | Hz | speedup |
|---|---|---|---|
| 원본 baseline (PyTorch fp32, 10-step) | 528 ms | 1.9 | 1.0× |
| 배포 (PyTorch fp32, 5-step) | 65.5 ms* | 15.3 | 8.1× |
| **배포 (TRT FP16 sm_120, 5-step)** | **11.0 ms*** | **91** | **48×** |
| 배포 uncontended (TRT, 튜닝) | 9.1 ms | **110** | 58× |

*`verify/verify_dual_accuracy.py`로 실측 (15 sample 평균)

### S2 (추론)

| 시나리오 | decode | 크기 |
|---|---|---|
| 원본 f16 GGUF (참조) | 15.6 tok/s | 14.2 GiB |
| **배포 Q4_K_M-latent GGUF** | **40.8 tok/s** | 4.46 GiB |
| speedup | **2.6×** | -68% |

### Dual e2e (S2 + S1)

| 시나리오 | 결과 |
|---|---|
| 원본 baseline dual decision | 1257 ms/frame |
| **배포 async, 실 vln-ce 데이터** | **S1 27.4 Hz + S2 3.62 Hz** |
| S2 in-episode (KV reuse 미사용) | ~2.9 s |
| S2 in-episode (KV reuse) | ~1.5 s (2×) |
| async overlap (S1이 S2 추론 중에도 tick) | S1 41.1 Hz 유지 |

**핵심 개선의 대부분은 S1 TRT FP16 (48-58×)** — Q4 양자화(2.6×)보다 훨씬 큰 기여.

---

## 3. 추론 정확도

### 3-1. S1 최적화 정확도 (배포 5090에서 실측)

**verify_s1_opt.py** — S1 최적화 4단계가 fp32 reference와 얼마나 일치하나 (12 프레임, seed 고정):

| variant | action match | latency | Hz | speedup |
|---|---|---|---|---|
| fp32 reference (PIL prep, eager cond) | 12/12 (100%) | 83.1 ms | 12.0 | 1.00× |
| TRT baseline (PIL prep, eager cond) | **12/12 (100%)** | 38.9 ms | 25.7 | 2.14× |
| #3 TRT +cond (TRT memory_encoder+QFormer) | **12/12 (100%)** | 37.4 ms | 26.7 | 2.22× |
| **#3+#4 +gpuprep (production)** | **12/12 (100%)** | **10.1 ms** | **98.8** | **8.21×** |

**결론**: S1 최적화 4단계 모두 fp32 대비 **discrete action 100% 동일**.

### 3-2. S1 goal-frame cache 등가성 (실측)

**verify_rgb_cache.py** — DINOv2 goal-frame cache reuse가 recompute와 동일한가 (10 step):

```
cache reuse exact (cache == split, max|Δ|=0) ....... True   ← BIT-EXACT
discrete actions identical to production ........... True   ← 100% match
trajectory max|Δ| vs production (batch-1 vs batch-2). 2.32e-05  (fp GPU noise, action 무영향)
```

### 3-3. Dual e2e 정확도 (배포 5090에서 실측, 신규)

**verify/verify_dual_accuracy.py** — 배포 파이프라인(Q4 GGUF S2 + TRT S1) vs baseline(Q4 GGUF S2 + fp32 S1), 15 vln-ce 샘플, seed 고정:

| 지표 | 값 |
|---|---|
| **ACTION AGREEMENT** | **15/15 = 100%** |
| WAYPOINT max &#124;Δ&#124; | 2.34e-02 (약 2cm) |
| WAYPOINT mean &#124;Δ&#124; | 9.61e-05 |
| Speedup (TRT vs fp32) | 5.98× |
| S2 동일성 (같은 GGUF) | 100% deterministic |

**결론**: 배포 파이프라인은 baseline(같은 S2, fp32 S1) 대비 **discrete action 완전 동일**, waypoint는 fp16 노이즈 수준(2cm)만 차이. 최적화로 인한 실제 navigation 결정 손실 없음.

### 3-4. S2 Q4 양자화 정확도 (원본 트리 검증값 인용)

이 배포 번들에는 f16 GGUF가 없어서 우리 5090에서 직접 재측정 불가. 원본 트리 `RUN.md`의 값:

| 지표 | 값 | 조건 |
|---|---|---|
| S2 Q4 vs f16 action agreement | **97.9%** | 336 vln-ce 프레임 |
| S2 latent cosine similarity | ~0.96 | f16 대비 |
| S2 KV-reuse action agreement | 94% | 다중 스텝 |
| S1 5-step vs 10-step action | 97% | steps 벤치 |

측정 재현하려면 원본 개발 트리에서 `quantization_kit/bench/{validate_q4.py, bench_dual.py}` 이식 필요.

### 3-5. 원본 safetensors S2 vs 배포 GGUF Q4 (미측정)

원본 `InternVLA-N1-DualVLN` safetensors의 S2 파이프라인은 이 배포 번들에서 재구성 안 됨:
- 원본 `InternVLAN1Policy` (safetensors 로드) 실행 코드는 있음 (`internnav/model/basemodel/`)
- 하지만 원본 vs 배포 e2e 비교 스크립트는 없음
- 원본 트리에서는 Habitat SR/SPL로 통합 검증했을 것 (이 번들 밖)

### 3-6. S2 look_down 프레임 해상도 → pixel-goal 성공률 (2026-08-13, 배포 5090에서 실측)

**verify/verify_lookdown_resolution.py** — 실제 lookdown.jpg 30장(`quantization_kit/data/vln-ce/samples/sample_0000{0..29}`,
336개 전부 `look_down: true`로 기록된 실제 스텝)을 같은 GGUF S2·같은 프롬프트로, **해상도만** 바꿔 넣어 텍스트
출력에 pixel 좌표(숫자)가 나오는지 비교:

| 조건 | pixel-goal 출력 성공률 |
|---|---|
| 원본 카메라 해상도 (640×480) | **30/30 (100%)** |
| 384×384 고정 리사이즈 | **12/30 (40%)** |

**결론**: look_down 프레임을 저해상도로 넣으면 같은 모델·같은 알고리즘에서 pixel-goal 성공률이 60%p 떨어지고,
나머지는 discrete action("↓" 반복 등)으로 샌다 — 과거 `h1_internvla_n1_async_lite_cfg.py`가 448 고정
리사이즈를 쓸 때 S2가 look_down 후 pixel을 못 찍고 "↓"를 반복 출력하던 루프의 근본 원인이 정확히 이것이었다.
현재 config는 `lookdown_full_res: True`로 look_down 프레임만 원본 해상도로 전달해 원본(orig) 정책과
input format을 일치시킨다 (planning/history 이미지는 여전히 384로 리사이즈, S2 속도에 영향 없음).

---

## 4. 종합

| 항목 | 원본 | 배포 (5090 최적화) | 비고 |
|---|---|---|---|
| **S1 크기** | 0.18 GB | 0.18 GB | subset 추출, 원본 텐서 그대로 |
| **S1 속도** | 528 ms | **11.0 ms** | **48× 빠름** (TRT sm_120 FP16) |
| **S1 정확도** | 기준 | **100% (bit-exact + action match)** | 최적화 손실 없음 |
| **S2 크기** | 14.2 GiB | **4.46 GiB** | **3.3× 작음** (Q4) |
| **S2 속도** | 15.6 tok/s | **40.8 tok/s** | **2.6× 빠름** |
| **S2 정확도 (원본 트리 검증)** | 기준 | 97.9% action agree | Q4 양자화 손실 ~2% |
| **Dual pipeline 정확도 (5090 실측)** | 기준 | **100% action agree, waypoint 2cm** | S1 최적화 무손실 |
| **Dual pipeline 속도 (async)** | 1257 ms | **S1 27.4Hz + S2 3.62Hz** | overlap으로 S1 41 Hz 유지 |

**결론**:
- **S1**: 무손실 최적화 성공 (bit-exact + 100% action match, 48× 빠름)
- **S2**: Q4 양자화로 3.3× 작아지고 2.6× 빨라짐, 정확도 손실 ~2% (원본 트리 검증)
- **Dual**: 우리 실측 기준 배포 파이프라인 = baseline과 동일한 nav 결정 반환, 6× 빠름

실 사용에 문제 없는 수준의 최적화.

---

<!-- BEGIN eval-comparison: auto-generated by verify/compare_report.py -->

## 5. Isaac Sim eval — 양자화 스택 vs 원본

_2026-08-20T02:44:04+00:00 · `verify/compare_report.py`가 자동 생성한다. 이 구역은 직접 편집하지 말 것._

`h1_internvla_n1_async_lite_cfg.py` (S2 = Q4_K_M-latent GGUF, S1 = lite + TRT) 와 `h1_internvla_n1_async_cfg.py` (7B in-process) 를 **같은 에피소드**에서 비교한다. 생성 절차는 [WORKFLOW.md](WORKFLOW_q4_0812.md) Step 11–12.

### 5-1. 요약

| 항목 | orig | lite |
|---|---|---|
| 에피소드 | 10 | 10 |
| max_new_tokens | 128 | 128 |
| SR | 0.7000 | 0.6000 |
| SPL | 0.6657 | 0.5132 |
| NE | 2.1764 | 3.5008 |
| sim loop | 5.93 steps/s | 7.70 steps/s |

### 5-2. 전체 리포트

```
==========================================================================
ACCURACY   result_h1.json
==========================================================================
  orig: /ws/src/InternNav/checkpoints/InternVLA-N1-DualVLN/logs_stop/result_h1.json
  lite: /ws/src/InternNav/checkpoints/InternVLA-N1-lite2-DualVLN/logs_stop/result_h1.json

  split: val_unseen
    metric         orig       lite   delta (lite - orig)
    Count            10         10   +0.0000 same
    SR           0.7000     0.6000   -0.1000 worse
    SPL          0.6657     0.5132   -0.1525 worse
    OS           0.8000     0.6000   -0.2000 worse
    NE           2.1764     3.5008   +1.3244 worse
    TL          10.9461     9.8855   -1.0606 better
    FR           0.0000     0.0000   +0.0000 same
    StR          0.0000     0.0000   +0.0000 same

==========================================================================
SPEED
==========================================================================

  wall clock (includes Isaac Sim boot + scene load, same for both)
    variant   episodes  max_tok    wall_s  s/episode  exit
    orig            10      128       332       33.2     0
    lite            10      128       258       25.8     0
    -> lite is 1.29x the orig wall clock (faster)

  per-episode (from the run log; excludes Isaac boot + scene load)
    trajectory     orig steps   orig s  orig fps  lite steps   lite s  lite fps
    7192_1809             243    53.85      4.51         172    22.59      7.61
    7192_1808             200    36.82      5.43         184    22.26      8.27
    7192_1807             191    31.99      5.97         195    26.57      7.34
    7007_1776             153    21.07      7.26         188    25.34      7.42
    7007_1775             154    22.07      6.98         173    26.30      6.58
    7007_1774             194    33.13      5.86         144    14.98      9.61
    6994_1773             174    29.65      5.87         168    19.43      8.65
    6994_1772             166    23.95      6.93         173    21.36      8.10
    6994_1771             173    28.06      6.17         177    22.20      7.97
    6982_1767             169    25.61      6.60         209    30.38      6.88
    orig  total: 10 ep, 1817 steps / 306.2s  ->  5.93 steps/s
    lite  total: 10 ep, 1783 steps / 231.4s  ->  7.70 steps/s
    -> lite runs the sim loop at 1.298x the orig step rate
      (total duration alone is misleading: the two take different paths, so step counts differ)

  per-call latency (VLN_TIMING)  ms
    event           n  orig p50  orig p90      n  lite p50  lite p90   speedup(p50)
    agent_reset      9       0.1       0.1      9       1.1       1.1   0.14x
    agent_step    795       4.6    1003.6    766       4.6    1001.1   0.99x
    s1            146     137.8     153.0    142      12.7      14.4   10.89x
    s2            217     322.5     613.3    206     319.1     336.6   1.01x
    s2_skip       578       3.8       4.4    560       4.3       4.4   0.88x
      s2 = System-2 call, s1 = System-1 call, s2_skip = frame appended without inference,
      agent_step = whole agent.step() incl. both
```

### 5-3. 실행 이력

| 시각 | ep | tok | orig SR | lite SR | orig SPL | lite SPL | orig steps/s | lite steps/s |
|---|---|---|---|---|---|---|---|---|
| 2026-08-12T06:24:40 | 3 | 128 | 0.4565 | 0.3333 | 0.4138 | 0.3260 | 0.00 | 5.05 |
| 2026-08-12T06:29:00 | 3 | 128 | 0.4565 | 0.3333 | 0.4138 | 0.3260 | 0.00 | 5.05 |
| 2026-08-12T06:29:00 | 3 | 128 | 0.4565 | 0.3333 | 0.4138 | 0.3260 | 0.00 | 5.05 |
| 2026-08-12T11:38:36 | 10 | 128 | 0.7000 | 0.6000 | 0.6720 | 0.5271 | 6.39 | 2.84 |
| 2026-08-13T05:30:28 | 10 | 128 | 0.6000 | 0.7000 | 0.5711 | 0.6295 | 6.12 | 6.52 |
| 2026-08-19T05:31:08 | 10 | 128 | 0.8000 | 0.6000 | 0.7350 | 0.5849 | 6.40 | 6.61 |
| 2026-08-20T02:38:17 | 10 | 128 | 0.7000 | 0.4000 | 0.6657 | 0.2907 | 5.93 | 5.67 |
| 2026-08-20T02:44:04 | 10 | 128 | 0.7000 | 0.6000 | 0.6657 | 0.5132 | 5.93 | 7.70 |

<!-- END eval-comparison -->







## 부록: 정확도 측정 스크립트

이 번들에서 사용 가능한 정확도 측정:

| 스크립트 | 대상 | 실행 |
|---|---|---|
| `quantization_kit/verify_s1_opt.py` | S1 최적화 4단계 vs fp32 reference (12 프레임) | `convenv_x86/bin/python quantization_kit/verify_s1_opt.py` |
| `quantization_kit/verify_rgb_cache.py` | S1 goal-frame cache 등가성 (10 step) | `convenv_x86/bin/python quantization_kit/verify_rgb_cache.py` |
| `verify/verify_dual_accuracy.py` (신규) | **Dual e2e: 배포 파이프라인 vs baseline** (임의 N sample) | `convenv_x86/bin/python verify/verify_dual_accuracy.py --n 15` |
| `verify/verify_lifecycle.sh` | 배포 매니저 lifecycle (10단계 시나리오) | `./verify/verify_lifecycle.sh` |

**이 배포 번들에 없는** 것 (원본 트리에만):
- `quantization_kit/bench/validate_q4.py` — Q4 vs f16 336 프레임 벤치
- `quantization_kit/bench/bench_dual.py` — dual e2e latency + agreement 벤치
- `script/eval.py` — Habitat SR/SPL 평가

원본 대비 S2 GGUF Q4의 정확도를 재검증하려면 위 파일들이 필요합니다.
