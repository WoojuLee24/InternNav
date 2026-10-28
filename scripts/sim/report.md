# scripts/sim — 측정 결과

InternVLA-N1을 경량화하고 Isaac Sim RBQ10·gdm에 연동해 잰 결과다.
- **절 번호는 [command.md](command.md)와 같다.** 이 문서의 §5 결과는 command.md §5를 실행해서 얻었다.
- 측정 환경: RTX 5090, sim 실행 중. 측정 날짜는 절마다 적는다.

| 절 | 내용 | 명령 (command.md) |
|---|---|---|
| [0. 결론](#0-결론) | 기본 구성과 선택 이유 | — |
| [5. dataset 검증 결과](#5-dataset-검증-결과) | S1 단독, S2 단독, S2+S1 전체, 분리 구조 | [§5](command.md#5-dataset-검증) |
| [6. sim 주행 결과](#6-sim-주행-결과) | 구성별 SR, Mode A/B | [§6](command.md#6-sim-주행-검증) |
| [7. 연동 기능 검증 결과](#7-연동-기능-검증-결과) | 토픽, 좌표계, 투영, PID 등 결과 기록 | [§7](command.md#7-연동-기능-검증) |

## 용어

| 용어 | 뜻 |
|---|---|
| S2 | 지시문과 이미지를 보고 판단하는 큰 모델(Qwen2.5-VL). 약 1 Hz. pixel goal 또는 discrete action, 그리고 S1에 줄 latent를 낸다 |
| S1 | S2 latent와 현재 영상으로 경로를 그리는 작은 diffusion 모델. 수십 Hz |
| HF bf16 / PyTorch bf16 | 원본 모델을 그대로 실행한 것. **기준선** |
| GGUF Q8_0 / FP16 / Q4 | llama.cpp로 S2를 실행하는 형식. Q8_0 = decoder 8 bit, Q4 = 4 bit |
| TRT FP16 | S1을 TensorRT 16 bit 엔진으로 실행한 것 |
| HF parity | GGUF S2가 HF와 같은 입력·생성 방식을 쓰도록 맞춘 옵션(`S2_HF_PARITY=1`). 맞춘 항목은 [§5-3](#5-3-s2s1-전체) |
| 캐시 512 | S2 vision 임베딩 캐시 512개(`S2_EMBD_CACHE=512`). 출력은 같고 더 빠르다 |
| async / sync | S1과 S2를 따로 돌리는지(async), 한 순서로 돌리는지(sync) |
| Mode A | S2 pixel goal → gdm planner가 길을 찾는다 (`lpp-sim-travmap`) |
| Mode B | S1 경로 → MPC가 그대로 따라간다 (`lpp-sim-vln`, planner 없음) |
| SR | 성공률. 성공 = 모델이 STOP을 내고 목표 반경 안(c1·c2 1.0 m, c3~c5 1.5 m) |
| c1~c5 | ETRI 실내 장애물 없는 구간 5개: c1 프린터, c2 마커, c3 교차로, c4 프린터 지나 교차로, c5 중간 출발 |
| ADE / FDE | 두 경로의 평균 / 마지막 점 거리 차 |
| 로봇 명령 일치 | 슬롯마다 S2 판단(경로 또는 discrete action)이 같고, 경로라면 첫 discrete action까지 같은 비율 |
| p50 / p95 | 지연의 중앙값 / 95번째 백분위 |

---

## 0. 결론

- **대상 모델**: 공개 checkpoint `InternRobotics/InternVLA-N1-DualVLN`(HF commit `a698a9e`, shard 4개 sha256 일치)
- **기본 구성**: S2 GGUF Q8_0(HF parity, 캐시 512) + S1 TRT FP16 10 step, S1/S2 프로세스 분리 + ZMQ + `dual_server` async(HTTP :5801)
- **기준선**: 원본 HF bf16 S2 + PyTorch bf16 S1
- **평가 설정**: S2 입력 384 → 392(look-down 640×480 → 644×476), history 8, 생성 토큰 128, S1 10 step. 공식 `habitat_dual_system_cfg.py`와 같다.

| | 기준선 | 경량 기본 구성 | 결과 |
|---|---|---|---|
| S2 | HF bf16 | GGUF Q8_0 | **가속 + 경량화.** sim S2 2056 → 486 ms(4.2배). dataset 텍스트 일치 96%, latent cos 0.9988, 정답 근접률 HF와 같거나 높음(0.967) ([§5-2](#5-2-s2-단독)) |
| S1 | PyTorch bf16 10 step | TRT FP16 10 step | **가속.** 단독 93.7 → 19.6 ms(4.8배), sim 루프 192 → 75 ms. 경로 차 1 mm. 같은 16 bit라 메모리 경량화는 아니다 ([§5-1](#5-1-s1-단독)) |
| S2+S1 dataset | HF + PyTorch | GGUF Q8_0 + TRT | **로봇 명령 일치 99.2%**, 경로 차 5 cm 이내 100% ([§5-3](#5-3-s2s1-전체)) |
| S2+S1 sim | HF + PyTorch 5/10 | GGUF Q8_0 + TRT 9/20 (10-27 재측정 12/20) | SR은 같은 수준이다. c1·c2는 항상 실패, c3·c4는 항상 성공, 차이는 c5에서만 난다 ([§6-1](#6-1-구성별-sr-mode-b)) |
| 구조 | 단일 서버(동기) | 분리 + async | 응답 p95 2.0 → 0.8 s, 처리량 +38%. sync 분리는 결과가 100% 같다 ([§5-4](#5-4-s1s2-분리-구조)) |

- **S2 속도 이득의 출처**: 대부분 llama.cpp 실행 방식에서 온다(FP16 GGUF도 544 ms). HF는 look-down과 latent 단계마다 이미지 10장을 다시 인코딩하고 약 2천 토큰을 다시 처리하지만, llama.cpp는 KV cache를 이어 쓴다.
- **Q8_0을 고른 이유**: FP16과 정확도가 같으면서 더 빠르고(458 대 544 ms) 6 GB 작다.
- **Q4를 쓰지 않는 이유**: 메모리는 가장 적지만 로봇 명령 일치가 95.4%로 99%에 못 미친다.

---

## 5. dataset 검증 결과

VLN-CE r2r val 녹화 6 ep × 40 step. 측정 2026-10-05~06.

### 5-1. S1 단독
명령: [command.md §5-1](command.md#5-1-s1-단독). sim SR 칸은 [command.md §6-1](command.md#6-1-구성별-sr-mode-b)의 `run_trt_sr.sh` 결과다.

| S1 | dataset 경로 차 중앙값 (ADE/FDE) | 첫 action 일치 | 단독 latency | sim SR (S2 HF) | sim 루프 S1 p50 |
|---|---|---|---|---|---|
| PyTorch bf16 10 step (기준) | 0 | 100% | 93.7 ms | 5/10 | 192 ms |
| **TRT FP16 10 step** | 1 / 2 mm | 100% | **19.6 ms** | **5/10** | **90 ms** |
| TRT FP16 5 step | 13 / 20 mm | 98% | 10.9 ms | 4/10 | 76 ms |

- 재현 엔진: denoise 가중치 319개가 원본 TRT 엔진과 bit 단위로 같다. 원본 TRT 대비 waypoint 차는 1 mm 이하이고 action은 100% 같다.
- TRT 오차는 bf16↔fp32 수준이다. 5 step의 오차는 step 수 때문이고 TRT 때문이 아니다(PyTorch 5 step도 13 mm).

### 5-2. S2 단독
명령: [command.md §5-2](command.md#5-2-s2-단독). S2 48회, 기준 HF bf16, GGUF는 HF parity + 캐시 512.

| 모델 | S2 p50 | 행동(종류+action) 일치 | 텍스트 일치 | pixel 완전 일치 | latent cos 중앙값 / 최소 | S1 경로 차 | 정답 근접률(30 px) |
|---|---|---|---|---|---|---|---|
| HF bf16 (기준) | 1808 ms | 100% | 100% | 100% | 1.000 / 1.000 | 0 | 0.933 |
| 참고: HF fp16 | 1961 ms | 100% | 94% | 91% | 0.9994 / 0.985 | 1.1 mm | 0.967 |
| **GGUF Q8_0** | **458 ms** | 100% | **96%** | **94%** | 0.9988 / 0.950 | 0.9 mm | 0.967 |
| GGUF FP16 | 544 ms | 100% | 94% | 91% | 0.9993 / 0.929 | 1.0 mm | 0.967 |
| GGUF Q4 | 465 ms | 100% | 81% | 73% | 0.977 / 0.781 | 4.3 mm | 0.967 |

- S1 경로 차는 각 latent를 같은 HF S1 bf16에 넣었을 때의 ADE 중앙값이다.
- 캐시 on/off 출력은 bit 단위로 같다(text/pixel 48/48, latent 차 0.0). 캐시를 켜면 S2가 약 100 ms 빨라진다.
- GGUF FP16·Q8_0은 HF를 fp16으로 바꿨을 때와 같은 수준으로 HF bf16과 일치한다.

### 5-3. S2+S1 전체
명령: [command.md §5-3](command.md#5-3-s2s1-전체). S1 호출 슬롯 240개, 기준 HF bf16 + PyTorch bf16. 모든 조합의 S1은 case마다 같은 초기 noise를 쓴다. 목표는 로봇 명령 일치 99% 이상이다.

| 조합 | 로봇 명령 일치 | 경로 ADE < 2 cm | 경로 ADE < 5 cm | ADE 중앙값 / p90 |
|---|---|---|---|---|
| HF + TRT | 99.6% | 100% | 100% | 0.11 / 0.33 cm |
| 참고: HF fp16 + PyTorch | 98.8% | 98.3% | 100% | 0.10 / 0.70 cm |
| **GGUF Q8_0 + TRT** | **99.2%** | **99.2%** | **100%** | 0.16 / 0.80 cm |
| GGUF Q8_0 + PyTorch | 98.8% | 97.9% | 100% | 0.10 / 0.72 cm |
| GGUF FP16 + TRT | 99.2% | 96.7% | 99.2% | 0.14 / 0.75 cm |
| GGUF FP16 + PyTorch | 98.3% | 96.7% | 98.8% | 0.11 / 0.73 cm |
| GGUF Q4 + TRT | 95.4% | 84.2% | 93.3% | 0.64 / 4.9 cm |
| GGUF Q4 + PyTorch | 94.6% | 85.0% | 92.5% | 0.64 / 5.3 cm |
| 기준선: HF + PyTorch, 초기 noise만 변경 | 90.8% | 51.2% | 78.3% | 3.46 / 9.5 cm |

- **FP16·Q8_0**: 남은 약 1% 불일치는 HF fp16(98.8%)과 같은 수준이다. 15° 회전 경계 근처에서 0.5 cm 차이가 첫 action을 바꾼 경우다.
- **Q4**: 4 bit 양자화 오차(latent cos 0.977) 때문에 99%에 못 미친다.
- **S1 TRT의 영향**: 0.4%p 이하다.
- 이 표의 S1 지연은 직접 비교하지 않는다. TRT 쪽은 CPU 전처리를 포함한다. 단독 지연은 [§5-1](#5-1-s1-단독)을 본다.

**HF와 같게 맞춘 항목** (`--hf-parity`, 모두 opt-in. 기본 경로는 그대로)
1. **해상도**: HF smart_resize(28 px 정렬)와 같게 한다. 정면 392×392, look-down 644×476이다. 가장자리 attention window를 처리한다(`MTMD_QWEN25VL_HF_ALIGN=1`).
2. **latent 추출**: HF `generate_latents`처럼 `<|im_end|>` 뒤에 latent query를 붙인다(`S2_LATENT_WITH_EOS=1`).
3. **chat 형식** (`S2_HF_CHAT=1`): HF chat template의 기본 system 메시지를 넣는다. look-down 턴 앞의 답변 `<|im_end|>`를 KV에 유지한다.
4. **프롬프트 공백** (hfparity `_format_prompt`): HF policy의 `split_and_clean()`을 적용한다. `<image>` 앞뒤 공백과 history 사이 줄바꿈을 지운다.
5. **repetition penalty**: HF처럼 1.05를 입력 전체에 적용한다. runner의 `--repeat-last-n -1`은 이 llama.cpp 빌드에서 context 크기로 바뀌지 않아 penalty가 꺼진다(`common/common.cpp:1211` 주석 처리). 그래서 `repeat_last_n=8192`를 넘기고, 프롬프트 토큰도 penalty 기록에 넣는다.

**영향이 없음을 확인한 것**
- **vision encoder**: 같은 입력에서 GGUF f16과 HF fp32의 cos가 0.99999(rel L2 0.7%)다. HF bf16 vs fp32(2.8%)보다 정확하다.
- **GEMM 누적, flash attention, KV 정밀도**: fp32 누적, FA off, KV f32로 바꿔도 출력이 같았다.

### 5-4. S1/S2 분리 구조
명령: [command.md §5-4](command.md#5-4-s1s2-분리-구조). HF backend.

| 항목 | 결과 |
|---|---|
| 분리 sync vs 단일 서버 (dataset 240 요청) | action, S2 텍스트, pixel 100% 일치. S1 차 1e-7 m |
| MPS 70/30 | 텍스트 91.9% 일치. bit 단위 재현이 깨진다 |
| async 지연 | p95 2022 → 809 ms, 처리량 1.77 → 2.44 응답/s |
| sim 직진 40 s (`run_live_speed.sh`) | 로봇 속도는 같다(RTF 약 0.6이 결정). 경로 갱신 공백 최대 2.4 → 0.75 s |

---

## 6. sim 주행 결과

### 6-1. 구성별 SR (Mode B)
명령: [command.md §6-1](command.md#6-1-구성별-sr-mode-b). async. 측정 2026-10-06.

| S2 + S1 | SR | c1 / c2 / c3 / c4 / c5 | S2 p50 / p95 | S2 FPS | S1 p50 / p95 | GPU 메모리 최대 |
|---|---|---|---|---|---|---|
| **(기준선) HF bf16 + PyTorch bf16** | 5/10 | 0/2, 0/2, 2/2, 2/2, 1/2 | 2056 / 2143 ms | 0.53 | 192 / 218 ms | 29.4 GB* |
| HF bf16 + TRT | 5/10, 6/10 | 0/2, 0/2, 2/2, 2/2, 1/2 · 2/2 | 1796~1813 / 1844~1865 ms | 0.60 | 90~92 / 116 ms | 26.2 GB |
| **GGUF Q8_0 + TRT** | 9/20 | 0/4, 0/4, 4/4, 4/4, 1/4 | **486 / 521 ms** | **2.16** | 75 / 106 ms | 18.5 GB |
| GGUF FP16 + TRT | 10/20 | 0/4, 0/4, 4/4, 4/4, 2/4 | 561 / 582 ms | 1.88 | 75 / 108 ms | 24.0 GB |
| GGUF FP16 + PyTorch | 8/20 | 0/4, 0/4, 4/4, 4/4, 0/4 | 658 / 758 ms | 1.59 | 145 / 203 ms | 23.6 GB |
| GGUF Q4 + TRT | 10/20 | 0/4, 0/4, 4/4, 4/4, 2/4 | 496 / 549 ms | 2.08 | 79 / 114 ms | 17.2 GB |

- **횟수**: 기준선과 HF + TRT는 5 ep × 2회, GGUF 구성은 5 ep × 4회다.
- **SR**: 모든 구성에서 c1·c2는 항상 실패하고 c3·c4는 항상 성공한다. 차이는 c5에서만 나며, 횟수가 적어 SR로는 구분되지 않는다.
- **속도**: 기준선 대비 Q8_0 + TRT는 S2가 4.2배, S1이 2.6배 빠르다. S2가 빨라지면 GPU 경합이 줄어 S1도 빨라진다.
- **PyTorch S1**: S1 지연이 2배(75 → 145 ms)이고, GPU 경합으로 S2도 느려진다.
- **GPU 메모리**: sim(약 4.2 GB)과 S1을 포함한 GPU 전체 최대값이다. 측정 시점마다 sim 상태가 달라 ±2 GB 정도 차이가 난다.
- \* 기준선 GPU 메모리는 10-06에 따로 측정했다. 프로세스별로는 **S2 HF bf16 18.2 GB + S1 PyTorch 0.9 GB**이고, 이때 sim은 8.0 GB였다(다른 측정 때는 약 4.2 GB). 그래서 전체값을 다른 행과 그대로 비교하면 안 된다.
- **2026-10-27 재측정** (Q8_0 + TRT, 5 ep × 4회): **12/20** (c1 0/4, c2 0/4, c3 4/4, c4 4/4, c5 4/4). 달라진 것은 c5뿐이라(1/4 → 4/4) "나빠지지 않았다"로 읽는다. 자세한 내용은 [vln-deploy/docs/Report.html](../../vln-deploy/docs/Report.html)에 있다.

### 6-2. Mode A와 Mode B
명령: [command.md §6-2](command.md#6-2-mode-a--mode-b-직접-실행). S2 GGUF Q8_0 + S1 TRT, 같은 날 같은 서버로 측정했다. 측정 2026-10-27, 기록 [.claude/memory/261027_sim_vln_travmap_remeasure_result.md](../../.claude/memory/261027_sim_vln_travmap_remeasure_result.md).

| mode | 회차 | c1 | c2 | c3 | c4 | c5 | SR |
|---|---|---|---|---|---|---|---|
| A | 1 | ✗ 6.1 | ✗ 5.8 | ✅ 0.6 | ✅ 0.5 | ✗ 2.0 | 2/5 |
| A | 2 | ✗ 3.7 | ✗ 5.7 | ✅ 0.2 | ✗ 2.0 | ✗ 1.6 | 1/5 |
| B | 1 | ✗ 2.8 | ✗ 5.8 | ✅ 1.2 | ✅ 1.0 | ✅ 0.8 | 3/5 |
| B | 2 | ✗ 2.7 | ✗ 4.9 | ✅ 0.8 | ✅ 1.2 | ✅ 0.8 | 3/5 |

- 숫자 = STOP 때 목표까지 거리(m)
- 합계 A 3/10, B 6/10. 20회 모두 모델 STOP으로 끝났다(시간 초과 0).
- c1·c2는 두 모드 모두 실패한다. 모델 판단 문제다.
- A는 c5에서 교차로를 지나서 멈춘다. planner가 앞쪽 GP까지 가서 STOP 반영이 늦는 것으로 추정한다(확인 안 함).

---

## 7. 연동 기능 검증 결과

명령: [command.md §7](command.md#7-연동-기능-검증). 결과는 `.claude/memory/`의 기록에 있다.

| 날짜 | 기록 | 내용 |
|---|---|---|
| 10-02 | [261002_sim_vln_quant_result.md](../../.claude/memory/261002_sim_vln_quant_result.md) | S2 bitsandbytes 양자화(bf16/nf4/int8) VRAM·지연·출력 일치 (`verify_quant.py`) |
| 10-05 | [261005_sim_vln_live_result.md](../../.claude/memory/261005_sim_vln_live_result.md) | 실제 sim+gdm에서 V0~V8: 토픽·TF, VRAM, 카메라 crop/K, depth vs lidar, PID, 주행. bridge 기본 파라미터의 근거 |
| 10-05 | [261005_lookdown_models_result.md](../../.claude/memory/261005_lookdown_models_result.md) | 30° look-down 카메라, 선명 구간 에피소드(c1~c5), 모델 3개 비교, 시각화 |
| 10-07 | [261007_bridge_odom_frame_result.md](../../.claude/memory/261007_bridge_odom_frame_result.md) | bridge 좌표계를 odom 기준으로 변경. map≠odom에서 goal 오차 167.6 cm → 0.0 cm (`frame_test.sh`) |
| 10-27 | [261027_sim_vln_travmap_remeasure_result.md](../../.claude/memory/261027_sim_vln_travmap_remeasure_result.md) | Mode A를 `lpp-sim-travmap`으로 재측정 ([§6-2](#6-2-mode-a와-mode-b)) |
