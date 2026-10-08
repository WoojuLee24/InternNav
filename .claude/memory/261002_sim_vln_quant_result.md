# 261002 sim VLN 연동: S2 양자화(bitsandbytes) 검증 결과 (V1a/V1b)

## 무엇을 했나
- `scripts/sim/quant.py`: HF transformers + bitsandbytes로 **LLM decoder(`model.layers`)만** 양자화했다. vision, lm_head, latent query, S1(traj_dit/rgb_model/memory_encoder/navdp)은 bf16으로 둔다.
  - vln-deploy의 GGUF 경로는 생성 스크립트(`strip_to_qwen.py`, `gguf_add_latent_tokens.py`)가 없어 재현할 수 없다. bnb 경로는 `generate_latents()`를 그대로 쓰므로 latent 추출을 다시 구현할 필요가 없다.
- `scripts/sim/verify/verify_quant.py`: VLN-CE 6개 scene에서 episode 1개씩, 40 step(총 240 step, S2 86회)을 bf16/nf4/int8로 같은 순서로 replay한다. step마다 seed를 고정해 S1 noise를 통일한 뒤 bf16과 비교한다.

## 명령
`python3 scripts/sim/verify/verify_quant.py --model_path /checkpoints/InternVLA-N1-DualVLN --quants bf16,nf4,int8 --episodes 6 --max_steps 40`
- `--quants`: 첫 항목은 반드시 bf16(reference)이다.
- `--episodes` / `--max_steps`: scene 수와 episode당 step 수
- 결과: `scripts/sim/verify/out/quant/InternVLA-N1-DualVLN/report.json`, `*_outputs.npy`

## 결과 (RTX 5090, InternVLA-N1-DualVLN)
| | bf16 | nf4 | int8 |
|---|---|---|---|
| peak VRAM (torch) | 16.7 GiB | **7.7 GiB** | 11.0 GiB |
| S2 step p50 / p95 | 451 / 1591 ms | 459 / 1654 ms | 495 / 1792 ms |
| S1 step p50 | 85 ms | 82 ms | 82 ms |
| S2 출력 일치(종류+action, 86회) | — | 96.5% | **98.8%** |
| S2 텍스트 완전 일치 | — | 73.3% | 88.4% |
| pixel L2 중앙값 / p90 (384 공간, 34회) | — | 4.0 / 56.7 px | **0.0** / 32.2 px |
| latent cosine 중앙값 / 최소 | — | 0.931 / 0.755 | **0.991** / 0.896 |
| S1 waypoint ADE / FDE 중앙값 | — | 0.072 / 0.112 m | 0.025 / 0.042 m |

- 합격 기준 판정(action ≥95%, latent cos ≥0.98, pixel 중앙값 ≤10px, ADE ≤0.15m):
  - **int8은 모두 통과한다.**
  - nf4는 latent cos 0.93으로 미달이고 나머지는 통과한다.
- bnb 0.48.1이 sm_120(5090)과 cu130에서 정상 동작한다(V1a 통과).
- 속도 이득은 없다. 5090에서는 bf16이 오히려 가장 빠르다. 양자화의 의미는 VRAM 절감뿐이다.

## 다음
- V1c: sim과 gdm을 동시에 띄운 상태에서 bf16으로 총 VRAM을 측정한다. 32GB 안에 들어가면 bf16을 쓰고, 부족하면 int8, 그래도 부족하면 nf4를 쓴다.
- V1d: closed-loop SR을 비교한다(V13 시나리오에서 bf16 vs int8).
- w-NavDP 체크포인트에 대해서도 같은 검증을 수행한다.
- 버그 수정: realworld agent는 S1을 no_grad 없이 호출해 `traj_to_actions(.numpy())`가 실패한다. `SimVLNAgent.step`을 `torch.inference_mode()`로 감쌌다(원본은 수정하지 않음).

---
# 추가: pixel goal 좌표 규약 (V5)과 배선 테스트 (V9/V10/V12)

## pixel goal 좌표는 **서버로 보낸 입력 프레임(640x480) 좌표 [row, col]**이다. 384 공간이 아니다.
- 근거 1: bf16 replay의 S2 pixel 34개는 row 208~455, col 43~491 범위다. 384를 넘는 값이 row 15개, col 9개 있어 384 공간으로는 불가능하다.
- 근거 2: VLN-CE 라벨 `goal.125cm_30deg`가 640x480 프레임 기준이다(`quantization_kit/prepare_dataset.py:112`).
- 근거 3: fake_sim 배선 테스트에서 S1 끝점 투영과의 Δpx 중앙값이 입력 공간 30px, 384 공간 295px였다. 다만 fake 영상은 카메라 기하가 달라 참고용이다.
- 조치: `vln_bridge_node.py`의 `pixel_space` 기본값을 `input`으로 했다(`s2` 옵션은 유지).
- 주의: S2는 대부분 `↓`(look-down) 다음에 pixel을 낸다. 학습 라벨은 30° look-down 영상 기준이다. sim 카메라는 14.8° 고정이고, 서버는 realworld와 같이 같은 프레임으로 재질의한다. 투영은 보낸 프레임 기준이라 기하적으로는 일관되지만 모델 입력에는 domain gap이 있다.

## 배선 테스트 (`scripts/sim/verify/plumbing_test.sh`, fake_sim + server + bridge, ROS_DOMAIN_ID=77 격리)
`scripts/sim/verify/plumbing_test.sh /checkpoints/InternVLA-N1-DualVLN bf16 90`
- 인자: 체크포인트, quant, 모드당 실행 초
- Mode A: GDGGuidancePoint VIA_GP가 `/gdg/data/gp`로 발행되고 fake_sim이 받은 상대 좌표가 일치했다(0.91, -0.69). `/gdq/msg/cmd_vel` publisher는 1개다.
- Mode B: `/vln/trajectory`(map, 로봇 앞 0.3m부터)가 나오고, `/gdq/msg/cmd_vel` publisher는 bridge 1개이며 `/gdg/data/gp` publisher는 0개다. discrete 회전([2,2] 등)은 PID 경로로 처리됐다.
- V8 (fake): stamp 시점 TF lookup 성공률 95~96%
- 수정한 버그: 종료 시 context가 무효화된 뒤 publish해서 생기던 traceback

---
# 261005 추가: 시스템별 검증 (S2 / S1 / dual), GT 정확도

## vln-deploy는 양자화를 어디까지 검증했나 (`vln-deploy/docs/COMPARISON.md`)
- S1은 양자화가 아니라 원본 텐서를 추출한 것이다. TRT 최적화의 action 100% 일치와 속도는 실측했다.
- S2 Q4_K_M:
  - 크기(14.2 → 4.46 GiB)와 decode 속도(15.6 → 40.8 tok/s)만 측정했다.
  - 정확도(action 일치 97.9%, latent cos 약 0.96)는 **원본 트리 값을 인용**했을 뿐 이 번들에서는 측정하지 않았다(§3-4).
  - **원본 bf16 safetensors 대비 정확도는 측정하지 않았다**(§3-5).
  - 런타임 GPU 메모리도 측정하지 않았다.
- sim: Isaac 10 ep SR 0.70 → 0.60. n이 작아 정확도를 판단할 수 없다고 자체 기록돼 있다.

## 이번 측정 (`verify_quant.py`, VLN-CE 6 ep / 240 step, look-down은 실제 30° 프레임)
`python3 scripts/sim/verify/verify_quant.py --model_path /checkpoints/InternVLA-N1-DualVLN --quants bf16,nf4,int8 --episodes 6 --max_steps 40`
- 결과: `scripts/sim/verify/out/quant/InternVLA-N1-DualVLN_lookdown30/report.json`

| 시스템 | 지표 | bf16 | nf4 | int8 |
|---|---|---|---|---|
| S2 | weight | 15.4 GiB | **6.3 GiB** | 9.4 GiB |
| S2 | activation peak | 0.96 GiB | 1.00 GiB | 1.30 GiB |
| S2 | latency p50 / p95 (호출당) | 454 / 1207 ms | 451 / 1130 ms | 463 / 1235 ms |
| S2 | vs bf16: 출력 일치 / 텍스트 / pixel L2 p90 / latent cos | — | 96.5% / 81% / 7.5 px / 0.949 | **98.8% / 93% / 7.8 px / 0.994** |
| S2 | vs GT: pixel 출력률 / L2 중앙값 / hit@30px | 90.9% / 10.0 px / 83% | 90.9% / 10.4 px / 83% | 90.9% / 10.0 px / 80% |
| S1 | weight / activation peak | 0.21 / 0.04 GiB | 0.17 / 0.04 GiB | 0.17 / 0.04 GiB |
| S1 | latency p50 / p95 | 79 / 83 ms | 77 / 82 ms | 76 / 81 ms |
| S1 | vs bf16 ADE / FDE (latent 차이로 인한 차이만) | — | 0.010 / 0.013 m | 0.003 / 0.004 m |
| dual | peak / step p50 | 16.7 GiB / 108 ms | 7.7 GiB / 107 ms | 11.0 GiB / 105 ms |

- **GT 라벨 규약 버그 발견**: VLN-CE `goal.125cm_30deg`는 [x, y]로 저장돼 있다(첫 값이 500까지 나와 행일 수 없다). `quantization_kit/prepare_dataset.py`의 "[y,x]" 주석은 틀렸다. agent의 pixel 출력은 [row, col]이다.
- GT 정확도는 세 가지 모두 거의 같다(n=33으로 작음). 양자화의 정확도 손실은 bf16 대비 비교로만 드러난다(nf4 latent cos 0.949).
- sim(live) VRAM: sim+gdm+bf16 = 23.5 GB / 32.6 GB → **이 구성(lpp-sim)에서는 bf16으로 충분**하다. int8은 travmap 등 추가 GPU 모듈이 붙을 때의 대안이다.
- sim(closed-loop) 비교(V1d)는 baseline 성공률이 낮고 편차가 커서 보류했다. `261005_sim_vln_live_result.md` 참고.
