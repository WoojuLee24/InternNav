# eval_timing.py 증분 플러시 수정 결과 (2026-08-13)

## 문제
`compare_eval.sh`의 per-call timing CSV(`VLN_TIMING`)가 생성되지 않음.
- 원인: `vln-deploy/verify/eval_timing.py`가 계측 행을 메모리에 모아 `atexit`에서 한 번에 쓰는데,
  eval 종료 경로가 `env.close()` → InternUtopia `vec_env.py:210` `simulation_app.close()` —
  Isaac Sim이 **atexit을 실행하지 않고 프로세스를 종료**시켜 전량 유실.
- 증거: eval 자식 로그에 `[vln_timing] instrumented`는 있으나 `[vln_timing] N events ->` 플러시 메시지 없음.

## 변경 (vln-deploy/verify/eval_timing.py 한 파일)
1. `_flush()`: 일회성 "w" → **증분 append** (첫 플러시만 헤더, 이후 "a"; 스냅샷 후 버퍼 비움 — S2 스레드 동시 append 안전).
2. 플러시 트리거: `agent.reset`(에피소드 경계) + `agent.step`에서 버퍼 **100행 이상**이면 플러시.
   메인 스레드만 플러시하므로 동시 쓰기 없음. hard-kill 시 유실 한도 <100행(~50 agent step).
3. atexit 유지(정상 종료 시 잔여분). CSV 스키마 불변 → `compare_report.py` 수정 불필요.

## 검증
- 단위: 첫 플러시 헤더+행 / append 중복 헤더 없음 / 빈 버퍼 no-op — PASS.
- 실전(2 에피소드, lite, `EVAL_OUTPUT_DIR`로 기존 lmdb 보존): ep1 362행 전체 + ep2 마지막 ~90행만 유실(한도 내) — PASS.
- `compare_report.read_timing` 파싱 — PASS.

## [폐기] per-call 비교 — orig vs lite, 같은 에피소드(6982_1766/1765), max_new_tokens=128
> **look_down 해상도 수정 전(↓루프 상태) 측정이라 더 이상 유효하지 않다.** 아래 표의 "s2 2.4x 빠름"은
> 인용 금지 — 수정 후 값은 이 문서 §"수정 후 전체 10-ep 재비교"(575 ms)와
> `260819_s2_standalone_latency_result.md`(단독 벤치 539 ms) 참고.

| event | orig median (n) | lite median (n) | lite 배속 |
|---|---|---|---|
| s2 | 367 ms (23) | 153 ms (32) | **2.4x 빠름** |
| s1 | 143 ms (16) | 12.9 ms (14) | **11x 빠름** (TRT) |
| s2_skip | 3.8 ms (68) | 4.0 ms (65) | 동일 |
| 에피소드 벽시계 | 41.6 s (311 steps) | 78.4 s (389 steps) | 1.9x 느림 |

**모델은 lite가 2.4~11배 빠른데 에피소드는 1.9배 느리다** — 원인 정량 확정:
1. 스텝 수 389 vs 311 (행동 차이로 더 헤맴)
2. S2 호출 수 32 vs 23
3. `sleep(0.5)` 폴링 바닥: S2가 153ms에 끝나도 벽시계는 ~0.5s. 낭비가 orig ~130ms/call,
   lite ~350ms/call → lite의 per-call 우위가 폴링에 전부 흡수됨.
   → sleep 0.5→0.05s 축소 시 lite가 가장 크게 이득.

## 연관 분석 (2026-08-12 10-episode 비교의 저속 원인)
- lite가 느렸던 건 모델이 아니라 scene 6994의 **"↓ 루프"**: 양자화 S2가 waypoint 대신 "↓"(action 5)를
  반복 출력 (ep7 S2 출력 686회 중 651회) → 매 스텝 S2 추론 → `internvla_n1_agent.py:274` `time.sleep(0.5)`
  폴링 때문에 스텝당 최소 0.5s → 1.79 fps.
- 히스토리는 `num_history=8` linspace 캡 — 에피소드 길이에 따른 호출당 비용 증가 없음 (ep7 사분위 step time 평탄).
- 후속 개선 후보: ① sleep 0.5→0.05s, ② 연속 look_down 캡, ③ S2 양자화 상향(GGUF 재생성 도구는 배포처에만).

## ↓ 루프 근본 원인: S2 입력 해상도의 코드 차이 (2026-08-13 확정)
- orig 정책: planning 이미지 384×384(`resize_w/h`), **look_down 프레임은 원본 해상도**(`internvla_n1_policy.py:127-128`).
- lite 정책(수정 전): 전부 448×448 고정 — "112-aligned parity" 주석은 실측으로 반증(runner mtmd가 내부 smart-resize; 384·640×480 모두 정상 인코딩+LATENT).
- 전이 통계: orig은 [5]→pixel 76/76(100%), lite(수정 전)는 [5]→[5]가 1005/1094(92%), **전 에피소드에서** 발생.
- 수정: ① `internvla_n1_policy_llamacpp.py`에 `lookdown_full_res` 옵션(기본 False=기존 동작) + s2_step guard clause,
  ② `internvla_n1_agent_llamacpp.py` 전달 1줄, ③ lite config `resize_w/h: 384` + `lookdown_full_res: True`.
- 수정 후 스모크(2 ep): **[5]→[5] 0/9, 100% pixel**, fps 8.39/7.43 (orig 스모크 8.19/6.95와 동급). 코드 차이가 주원인으로 확정.
- 잔여 확인: 6994 scene 포함 전체 10 에피소드 재비교 필요 (compare_eval.sh가 lmdb 정리 포함).

## 주의: lmdb 오염 (2026-08-13 스모크들)
- `runner.py:435`가 `EVAL_OUTPUT_DIR`을 무조건 `<model-path>/logs_stop`으로 덮어씀 → 격리 시도 무효.
- 스모크 에피소드가 본 lmdb에 append됨: lite2 Count 10→16, orig 10→12. 기존 10-ep 에피소드 데이터 자체는 보존,
  집계본은 `verify/compare_out/history.jsonl`·`docs/COMPARISON.md`에 안전. `compare_eval.sh` 재실행 시 lmdb가 정리되며 해소.
- runner.py 경유 실행은 출력 격리 불가 — 격리하려면 eval.py 직접 실행 필요.

## 수정 후 전체 10-ep 재비교 결과 (2026-08-13T05:30:28, lmdb 클린 상태)
`compare_eval.sh -n 10 --lite-model .../InternVLA-N1-lite2-DualVLN` 재실행. 리포트는 `vln-deploy/docs/COMPARISON.md` §5, 이력은 `verify/compare_out/history.jsonl`에 자동 반영됨.

- **↓ 루프: 6994 포함 10 에피소드 전체에서 [5]→[5] 0건** (총 [5] 출력 81회 전부 정상적으로 pixel→S1로 이어짐). 해상도 수정으로 완전히 해소 확정.
- 정확도: SR orig 0.60 / lite 0.70, SPL 0.571/0.630, NE 2.27/1.86 — lite가 근소 우위지만 n=10 노이즈 범위. **핵심은 어제의 lite 열세(SR -0.10)가 사라졌다는 것.**
- 속도: orig 6.12 steps/s vs lite 6.52 steps/s — **lite가 근소 우위로 역전** (어제는 lite 2.84 vs orig 6.39로 절반 이하였음).
- per-call(VLN_TIMING, p50): S1 orig 142.6ms → lite 13.0ms (**11x**, TRT). **S2는 orig 315.9ms → lite 575.0ms로 lite가 0.55x, 더 느림.**
  전체 우위는 S1 가속 + 루프 제거에서 나오는 것이지 S2 자체의 속도 이득은 없음.

## S2가 lite에서 더 느린 이유: llama.cpp/mtmd의 배치=1 하드 제약 (업스트림 한계, 2026-08-13 확인)
- 실측(`verify/compare_out/lite_run.log`): 한 S2 호출에 이미지 N장(히스토리+현재, 최대 9장)이 있으면 **한 장씩 순차** encode+decode
  (`encoding image slice... in ~15-22ms` × N회 반복). orig(HF Qwen2-VL)은 전 이미지를 한 배치 forward로 처리.
- 근본 원인: `vln-deploy/quantization_kit/llama.cpp/tools/mtmd/clip.cpp:829`
  `GGML_ASSERT(imgs.entries.size() == 1 && "n_batch > 1 is not supported")`, `clip_image_batch_encode`(3288행)도
  `batch_size != 1`이면 `false` 반환 — 원저자 TODO 주석: "batch>1 미구현, cgraph 커지는 문제로 진짜 배치 지원 불필요 판단".
- **InternNav/vln-deploy 코드의 문제가 아니라 third_party `llama.cpp` submodule 자체의 제약** — 고치려면 llama.cpp 패치 또는
  upstream에서 이 TODO 구현 여부 확인 후 업그레이드 필요. 정책 코드 수정만으로는 해결 불가.
- 부차 요인: Q4_K_M은 prefill(=긴 멀티모달 프롬프트 최초 처리, S2 호출의 대부분)에서 dequant 오버헤드로 오히려 손해 —
  양자화의 메모리 대역폭 이점은 decode 단계에서만 발휘됨.
- 시사점: GGUF 양자화의 실질 이득은 이 조건(5090, 128 토큰)에서 **지연시간 단축이 아니라** Jetson급 저전력 기기에서의
  메모리/상시 로드 가능성 — 원래 배포 타겟(Jetson Thor)에서의 목적과 일치.

## A/B 실증: look_down 해상도 → pixel-goal 성공률 (2026-08-13, verify/verify_lookdown_resolution.py)
"해상도를 다시 고정하면 ↓ 루프가 재발하는가"를 실측으로 확인. 같은 실제 lookdown 프레임 30장
(`quantization_kit/data/vln-ce/samples/sample_0000{0..29}`, 336개 전부 `meta.json`에 `look_down: true`
기록된 실제 스텝)을 같은 GGUF S2·같은 프롬프트로, **해상도만** 바꿔 넣었다 — 알고리즘/제어로직은
불변(orig·lite가 공유하는 `internvla_n1_agent.py`는 손대지 않음; 변수는 순수 이미지 데이터뿐).

| 조건 | pixel 좌표 출력 |
|---|---|
| 원본 카메라 해상도 (640×480, `lookdown_full_res=True`) | **30/30 (100%)** |
| 384×384 고정 (`lookdown_full_res=False`) | **12/30 (40%)** |

같은 모델·같은 로직에서 해상도만으로 성공률이 60%p 떨어짐 — "↓ 루프가 로직 버그가 아니라 입력
정보량 부족으로 모델이 반복 실패한 것"이라는 인과를 직접 증명. 재현 스크립트:
`quantization_kit/convenv_x86/bin/python verify/verify_lookdown_resolution.py --n 30`
(`InternVLAN1LlamaCppPolicy.s2_step`/`step_no_infer`를 그대로 호출 — 프롬프트 조립은 항상 프로덕션 코드와 동일).

## 재실행 커맨드 (per-call 표 포함 전체 비교)
`cd /ws/src/InternNav && ./vln-deploy/verify/compare_eval.sh -n 10 --lite-model /ws/src/InternNav/checkpoints/InternVLA-N1-lite2-DualVLN`
