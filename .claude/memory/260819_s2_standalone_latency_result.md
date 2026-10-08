# S2 단독 per-call latency A/B (Isaac Sim 없이) — 2026-08-19

## 목적
문제 2("lite S2가 느린 건 llama.cpp가 이미지를 한 장씩 순차 인코딩하기 때문")을 **eval 없이** 확인.
기존 근거는 `compare_eval.sh`의 VLN_TIMING(p50 orig 315.9 / lite 575.0 ms)뿐이었는데, 이 값은
에피소드 길이·↓루프·`sleep(0.5)` 폴링이 섞인 실행에서 나온 것이라 "모델 자체가 느린가"를
단독으로 증명하지 못했다.

## 방법 — 신규 `vln-deploy/verify/verify_s2_latency.py`
- 같은 실제 vln-ce 샘플(`quantization_kit/data/vln-ce/samples`)에 `s2_step()`만 호출해 시간 측정.
  제어 로직(`internvla_n1_agent.py`) 개입 없음.
- 파라미터는 두 eval config의 `model_settings`를 그대로 읽어 parity 보장 (resize 384, num_history 8,
  max_new_tokens 128 — orig `s2_step`은 policy:183에서 128 하드코딩).
- **핵심: 히스토리 장수 스윕.** 순차 인코딩 가설이 맞으면 lite만 장수에 비례해 늘어야 한다.
- Isaac 인터프리터 필수 — orig이 `flash_attention_2`를 쓰는데 `convenv_x86` venv에는 flash_attn이 없다
  (그래서 `verify_dual_accuracy.py`가 "번들 안에서 orig S2 비교 불가"라고 적어둔 것).
- 변형은 따로 돌린다(같은 프로세스에 7B 16 GB + GGUF 러너 7 GB 동시 상주 시 OOM). CSV는 누적 병합되어
  따로 돌려도 표가 합쳐진다.

```
cd /ws/src/InternNav && /workspace/isaaclab/_isaac_sim/python.sh vln-deploy/verify/verify_s2_latency.py --n 11 --variant lite --history 0,2,4,8 --warmup 1
cd /ws/src/InternNav && /workspace/isaaclab/_isaac_sim/python.sh vln-deploy/verify/verify_s2_latency.py --n 11 --variant orig --history 0,2,4,8 --warmup 1
```

## 결과 (조건당 10콜 median, 5090, GPU 단독 점유)

| 이미지 장수 | orig (HF bf16) | lite (GGUF Q4) | lite/orig |
|---|---|---|---|
| 1 | 78.5 ms | 84.1 ms | 1.07x |
| 3 | 119.1 ms | 201.7 ms | 1.69x |
| 5 | 170.3 ms | 310.1 ms | 1.82x |
| 9 | 293.0 ms | 539.1 ms | 1.84x |

최소자승 기울기: **orig 27.2 ms/장 (절편 43.0) vs lite 56.7 ms/장 (절편 28.6)**

- **1장이면 두 모델이 동급**(1.07x). 즉 Q4 양자화 자체는 손해가 아니다.
- 격차는 전부 **장당 비용 2.1배**에서 온다 — `clip.cpp:829 GGML_ASSERT(entries.size()==1)`의
  순차 인코딩 예측과 정확히 일치. 로그에도 `encoding image slice... in 14 ms`가 장수만큼 반복된다.
- 절편은 lite가 오히려 낮음(28.6 vs 43.0) → 텍스트 디코드/런타임 오버헤드는 lite가 유리.

## eval 수치와의 교차검증
eval 실전 조건은 9장(num_history 8 + current). 그 행의 **1.84x**가 VLN_TIMING p50 비율
**575.0/315.9 = 1.82x**와 일치 → 이 단독 벤치가 eval-level 관측을 Isaac 없이 재현한다.
앞으로 문제 2 관련 실험은 이 스크립트로 하면 된다 (eval 1회 ~10분 → 벤치 ~2분).

## 원인 분해 — "순차 인코딩"은 주범이 아니었다 (같은 날 추가 측정)

9장(eval 실전 조건) 기준으로 각 단계를 따로 쟀다. orig은 PyTorch에서 직접 계측
(`processor` / `model.visual` / `model.forward`), lite는 러너 stderr 로그(`image slice encoded in N ms`)와
PIL 저장 시간 실측.

| 단계 | orig | lite | 비고 |
|---|---|---|---|
| 이미지 → 파일/텐서 준비 | 34.5 ms (processor, CPU) | **200 ms** (PNG 저장 22.2 ms × 9장) | lite만 파일 왕복 |
| 비전 인코딩 | 73.7 ms (배치 ViT 1회) | 134 ms (15 ms × 9회 순차) | llama.cpp 제약 |
| PNG 디코드(러너측) | — | ~22 ms | 2.5 ms × 9 |
| LLM prefill + 생성 | ~185 ms (1764 tok) | ~183 ms (1296 tok) | **거의 동급** |
| 합계(실측) | **293.0 ms** | **539.1 ms** | |

1. **PNG 저장이 최대 항목**(200 ms). `internvla_n1_policy_llamacpp.py:161 _save()`가 매 s2_step마다
   모든 이미지를 PIL 기본 압축(compress_level=6) PNG로 디스크에 쓴다. `kv_reuse=False`라 히스토리
   8장이 매 스텝 다시 저장된다. **llama.cpp가 아니라 InternNav 코드 문제.**
2. 순차 인코딩은 실재하지만 2번째 요인 (14.9 vs 6.85 ms/장, 2.2x).
3. **Q4_K_M prefill은 거의 중립.** lite가 이미지당 토큰을 오히려 26% 적게 쓴다(144 vs 196 — mtmd가
   336으로, HF가 392로 smart-resize). LLM 구간 183 vs 185 ms로 사실상 동률.

### 검증 A/B — `_save`만 바꿔 같은 벤치 재측정 (median ms)

| 이미지 장수 | orig | lite 현재(PNG 기본) | lite + PNG `compress_level=0`(무손실) | lite + JPEG q95 |
|---|---|---|---|---|
| 1 | 78.5 | 84.1 | 71.2 | 61.5 |
| 9 | 293.0 | 539.1 | **368.5** | **345.4** |
| 장당 기울기 | 26.4 | 56.7 | 37.2 | 35.5 |
| 9장 lite/orig | — | 1.84x | **1.26x** | 1.18x |

저장 포맷별 비용(384×384): PNG 기본 21.4 ms(194 KB) / PNG `compress_level=0` 3.4 ms(433 KB) /
**BMP 0.29 ms(432 KB)** / PPM 0.23 ms / JPEG q95 0.41 ms(52 KB).

**22 ms의 정체는 디스크 I/O가 아니라 PNG zlib 압축이다.** 근거: `compress_level=0`은 파일이 2배
큰데도 6배 빠르고, BMP는 같은 크기에 0.29 ms — I/O는 0.3 ms 수준이고 나머지 21 ms가 전부 압축.
→ tmpfs/`/dev/shm`으로 옮겨도 효과 없다. **포맷만 BMP로 바꾸면 사실상 "저장 안 하는" 비용이 된다.**

| 이미지 장수 | orig | lite 현재(PNG) | **BMP** | PNG compress_level=0 | JPEG q95 |
|---|---|---|---|---|---|
| 1 | 78.5 | 84.1 | 63.2 | 71.2 | 61.5 |
| 9 | 293.0 | 539.1 | **337.7** | 368.5 | 345.4 |
| 9장 lite/orig | — | 1.84x | **1.15x** | 1.26x | 1.18x |

BMP가 1순위: 무손실(디코드 결과 동일 → 정확도 영향 0) + 러너 수정 불필요
(`mtmd_helper_bitmap_init_from_file` → `stbi_load_from_memory`가 확장자가 아니라 내용으로 포맷 판별,
stb는 BMP 지원). 진짜 in-memory 전달도 `mtmd_helper_bitmap_init_from_buf`(mtmd-helper.cpp:481)로
가능하지만 절감액이 0.29 → ~0.05 ms라 의미 없다.
추가 후보: `kv_reuse=True`면 planning에서도 히스토리 재저장·재인코딩이 사라진다(현재 기본 False).

## "양자화했는데 왜 LLM이 안 빨라지나" — prefill/decode 분리 측정

같은 9장 프롬프트를 생성 상한 1 / 128로 두 번 돌려 기울기로 분리
(`scratchpad/probe_decode.py`, 이미지 저장은 타이밍 밖):

| 구간 | orig (bf16) | lite (Q4_K_M) | |
|---|---|---|---|
| prefill (프롬프트 처리 1회) | 216.7 ms | 311.8 ms | orig 1.44x 빠름 |
| decode (출력 토큰당) | 14.03 ms (71 tok/s) | **3.96 ms (253 tok/s)** | **lite 3.5x 빠름** |

**양자화는 예상대로 잘 먹었다 — 단 decode에서만.** decode는 메모리 대역폭 바운드라 가중치가
15 GB→4.7 GB로 줄면 그대로 빨라진다(이론 3.2x, 실측 3.5x). prefill은 연산 바운드라 Q4를 다시
풀어서 계산해야 하므로 이득이 없다.

문제는 **이 워크로드가 거의 전부 prefill**이라는 것: 입력 1300~1800토큰, 출력은 `↓`나 `(431, 185)`
수준의 몇 토큰. 손익분기는 **출력 ~9.4토큰**(216.7+14.03n = 311.8+3.96n)이라 실제 출력 길이가
딱 그 근처다. 출력이 128토큰이면 orig 1998 ms vs lite 704 ms로 lite가 2.8x 빨라진다.
→ GGUF 양자화가 이 배포에서 지연시간 이득을 못 주는 건 구현 문제가 아니라 **워크로드 형태** 때문이다.

## KV 캐시 조건 — planning은 같고 look_down은 다르다

| 호출 종류 | orig | lite(`kv_reuse=False`) | 조건 |
|---|---|---|---|
| planning | `generate(past_key_values=None)` → 매번 full prefill | 매 스텝 `RESET`(chat history + KV clear, s2-runner.cpp:68) | **동일** |
| look_down | `conversation_history`에 쌓아 **전체 이미지를 다시** processor에 → ViT+prefill 전부 재계산 | RESET 없이 새 프레임 1장만 추가, KV 재사용 | **다름** |

실측 (히스토리 8장 + current로 planning 후 이어서 look_down, n=6 median):

| 호출 | orig | lite |
|---|---|---|
| planning (9장) | 344.5 ms | 535.0 ms |
| look_down (추가 1장) | **525.8 ms** | **155.5 ms** |

orig은 look_down이 planning보다 비싸고(526 > 344), lite는 3.4x 싸다. 위 장수 스윕 벤치는 planning만
쟀으므로 공정하다. 반면 **eval의 `s2` 이벤트 p50(575 vs 315.9)에는 두 종류가 섞여 있어** 조건이
다른 구간을 포함한다 — 문제 2를 논할 때는 planning 기준(1.84x → BMP 적용 시 1.15x)을 써야 한다.

## eval 검증 결과 (2026-08-20, `compare_eval.sh -n 10`)

| 지표 | orig | lite 수정 전 | **lite BMP** | lite BMP + kv_reuse=True |
|---|---|---|---|---|
| S2 p50 | 322.5 ms | 575.0 (0.55x) | **319.1 (1.01x)** | 76.6 (4.21x) |
| S1 p50 | 137.8 ms | 13.0 | 12.7 | 12.5 |
| sim loop | 5.93 steps/s | 6.52 | **7.70 (1.30x)** | 5.67 |
| SR / SPL | 0.70 / 0.666 | 0.70 / 0.630 | 0.60 / 0.513 | **0.40 / 0.291** |
| 스텝 수(10 ep) | 1,817 | — | 1,783 | **4,964** |

- **BMP 채택.** S2가 orig과 동급(1.01x)이 되고 sim loop는 orig의 1.30x. 벤치 예측(9장 337.7 ms)과
  eval 실측(319.1 ms)이 일치.
- **BMP 무손실 증명**: 같은 이미지를 PNG/BMP로 저장해 러너에 넣으면 텍스트 6/6 동일 +
  **LATENT float32 6/6 비트 동일** → 디코드된 픽셀이 완전히 같다. 정확도 영향 0.
- **`kv_reuse=True` 기각.** S2는 4.21x까지 빨라지지만 SR 0.70 → 0.40, 스텝 수 2.7배(2 에피소드가
  상한 1100까지 헤맴), sim loop는 오히려 5.67로 하락. 히스토리가 프롬프트 내 이미지 목록이 아니라
  멀티턴 KV로 들어가고 instruction이 매 턴 반복되어 **학습 프롬프트 구조와 달라진 것**이 원인.
  호출당 이득을 스텝 수 증가가 전부 먹는다. 배포 스택 `run.sh async`의 `--kv-reuse` 기본 True도
  정확도 근거 없이 정해진 값으로 보이므로 재검토 대상.
- SR 0.60 vs 0.70(1 에피소드)은 n=10 노이즈: **같은 config의 orig도 08-13 0.60 → 08-20 0.70**으로
  흔들렸다(Isaac 비결정성, `260804_gs_vlnpe_reproducibility_limits.md`). 정확도 판정은 n을 늘려야 한다.

## resize 336 vs 448 A/B (`verify_lookdown_resolution.py --n 30`)

| 조건 | pixel 출력 |
|---|---|
| planning 336 + look_down 원본 | **30/30 (100%)** |
| planning 448 + look_down 원본 | 29/30 (97%) |
| planning 336 + look_down 336 고정 | 12/30 (40%) |
| planning 448 + look_down 448 고정 | 20/30 (67%) |

planning 해상도는 336/448 차이 없음 → `resize 384`(실효 336, 144토큰) 유지. 학습(392 px/196토큰)과의
불일치가 이 지표에서는 안 나타난다. 반면 look_down은 해상도가 낮아지면 무너지므로
`lookdown_full_res=True`가 필수. 단 이 지표는 "좌표를 출력했는가"만 보고 **좌표 정확도는 안 재므로**
정밀도 동등성의 증명은 아니다.

## 폐기: `eval_timing_incremental_flush_result.md`의 "s2 2.4x 빠름"
2-에피소드 표(orig 367 / lite 153 ms)는 **look_down 해상도 수정 전** 측정이다. 당시 lite는 ↓루프에
빠져 있었고 같은 표 안에서 "에피소드는 1.9배 느림"이라는 모순이 함께 기록돼 있다. 수정 후 값
(575 ms, 위 벤치의 539 ms)이 유효하다. 해당 문서의 그 절은 폐기 표시 필요.

## 산출물
- 스크립트: `vln-deploy/verify/verify_s2_latency.py` (신규, 기존 파일 수정 없음)
- 원자료: `vln-deploy/verify/compare_out/s2_latency.csv` (variant, hist_fed, n_images, ms, sample)
- 문서 반영: `vln-deploy/docs/issue_260818.md` 문제 2
