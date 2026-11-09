# 노은 S2 경량화 (GGUF Q8_0) 결과 — 2026-11-04, A5000

## 용어
- HF bf16 = 원본 모델(transformers bf16). 기준선
- GGUF Q8_0 = llama.cpp 형식, decoder 8 bit. 경량화 모델
- pixel 완전 일치 = S2 답 좌표가 HF 와 글자까지 같은 비율
- replay = 저장된 에피소드 프레임을 쌓으며 4 프레임마다 S2 질의(50 회), 같은 프롬프트·history·'↓' 이어가기

## 대상 · 산출물
- ckpt: `relabel_half_noeun-v01/s1.noeun-both-b1_wait-wait_internvla-n1-system2_20261029_104936` (Qwen2.5-VL 7B 그대로)
- 빌드: `vln-deploy/quantization_kit/tools/build_qwen_s2_gguf.sh <ckpt> <out>` (isaacsim a5000 이미지, **root + --gpus** 필요)
- `vln-deploy/quantization_kit/out/gguf_noeun_v01_b1wait/` text-Q8_0-embf16.gguf 9.1 GB + mmproj-f16.gguf 1.35 GB
- 비교용 F16: `out/gguf_noeun_v01_b1wait_f16/text-f16.gguf`
- 에이전트: `internnav/agent/internvla_n1_s2_gguf_agent.py` (`internvla_n1_s2_gguf`, S2 에이전트 상속 — `_load_model`·`_generate` 만 교체)
- VLN-PE: `NOEUN_AGENT=internvla_n1_s2_gguf` 로 같은 config 에서 선택

## 정확도 (HF bf16 기준, replay 50 회, logs/noeun_smoke ep0000 + 임의 지시문)
| 비교 | pixel 완전 일치 | 10 px 이내 |
|---|---|---|
| HF sdpa (attention 커널만 다름) | 92% | 98% |
| HF fp16 (정밀도만 다름) | 80% | 90% |
| **GGUF Q8_0** | **84%** | **92%** |
| GGUF F16 | 80% | 88% |
→ GGUF 는 HF fp16 과 같은 수준. 지연 p50 (2 턴 합) HF 1.66 s / Q8_0 1.38 s.

## 맞추면서 찾은 차이 2 개 (둘 다 고침)
1. **이미지 리사이즈**: llama.cpp 자체 리사이즈는 HF 와 픽셀이 다르다. HF 와 같은 smart_resize + PIL BICUBIC 으로 미리
   리사이즈해서 넘긴다 (384 -> 392, 640x480 -> 644x476). 24% -> 47%
2. **repetition penalty 꺼짐**: 이 llama.cpp(389ff61) 에서 `--repeat-last-n -1` 은 0 = penalty 끔이 된다
   (common.cpp 의 -1 -> n_ctx 변환이 주석, llama-sampler.cpp `max(last_n, 0)`). n_ctx 를 직접 준다. 47% -> 84%
   ⚠️ 같은 llama.cpp 를 쓰는 기존 DualVLN 스택도 -1 을 넘기면 같은 문제가 있다 (확인 안 함).
- vision encoder 출력은 HF 와 같다 (토큰 cosine 중앙값 1.0000) — `scripts/sim/verify/verify_vision_noeun.py`

## 99% 가능성 점검 (같은 이미지·같은 프롬프트, HF bf16 기준)
| 비교 | 완전 일치 | 3 px 이내 | 10 px 이내 |
|---|---|---|---|
| HF bf16 재실행 | 100% | 100% | 100% |
| HF sdpa | 92% | 98% | 98% |
| HF fp16 | 80% | 88% | 90% |
| GGUF Q8_0 (기본) | 84% | 88% | 92% |
| GGUF Q8_0 KV f32 | 84% | 88% | 92% |
| GGUF Q8_0 flash-attn off | 84% | 90% | 96% |
| GGUF BF16 (가중치 HF 와 동일) | 80% | 86% | 88% |
- 답 종류(pixel/행동/STOP/WAIT)는 모든 비교에서 100% 일치. 어긋남은 전부 pixel 숫자.
- GGUF 가 어긋난 8 질의 중 7 개는 HF 를 fp16 으로만 바꿔도 어긋난다 → 모델이 경계에 있는 입력 (커널 누적 순서 차이로 뒤집힘).
- 가중치를 BF16 으로 해도 안 오른다 → 양자화 아님. 비트 단위 동일 계산이 아니면 pixel 완전 일치 99% 는 불가.
- 5090 DualVLN 도 S2 텍스트 일치는 ≈96% (command.md 3-3). 99% 는 S1 최종 명령 기준(3-4).
- `runner_args` (model_settings / `--runner_args`) 로 llama.cpp 인자 실험 가능.

## 한계
- 지시문이 장면과 안 맞는 임의 문장이다 (이 머신 데이터는 placeholder 뿐, 라벨 학습셋은 node1). 모델이 이 입력에서 민감한 편
  (HF fp16 80%, DualVLN 은 91%). 라벨 val json 이 오면 같은 replay 를 다시 잰다.

## 명령
`CUDA_VISIBLE_DEVICES=3 /workspace/isaaclab/_isaac_sim/python.sh scripts/sim/verify/verify_s2_gguf_noeun.py run --backend gguf` (hf: `--backend hf --model <ckpt>`), 비교 `... cmp --ref hf --test gguf`
