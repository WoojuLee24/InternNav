---
name: project_quantization_kit
description: quantization_kit/ — S2 standalone inference + 데이터셋 생성 (VLN-PE/VLN-CE 분리), two-step look-down iteration 구현 현황
metadata:
  type: project
---

# quantization_kit — S2 양자화 벤치마크 키트

## 목적
InternVLA-N1 S2(System 2, Qwen2.5-VL 기반 플래너)를 양자화 실험에서 bf16 기준선과 1:1 비교할 수 있도록 독립 실행 가능한 inference + 데이터셋 생성 키트.

**Why:** 양자화 팀이 모델만 교체(S2Inferencer._load_model 오버라이드)하고 나머지 파이프라인을 동일하게 유지할 수 있어야 결과 비교가 공정함.

## 설치 (빈 Docker 환경)

```bash
# 1. torch (CUDA 버전에 맞게)
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8
# 2. kit 의존성 (transformers, accelerate, diffusers, qwen-vl-utils, pandas 등)
pip install -r quantization_kit/requirements.txt
# 3. internnav 패키지 (InternVLAN1ForCausalLM import용)
pip install -e .   # 또는: export PYTHONPATH=/path/to/InternNav
# 4. (선택) flash-attn — 없으면 --attn_implementation sdpa
pip install flash-attn==2.7.4.post1
```

## 파일 구조

| 파일 | 역할 |
|---|---|
| `s2_inference.py` | `S2Inferencer` 클래스 — 프롬프트 빌드, 이미지 전처리, two-step generate, 결과 파싱 |
| `prepare_dataset.py` | 샘플 생성 (`--dataset vln_pe` / `--dataset vln_ce`) |
| `generate_reference.py` | bf16 기준 출력 생성 (`--dataset vln_pe` / `--dataset vln_ce`) |
| `evaluate.py` | 양자화 모델 출력을 기준선과 비교 |
| `data/vln_pe/` | VLN-PE 샘플 + reference_outputs.jsonl |
| `data/vln_ce/` | VLN-CE 샘플 + reference_outputs.jsonl |

## Two-step S2 iteration (핵심)

프로덕션 evaluator(`habitat_vln_evaluator.py`)는 두 단계로 S2를 호출:
1. **Step 1**: FPV + 히스토리 → 모델이 `↓`(action 5) 출력
2. **Step 2**: `[원래user, assistant(↓), user(look-down이미지)]` 멀티턴 → 픽셀 좌표 출력

이 패턴이 없으면 `output_pixel`이 전부 null (step 1에서 `↓`만 출력하고 끝).

**트리거 조건**: `output_text`에 숫자 없음 + action 5(`↓`) 포함.

### VLN-PE vs VLN-CE look-down 처리 차이

| | VLN-PE (Isaac Sim) | VLN-CE (Habitat) |
|---|---|---|
| look-down 이미지 | **없음** — env 스텝 없이 동일 프레임 재사용 (`output['action']=-1`) | **있음** — `observation.images.rgb.125cm_30deg` 동시 캡처 |
| step 2 이미지 | `cur_pil` 그대로 재사용 | 실제 `lookdown.jpg` |

`s2_inference.py` `infer(lookdown_image=None)`: None이면 VLN-PE 동작(cur_pil 재사용), PIL/ndarray면 VLN-CE 동작.

## VLN-CE 데이터 포맷

소스: `data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r/<scene>/`
- JPEG 프레임: `videos/chunk-000/observation.images.rgb.125cm_0deg/episode_{ep:06d}_{step}.jpg`
- 파케이트: `data/chunk-000/episode_{ep:06d}.parquet` — `goal.125cm_30deg` 컬럼에 look-down view 픽셀 annotation `[y, x]` (불가시면 `[-1,-1]`)
- VLN-CE 샘플은 `goal.125cm_30deg != [-1,-1]`인 스텝만 포함 (waypoint가 look-down에서 보이는 스텝)
- `meta.json`에 `look_down: True`, `goal_pixel: [y, x]` 저장

## 출력 디렉토리 구조

```
quantization_kit/data/
  vln_pe/
    samples/sample_XXXXX/   ← current.jpg, history_*.jpg, instruction.txt, meta.json
    episodes/
    manifest.jsonl
    reference_outputs.jsonl
  vln_ce/
    samples/sample_XXXXX/   ← current.jpg, lookdown.jpg, history_*.jpg, instruction.txt, meta.json
    manifest.jsonl
    reference_outputs.jsonl
```

## 커맨드 (한 줄)

```bash
# 데이터셋 생성
python quantization_kit/prepare_dataset.py --dataset vln_pe
python quantization_kit/prepare_dataset.py --dataset vln_ce

# 기준선 출력 생성
python quantization_kit/generate_reference.py --dataset vln_pe --model_path checkpoints/InternVLA-N1-DualVLN
python quantization_kit/generate_reference.py --dataset vln_ce --model_path checkpoints/InternVLA-N1-DualVLN
```

## 주요 파라미터

- `max_new_tokens=128` — 프로덕션 evaluator hardcode 값과 동일 (1024 아님)
- `CONJUNCTIONS` 7개 — evaluator와 동일 목록, 추론 시 random.choice
- `num_history=8` — s2_step 기본값과 동일
- `resize 384×384` — policy 전처리와 동일

**How to apply:** S2 두 단계 흐름 수정 시 반드시 프로덕션 evaluator(`habitat_vln_evaluator.py`)의 look-down 분기와 정합성 확인.

관련: [[project_bev_injection]]
