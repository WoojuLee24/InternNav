# InternVLA-N1-S2 Quantization Kit

InternVLA-N1의 **System 2 (S2)** 플래너에 대한 독립 실행형 inference + 평가 키트입니다.
quantized S2 모델이 출력 충실도를 유지하는지 검증하고 latency / GPU 메모리를 측정하는 데 사용합니다.

S2는 Qwen2.5-VL 기반 Vision-Language 모델로, 현재 RGB 프레임·과거 RGB 프레임·자연어 내비게이션 지시를 입력받아
픽셀 좌표 `(y, x)` 또는 이산 행동 시퀀스를 출력합니다.

---

## 설치

### option 1: Dockerhub
docker pull dnwn24/internnav:torch2.9.1-cuda13.0-habitat-jazzy-5090

### option 2: build Dockerfile
docker build -t [image_name] -f docker/Dockerfile-torch2.9.1-cuda13.0-habitat-jazzy-5090 .

## 실행
bash docker/run-torch2.9.1-cuda13.0-habitat-jazzy-5090

## 구성 및 데이터 저장 위치

```
quantization_kit/
├── README.md
├── requirements.txt
├── s2_inference.py            # S2Inferencer 클래스 — quant 팀이 수정할 유일한 파일 (예시)
├── generate_reference.py      # bf16 baseline 생성 (최초 1회)
├── evaluate.py                # quantized 모델의 accuracy + latency 측정
├── prepare_dataset.py         # 소스 데이터에서 data/ 디렉토리 생성
└── data/
    ├── vln-pe/
    └── vln-ce/
        ├── samples/           # (current.jpg, lookdown.jpg, history_*.jpg, ...)
        ├── manifest.jsonl
        └── reference_dualvln_outputs.jsonl
        └── reference_internvla-n1_outputs.jsonl
```
---

## 원본 데이터 소스

| 옵션 | 소스 경로 | 특징 |
|---|---|---|
| `vln-pe` | `data/InternData-N1-v0.5-mini/vln-pe/traj_data/r2r/` | Isaac Sim GT replay, `.npy` 배열, FPV만 |
| `vln-ce` | `data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r/` | Habitat, JPEG 프레임, FPV + look-down 동시 캡처 |

VLN-CE 권장: look-down 이미지가 실제로 존재해 two-step S2 iteration을 제대로 테스트할 수 있습니다.

---

## 빠른 시작 (DualVLN, VLN-CE 예시)

```bash
# 1) 샘플 데이터셋 생성 (최초 1회) or quantization_kit/data/vln-ce에 data 저장
python quantization_kit/prepare_dataset.py --dataset vln-ce

# 2) bf16 reference 출력 생성 (최초 1회)
python quantization_kit/generate_reference.py --dataset vln-ce --model_path checkpoints/InternVLA-N1-DualVLN --output quantization_kit/data/vln-ce/reference_dualvln_outputs.jsonl

# 3) quantized 모델 평가
python quantization_kit/evaluate.py --dataset vln-ce --model_path /path/to/quantized --reference quantization_kit/data/vln-ce/reference_dualvln_outputs.jsonl --output_dir results/dualvln
```

---

## 샘플 입출력 형식

### VLN-PE 샘플 (`data/vln-pe/samples/sample_00042/`)
```
current.jpg       # 현재 FPV 프레임
history_0.jpg     # np.linspace(0, t-1, 8) 과거 프레임
...
history_7.jpg
instruction.txt
meta.json         # look_down: false
```

### VLN-CE 샘플 (`data/vln-ce/samples/sample_00042/`)
```
current.jpg       # FPV (125cm_0deg)
lookdown.jpg      # look-down (125cm_30deg) — two-step step 2 에서 사용
history_0.jpg
...
history_7.jpg
instruction.txt
meta.json         # look_down: true, goal_pixel: [y, x]
```

### reference_outputs.jsonl 한 행
```json
{
  "sample_id": "sample_00042",
  "instruction": "...",
  "llm_output_text": "(132, 91)",
  "output_pixel": [91, 132],
  "output_action": null,
  "latency_s": 0.41,
  "new_token_count": 12,
  "input_token_count": 1843
}
```

이산 행동 출력 시:
```json
{"llm_output_text": "↑↑→", "output_pixel": null, "output_action": [1, 1, 3]}
```

---

## Two-step S2 iteration

S2는 두 단계로 동작합니다:

1. **Step 1**: FPV + 히스토리 → 모델이 `↓`(look-down, action 5) 출력
2. **Step 2**: `[원래 user 턴, assistant(↓), user(look-down 이미지)]` 멀티턴 → 픽셀 좌표 출력

| 데이터셋 | step 2 이미지 |
|---|---|
| VLN-PE | look-down 없음 → 현재 프레임 재사용 |
| VLN-CE | 실제 `lookdown.jpg` 사용 |

---

## quantized 모델 연결 방법

`s2_inference.py`의 `S2Inferencer._load_model`만 override합니다:

```python
from s2_inference import S2Inferencer
my_model = load_my_quantized_model(...)
infer = S2Inferencer(model=my_model, processor=my_processor, tokenizer=my_tokenizer)
```

**bitsandbytes 4-bit 예시:**
```python
from transformers import BitsAndBytesConfig
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16",
                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = InternVLAN1ForCausalLM.from_pretrained(
    "checkpoints/InternVLA-N1-DualVLN",
    quantization_config=bnb, device_map={"": "cuda:0"},
)
```

`InternVLAN1ForCausalLM`이 `Qwen2_5_VLForConditionalGeneration`을 상속하므로
Qwen2.5-VL을 지원하는 quantizer(AutoGPTQ, AutoAWQ 등)라면 모두 동작합니다.

---

## 프로덕션 설정과의 대응

| 항목 | 프로덕션 S2 eval | 이 키트 |
|---|---|---|
| 모델 입력 크기 | 384×384 | 384×384 |
| `num_history` | 8 | 8 |
| `max_new_tokens` | 128 | 128 |
| CONJUNCTIONS | 7개 random.choice | 동일 |
| two-step look-down | ✅ | ✅ |

---

## 평가 메트릭

| 메트릭 | 설명 | 권장 기준 |
|---|---|---|
| `text_exact_match_rate` | baseline과 텍스트 완전 일치 비율 | > 0.95 |
| `output_type_match_rate` | pixel/action 출력 타입 일치 비율 | > 0.99 |
| `pixel_l2_mean` / `_median` | pixel 출력의 baseline과 L2 거리 | ≤ 5 px |
| `pixel_within_10px_rate` | baseline 10 px 이내 비율 | > 0.9 |
| `action_match_rate` | action 시퀀스 완전 일치 비율 | > 0.95 |
| `latency_ms_mean` / `_p95` | generate() wall-clock (CUDA sync 포함) | quant/bf16 비율로 판단 |
| `gpu_mem_peak_mb` | torch.cuda.max_memory_allocated() | quant/bf16 비율로 판단 |

---

## 제한사항

- **S2는 depth를 사용하지 않습니다.** RGB + 텍스트만 사용합니다.
- **엔드투엔드 SR/SPL 없음.** 시뮬레이터가 필요하므로 미제공. 전체 평가: `python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_mini_5090_bev_cfg.py`
- **System 1 (NavDP)은 이 키트 범위 외.**

---

## 키트 재구성 (키트 제작자용)

```bash
# VLN-CE
python quantization_kit/prepare_dataset.py --dataset vln-ce
python quantization_kit/generate_reference.py --dataset vln-ce --model_path checkpoints/InternVLA-N1-DualVLN --attn_implementation sdpa

# VLN-PE
python quantization_kit/prepare_dataset.py --dataset vln-pe
python quantization_kit/generate_reference.py --dataset vln-pe --model_path checkpoints/InternVLA-N1-DualVLN --attn_implementation sdpa

# sanity check (bf16 vs bf16 → text_exact_match_rate ≈ 1.0)
python quantization_kit/evaluate.py --model_path checkpoints/InternVLA-N1-DualVLN --dataset vln-ce --output_dir quantization_kit/results/sanity_bf16
```
