## Parameter
InternVLA-N1-w-NavDP vs InternVLA-N1-DualVLN
H1 (evaluation) vs Habitat (evaluation) vs dataset (training)
FPV vs BEV vs FPV_BEV
S2 vs S1

## InternVLA-N1-w-NavDP vs InternVLA-N1-DualVLN
# 실행 시 `system1`은 **config 파일이 아니라 로드되는 모델 config**가 결정
(`get_system1_type()` → `model.config.system1`, internvla_n1_arch.py:188).
- eval: 체크포인트 `config.json`의 `system1` 값
- train: `--system1` 인자 (internvla_n1_arch.py:169)

# 비교
| 체크포인트 | `system1` | S1 depth 입력 | S1 백본 |
|---|---|---|---|
| **InternVLA-N1-w-NavDP** | `navdp_async` | **사용** | DAT_RGBD_Patch_Backbone (rgb+depth) |
| **InternVLA-N1-DualVLN** | `nextdit_async` | **무시** | nextdit rgb_model (rgb only); 필요시 DepthAnythingV2 |

## BEV 모드별 S1/S2 입력 정리 (InternVLA-N1 vs DualVLN)
# `bev_s1_mode` / `bev_s2_mode` ∈ `fpv | bev | fpv_bev` 조합에서 실제로 들어가는 텐서를 코드 기준 정리. 검증 지점:
- S2: `output_ids = self.model.generate(**inputs, ...)` — `inputs = processor(text, images=input_images)`
- S1: `dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)`

공통 전제:
- `bev_s1_mode`·`bev_s2_mode`는 독립 → 9조합 = (S2표) × (S1표).
- S2(LLM): RGB 이미지만 입력. depth 없음 (두 변형 공통).
- S1: InternVLA-N1은 NavDP 사용하며 RGB와 Depth 사용 vs DualVLN은 nextdit사용하며 RGB만 사용
- S1은 BEV 변환 시 RGB만 변환. Depth는 항상 FPV (navdp).

---

## S2 입력 (두 변형 **공통**, RGB only)

`input_images`(PIL 리스트) → processor → `inputs.pixel_values` `[Σpatches,1176]`
+ `inputs.image_grid_thw` `[N,3]`. 단일 (b,h,w,c) 아님(이미지별 패치수 상이, 크기 혼합 OK).

이미지별 RGB shape:
| 이미지 | shape (H,W,3) |
|---|---|
| FPV history/현재 | Habitat (256,256,3) / H1 (384,384,3) (`resize_w/h`) |
| look-down FPV | Habitat (480,640,3) / H1 (480,640,3) (raw 센서) |
| BEV | (224,224,3) |

`bev_s2_mode` (look-down turn만 영향, history는 항상 FPV):
| `bev_s2_mode` | look-down turn 이미지 | `<image>` |
|---|---|---|
| `fpv` | look-down FPV ×1 | 1 |
| `bev` | BEV(224) ×1 (FPV 대체) | 1 |
| `fpv_bev` | FPV + BEV(224) ×2 | 2 |

> S2는 navdp/DualVLN 동일. (DualVLN도 LLM은 같은 QwenVL — 차이는 S1에만.)

---

## S1 입력 — **InternVLA-N1 (navdp_async)**

base `[goal,cur]` 2장, FPV 224: `images_dp [1,2,224,224,3]`, `depths_dp [1,2,224,224,1]`
(metric ≤5m). **RGB·depth 둘 다 백본에 소비됨.**

| `bev_s1_mode` | `images_dp` (RGB) | `depths_dp` (사용됨) | 비고 |
|---|---|---|---|
| `fpv` | FPV `[1,2,224,224,3]` | FPV `[1,2,224,224,1]` | 기존 |
| `bev` | **BEV** `[1,2,224,224,3]` | **FPV** `[1,2,224,224,1]` (유지) | = rgb_gt 규약 (시점 혼합) |
| `fpv_bev` | `[fpv_goal, fpv_cur, bev_goal, bev_cur]` `[1,4,224,224,3]` | FPV **복제** `[1,4,224,224,1]` | fpv_concat — 전용 ckpt 필요 |

→ navdp에서는 **"depth FPV 유지"가 실제로 의미 있는 제약** (depth가 입력되니까).

---

## S1 입력 — **DualVLN (nextdit_async)**

`generate_traj`가 **`depths_dp`를 받기만 하고 안 씀**. `images_dp`(RGB)만 소비.

| `bev_s1_mode` | `images_dp` (RGB, 소비) | `depths_dp` | 비고 |
|---|---|---|---|
| `fpv` | FPV `[1,2,224,224,3]` | (무시) | 기존 |
| `bev` | **BEV** `[1,2,224,224,3]` | (무시) | **순수 RGB→BEV 교체** |
| `fpv_bev` | `[fpv_goal, fpv_cur, bev_goal, bev_cur]` `[1,4,224,224,3]` | (무시) | T 2배 → 전용 ckpt 필요 |

→ DualVLN에서는 **"depth FPV 유지"가 무의미** (depth 자체가 입력 안 됨). BEV 실험은
**RGB 스트림만의 문제**로 단순화됨. depth 관련 config(`bev_depth_scale` 등)도 무의미.

---

## 전체 9조합 (S2 = s2_mode만, S1 = s1_mode만)

S2열은 두 변형 공통. S1 depths_dp만 변형에 따라 "사용/무시" 갈림.
| # | s1_mode | s2_mode | S2(RGB) | S1 images_dp | S1 depths_dp (navdp) | S1 depths_dp (DualVLN) |
|---|---|---|---|---|---|---|
| 1 | fpv | fpv | FPV ×1 | FPV `[1,2,224,224,3]` | FPV `[1,2,224,224,1]` | 무시 |
| 2 | fpv | bev | BEV ×1 | FPV `[1,2,224,224,3]` | FPV `[1,2,224,224,1]` | 무시 |
| 3 | fpv | fpv_bev | FPV+BEV ×2 | FPV `[1,2,224,224,3]` | FPV `[1,2,224,224,1]` | 무시 |
| 4 | bev | fpv | FPV ×1 | **BEV** `[1,2,224,224,3]` | FPV `[1,2,224,224,1]` | 무시 |
| 5 | **bev** | **fpv_bev** | **FPV+BEV ×2** | **BEV** `[1,2,224,224,3]` | FPV `[1,2,224,224,1]` | 무시 |
| 6 | bev | bev | BEV ×1 | **BEV** `[1,2,224,224,3]` | FPV `[1,2,224,224,1]` | 무시 |
| 7 | fpv_bev | fpv | FPV ×1 | **FPV+BEV** `[1,4,224,224,3]` | FPV복제 `[1,4,224,224,1]` | 무시 |
| 8 | fpv_bev | bev | BEV ×1 | **FPV+BEV** `[1,4,224,224,3]` | FPV복제 `[1,4,224,224,1]` | 무시 |
| 9 | fpv_bev | fpv_bev | FPV+BEV ×2 | **FPV+BEV** `[1,4,224,224,3]` | FPV복제 `[1,4,224,224,1]` | 무시 |

- `habitat_dual_system_mini_5090_bev_cfg.py` = **#5**.
- S1 `fpv_bev`(#7·8·9)는 T 2배 → fpv_concat 학습 ckpt 아니면 입력 모양 불일치 (두 변형 공통).

---

## 경로별 차이 (Habitat / Isaac(H1) / 학습)

S1/S2 모드 의미는 같으나(같은 provider/BEVProcessor) 세부 상이:
| 항목 | Habitat | Isaac (H1) | 학습 (bev_provider) |
|---|---|---|---|
| 분기 코드 | evaluator 인라인 | policy s2_step/s1_step | `model.forward` |
| **S2 BEV 주입** | ✅ | ✅ | ❌ 미구현 (s2_mode 무시, FPV) |
| FPV resize | 256 | 384 | traj_images 224 |
| S1 layout | 2프레임 `[1,2,224,224,*]` | 2프레임 `[1,2,224,224,*]` | 전체 궤적 `[B,T,224,224,*]` |
| S1 fpv_bev depth | 복제 `[1,4,224,224,1]` | 복제 `[1,4,224,224,1]` | 복제 안 함 (img `[B,2T,…]`, depth `[B,T,…]`) |
| BEV pitch | base 0 **+60** (LOOKDOWN×2) | 고정 `bev_cam_pitch_deg=30` | 데이터 `traj_cam_pitch_2` |

핵심: ① 학습엔 S2 BEV 없음(`bev_s1_mode`만 유효) ② pitch 산출 경로마다 다름
③ eval=2프레임 vs train=전체 궤적 (fpv_bev depth 처리도 다름).

---

## DualVLN으로 실행하기 (config 수정 없이)

`system1`은 ckpt/인자로 결정되므로 **config 파일 안 건드림**:
| 명령 | DualVLN 실행법 |
|---|---|
| habitat eval | `--model_path checkpoints/InternVLA-N1-DualVLN` 추가 (자동 nextdit) |
| h1 eval | `--model_path checkpoints/InternVLA-N1-DualVLN` 추가 |
| train | `--system1 nextdit_async` (mini_single.sh는 이미 기본; System2 base 시작) |

eval.py `--model_path`는 `model_settings['model_path']`를 override (eval.py:128-130). 에이전트
(`model_name='internvla_n1'`)는 그대로, ckpt만 교체.

---

## 디버그 검증
`debug_modes:"prompt"`(eval) / 학습 launcher `--debug_modes prompt` → 조합별 실제
`input_images` 순서·크기(FPV/BEV)·`<image>` 토큰 정합을 `images/NN.jpg`+`meta.json`로 확인.
상세 [[project_bev_injection]] prompt-dump 섹션. 전체 흐름·shape: `dual_system_flow.md`.
