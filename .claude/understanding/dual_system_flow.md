# S2 작동 흐름 (InternVLA-N1 Dual System)

## 관련 파일 및 핵심 함수

| 파일 | 함수/클래스 | 역할 |
|---|---|---|
| `internnav/agent/internvla_n1_agent.py:137` | `_start_s2_thread()` | S2 백그라운드 스레드 시작 |
| `internnav/agent/internvla_n1_agent.py:214` | `should_infer_s2()` | S2 추론 트리거 조건 판단 |
| `internnav/agent/internvla_n1_agent.py:247` | `step()` | 매 스텝 S1/S2 분기 및 action 반환 |
| `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py:110` | `s2_step()` | QwenVL 추론, pixel_goal/action 결정 |
| `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py:200` | `s1_step_latent()` | latent → generate_traj → action 변환 |
| `internnav/model/basemodel/internvla_n1/internvla_n1.py:58` | `forward()` | LLM forward (학습 및 generate_latents 내부 호출) |
| `internnav/model/basemodel/internvla_n1/internvla_n1.py:320` | `generate_latents()` | output_ids → traj_latent 추출 |
| `internnav/model/basemodel/internvla_n1/internvla_n1.py:349` | `generate_traj()` | latent → diffusion → trajectory 생성 |
| `internnav/model/utils/vln_utils.py:140` | `S2Input`, `S2Output`, `S1Input`, `S1Output` | 시스템 간 데이터 전달 구조체 |

---

## 데이터 구조

### S2Input
```python
# internnav/model/utils/vln_utils.py:140
S2Input(
    rgb,          # np.ndarray   — S2 추론에 사용된 이미지 (FPV 또는 look_down)
    depth,        # np.ndarray   — 현재 depth 이미지
    pose,         # 4x4 identity — 현재 미사용
    instruction,  # str          — 내비게이션 지시문
    look_down,    # bool         — 카메라 하향 여부
)
```

### S2Output (셋 중 하나만 non-None)
```python
# internnav/model/utils/vln_utils.py:151
S2Output(
    output_action,   # np.ndarray       — 이산 액션 시퀀스 (↑↑← 등)
    output_pixel,    # np.ndarray [x,y] — pixel_goal 좌표 (s2_input.rgb 기준)
    output_latent,   # Tensor [1,N,D]   — trajectory diffusion용 latent
    rgb_memory,      # np.ndarray       — s2_input.rgb 그대로 저장 (S1 입력용)
    depth_memory,    # np.ndarray       — s2_input.depth 그대로 저장 (S1 입력용)
)
# rgb_memory = s2_input.rgb → look_down이면 look_down 이미지, 아니면 FPV 이미지
```

---

## 입력 텐서 Shape (H1 / Isaac 기준)

H1 카메라 raw 해상도: `camera_resolution=[640, 480]` (W,H), `resize_w=resize_h=384`
(config: `h1_internvla_n1_async_cfg.py`). depth obs는 10m로 정규화된 [0,1] 값.

### Raw observation (`agent.step`, internvla_n1_agent.py:251-252)
| 변수 | shape | dtype/범위 | 비고 |
|---|---|---|---|
| `rgb` = obs['rgb'] | `(480, 640, 3)` | uint8 [H,W,C] | FPV 또는 look_down(로봇이 아래로 pitch한 프레임) |
| `depth` = obs['depth'] | `(480, 640)` → `(480, 640, 1)` | float [0,1] | ×10 = metres. ndim==2면 newaxis 추가 (L313-316) |

### S2 (QwenVL LLM) 입력
QwenVL은 **RGB만** 소비(depth 미사용 — depth는 S1용 memory로 통과). 각 이미지는
processor가 개별 처리 → 이미지마다 자체 `image_grid_thw`, 따라서 **shape가 섞여도 무방**.
| 입력 | shape (PIL = (W,H)) | 처리 위치 | 비고 |
|---|---|---|---|
| FPV rgb (non-look_down) | `(384, 384)` PIL | policy s2_step L106 resize | rgb_list에 append, history로 재사용 |
| look_down rgb | `(640, 480)` PIL = raw | **resize 안 함** (L114-115 분기) | rgb_list/episode_idx에 미반영 |
| `s2_input.depth` | — | — | QwenVL 미사용, `depth_memory`로 S1에 전달 |

### S1 (NavDP diffusion) 입력 (`s1_step_latent`, agent.py:333-344)
goal 프레임 = `rgb_memory`/`depth_memory` (look_down이면 look_down 이미지/depth, 아니면 FPV),
cur 프레임 = 현재 obs. 둘 다 **224×224로 resize**.
| 입력 | shape | 정규화 | 비고 |
|---|---|---|---|
| `rgbs` | `[1, 2, 224, 224, 3]` | /255.0 | stack([goal_rgb, cur_rgb]) |
| `depths` | `[1, 2, 224, 224, 1]` | ×10, clip ≤ 5.0m | stack([goal_depth, cur_depth]) |

> look_down_rgb (480,640,3) / look_down_depth (480,640) 은 **S2엔 raw PIL(640×480)로,
> S1엔 224×224로 resize**되어 들어간다 — 같은 소스가 경로별로 다른 크기로 소비됨.

### (참고) BEV 모드일 때
S1 `bev`: `rgbs`의 RGB만 BEV(224)로 치환, depth는 FPV 224 유지. S2 `bev`/`fpv_bev`:
look_down 자리에 BEV(224×224 PIL) 추가/치환 — FPV(640×480)와 크기 달라도 processor가
개별 처리하므로 정상. (habitat 경로 상세는 `habitat_dual_system_flow.md`)

---

## S2 출력 분기

`s2_step()` 내부에서 LLM 출력에 따라 두 갈래로 나뉨:

```
llm_output에 숫자 포함 → pixel_goal 분기
  → pixel_goal 파싱
  → generate_latents() 호출
  → S2Output(output_pixel, output_latent)

llm_output에 숫자 없음 → 이산 액션 분기
  → parse_actions() 호출
  → generate_latents() 호출 안 됨
  → S2Output(output_action)
  → output_action에 5(↓) 포함 시 → look_down 트리거
```

---

## 케이스 1: pixel_goal 직접 예측 — QwenVL 2회

LLM이 FPV 이미지에서 바로 좌표를 출력하는 경우 (코드상 지원, 모델 학습에 따라 발생 여부 결정)

```
[s2_step(look_down=False)]  ← internvla_n1_policy.py:110
  s2_input.rgb = FPV 이미지

  QwenVL ①  model.generate()  →  "310, 240"  (숫자 → pixel_goal)
  pixel_goal = [240, 310]  (FPV 이미지 기준 좌표)
  QwenVL ②  generate_latents()  →  traj_latent

  S2Output(output_pixel=[240,310], output_latent,
           rgb_memory=FPV 이미지, depth_memory=FPV depth)

[s1_step_latent()]  ← internvla_n1_policy.py:200
  rgbs   = [FPV 이미지(rgb_memory), 현재 FPV 이미지]
  depths = [FPV depth(depth_memory), 현재 FPV depth]
  latent = traj_latent
  → generate_traj() → action
```

---

## 케이스 2: look_down 후 pixel_goal 예측 — QwenVL 3회 (1 + 2)

LLM이 이산 액션 중 ↓(5)를 출력 → look_down 후 재추론하는 경우

```
── s2_step ① ────────────────────────────────────────────────────
[s2_step(look_down=False)]  ← internvla_n1_policy.py:110
  s2_input.rgb = FPV 이미지

  QwenVL ①  model.generate()  →  "↑↑↓"  (숫자 없음 → 이산 액션)
  parse_actions() → output_action = [1, 1, 5]
  generate_latents() 호출 안 됨

  S2Output(output_action=[1, 1, 5])

[agent.step() × 2]  ← internvla_n1_agent.py:283
  ↑(1), ↑(1) 소비 → action 반환

[agent.step()]
  ↓(5) 소비
  → look_down = True, action = -1 (로봇 이동 없음)
  → output_action/pixel/latent 모두 None 초기화

── s2_step ② ────────────────────────────────────────────────────
[다음 step: agent.step()]  ← internvla_n1_agent.py:257
  look_down == True → S2 스레드 강제 트리거

[s2_step(look_down=True)]  ← internvla_n1_policy.py:110
  s2_input.rgb = look_down 이미지

  conversation_history 초기화 안 함 (이전 문맥 유지)
  conversation_history = [
    {role: assistant, content: "↑↑↓"},        ← 직전 llm_output
    {role: user,      content: look_down 이미지 + 텍스트}
  ]
  look_down 이미지는 rgb_list/episode_idx에 반영 안 함

  QwenVL ②  model.generate()  →  "310, 240"  (pixel_goal)
  pixel_goal = [240, 310]  (look_down 이미지 기준 좌표)
  QwenVL ③  generate_latents()  →  traj_latent

  S2Output(output_pixel=[240,310], output_latent,
           rgb_memory=look_down 이미지, depth_memory=look_down depth)

[s1_step_latent()]  ← internvla_n1_policy.py:200
  rgbs   = [look_down 이미지(rgb_memory), 현재 FPV 이미지]
  depths = [look_down depth(depth_memory), 현재 FPV depth]
  latent = traj_latent
  → generate_traj() → action
```

---

## generate_latents 내부  ← internvla_n1.py:320

```
입력: output_ids [1, seq+new],  pixel_values,  image_grid_thw

① embed_tokens(output_ids) → text_embeds [1, seq+new, D]
② visual(pixel_values) → image_embeds
③ text_embeds[IMAGE_TOKEN_INDEX(151655) 위치] ← image_embeds
④ input_ids 뒤에 TRAJ_TOKEN(151667) × N_QUERY 추가 (RoPE용)
   text_embeds 뒤에 latent_queries(nn.Parameter) concat
   → [1, seq+new+N_QUERY, D]
⑤ LLM forward(output_hidden_states=True)  ← Qwen2_5_VLModel.forward():1090
⑥ hidden_states[-1][:, -N_QUERY:, :]  →  traj_latent [1, N_QUERY, D]
```

---

## pixel_goal을 포함한 prompt로 latent를 추출하는 이유

**pixel_goal 예측 시 latent를 쓰지 않는 이유:**
- pixel_goal은 `model.generate()`로 autoregressive하게 토큰을 생성하는 과정
- latent query 토큰은 generation이 끝난 후에야 sequence에 붙으므로, 이 시점엔 latent 자체가 존재하지 않음

**pixel_goal 포함 후 latent를 추출하는 이유:**
- `generate_latents()`는 `[input 토큰] + [pixel_goal 토큰] + [latent query 토큰]` 전체를 한 번에 forward
- latent query가 pixel_goal 토큰을 attend할 수 있어, "어느 픽셀로 가야 하는지"가 latent에 인코딩됨
- pixel_goal 없이 뽑은 latent는 목표 위치 정보가 빠진 채 trajectory를 생성하게 됨

→ **pixel_goal 텍스트가 Chain-of-Thought 역할**: 먼저 목표를 언어로 추론하고, 그 추론 결과를 conditioning으로 삼아 trajectory latent를 생성하는 구조.

---

## generate_traj 내부  ← internvla_n1.py:349

```
입력: traj_latents [1,N,D],  rgbs [1,2,224,224,3],  depths [1,2,224,224,1]
      (rgbs[0]=rgb_memory, rgbs[1]=현재 FPV)

① cond_projector(traj_latents) → 차원 변환
② rgb_model(rgbs) → image_feat → memory_encoder → rgb_resampler → memory_tokens
③ hidden_states = cat([memory_tokens, traj_latents])
④ FlowMatchEulerDiscreteScheduler: noise → 10 step denoising
   → trajectory [num_sample_trajs, 32, 3]
⑤ traj_to_actions() → 이산 action 리스트
```

---

## 전체 흐름 요약

```
RGB/Depth/Instruction
        ↓
[S2 Thread]  s2_step()
        ↓
QwenVL ① model.generate()
        ↓
llm_output
 ├─ 숫자 포함 (pixel_goal)
 │    QwenVL ② generate_latents() → traj_latent
 │    S2Output(pixel_goal, latent, rgb_memory=입력rgb)
 │    S1: s1_step_latent(rgbs=[rgb_memory, 현재FPV], latent)
 │    → generate_traj() → action
 │
 └─ 숫자 없음 (이산 액션)
      output_action 소비 (step마다 1개)
      ├─ 일반 액션 → action 직접 반환
      └─ ↓(5) → look_down=True
               s2_step(look_down=True)
               QwenVL ② generate() → pixel_goal
               QwenVL ③ generate_latents() → traj_latent
               S1: s1_step_latent(rgbs=[lookdown_rgb, 현재FPV], latent)
               → generate_traj() → action
```

---

## 핵심 규칙

| 항목 | 내용 | 코드 위치 |
|---|---|---|
| generate_latents 호출 조건 | llm_output에 숫자 있을 때만 (look_down 여부 무관) | `internvla_n1_policy.py:184` |
| pixel_goal 좌표 기준 | s2_step에 입력된 rgb 기준 (FPV 또는 look_down) | `internvla_n1_policy.py:185` |
| rgb_memory | s2_input.rgb 그대로 저장 (pixel_goal과 같은 이미지) | `internvla_n1_agent.py:205` |
| S1 첫 번째 프레임 | rgb_memory (pixel_goal 당시 이미지) | `internvla_n1_agent.py:320` |
| S1 두 번째 프레임 | 항상 현재 step의 FPV rgb | `internvla_n1_agent.py:325` |
| look_down 트리거 | output_action에 5(↓) 소비 시 | `internvla_n1_agent.py:291` |
| look_down 시 conversation | 이전 assistant 응답 유지 (multi-turn) | `internvla_n1_policy.py:144` |
| look_down 이미지 | rgb_list/episode_idx에 반영 안 함 | `internvla_n1_policy.py:113` |
| async 여부 | S2 스레드 분리되어 있으나 step()에서 blocking wait → 실질적으로 sync | `internvla_n1_agent.py:275` |
