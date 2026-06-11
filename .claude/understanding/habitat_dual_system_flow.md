# Habitat Dual System 작동 흐름 (HabitatVLNEvaluator)

## 관련 파일 및 핵심 함수

| 파일 | 함수/클래스 | 역할 |
|---|---|---|
| `internnav/habitat_extensions/vln/habitat_vln_evaluator.py:266` | `HabitatVLNEvaluator` | 평가 전체 관리 (모델 직접 로드, 단일 스레드) |
| `internnav/habitat_extensions/vln/habitat_vln_evaluator.py:468` | `_run_eval_dual_system()` | 에피소드/스텝 루프 |
| `internnav/habitat_extensions/vln/habitat_vln_evaluator.py:441` | `parse_actions()` | LLM 출력에서 이산 액션(↑←→↓ STOP) 파싱 |
| `internnav/model/basemodel/internvla_n1/internvla_n1.py:320` | `generate_latents()` | output_ids → traj_latent 추출 |
| `internnav/model/basemodel/internvla_n1/internvla_n1.py:349` | `generate_traj()` | latent → diffusion → trajectory 생성 |
| `internnav/model/utils/vln_utils.py` | `traj_to_actions()` | trajectory → 이산 action 리스트 변환 |

---

## 데이터 구조 (스텝 루프 내 상태 변수)

```python
rgb_list        # List[PIL.Image]  — 매 forward 스텝의 이미지 히스토리
look_down_image # PIL.Image        — 현재 스텝에서 LOOKDOWN×2로 캡처한 이미지
look_down_depth # Tensor [224,224] — 현재 스텝에서 LOOKDOWN×2로 캡처한 depth
messages        # List[dict]       — multi-turn chat history (processor 형식)
action_seq      # List[int]        — LLM 이산 액션 출력 버퍼 (소비 방식)
pixel_goal      # [x, y]           — LLM이 예측한 pixel 좌표 (forward 이미지 기준)
local_actions   # List[int]        — NavDP 출력 action 버퍼 (소비 방식)
traj_latents    # Tensor           — generate_latents() 결과 (NavDP 입력)
pix_goal_image  # Tensor [224,224,3] — pixel_goal 예측 당시 look_down_image (NavDP 재실행용)
pix_goal_depth  # Tensor [224,224,1] — pixel_goal 예측 당시 look_down_depth (NavDP 재실행용)
forward_action  # int              — pixel_goal 추적 중 소비된 스텝 수 (MAX_STEPS=8 초과 시 리셋)
```

---

## LLM 출력 분기

`model.generate()` 출력에 따라 두 갈래로 나뉨:

```
llm_outputs에 숫자 포함 → pixel_goal 분기
  → pixel_goal 파싱 ([coord[1], coord[0]])
  → generate_latents() 호출 → traj_latents
  → generate_traj() 호출 → local_actions (MAX_LOCAL_STEPS=4개)

llm_outputs에 숫자 없음 → 이산 액션 분기
  → parse_actions() → action_seq (↑←→↓ STOP)
  → generate_latents() 호출 안 됨
  → ↓(LOOKDOWN, 5) 포함 시 → env.step(LOOKDOWN)×2 후 look_down obs 수집
```

---

## 케이스 1: forward 이미지 → pixel_goal 예측 (action != LOOKDOWN)

LLM이 forward 이미지에서 바로 좌표를 출력하는 경우

```
[스텝 시작 — action != LOOKDOWN, 카메라 0°]
  rgb_list.append(현재 forward 이미지)

  [look_down 관측 수집 — NavDP 전용]
    env.step(LOOKDOWN) × 2  → 카메라 60° 아래
    look_down_image = down_obs["rgb"]
    look_down_depth = preprocess_depth(down_obs["depth"])
    env.step(LOOKUP) × 2   → 카메라 0° 복원

[LLM inference]  model.generate()  ← 카메라 0° 상태
  input_images = [rgb_list[history_id], 현재 forward 이미지]
  messages = [{role:user, content: instruction + forward 이미지들}]
  → llm_outputs = "310, 240"  (숫자 → pixel_goal)

[pixel_goal 처리]
  pixel_goal = [coord[1], coord[0]] = [240, 310]  (forward 이미지 기준)
  ⚠️ env.step(LOOKUP) × 2  ← 카메라 이미 0°인데 추가 실행 → -60° (위) 또는 clamp no-op
     (코멘트 "look down --> horizontal"은 action==LOOKDOWN 케이스 기준으로 작성됨)

  generate_latents(output_ids, pixel_values, image_grid_thw)  →  traj_latents

  [NavDP 입력 구성]
    image_dp = look_down_image.resize(224,224)  ← 이번 스텝 LOOKDOWN에서 캡처
    pix_goal_image = copy(image_dp)             ← 재실행용 저장
    images_dp = stack([pix_goal_image, image_dp])  (shape: [1,2,224,224,3])
    depth_dp = look_down_depth
    pix_goal_depth = copy(depth_dp)             ← 재실행용 저장
    depths_dp = stack([pix_goal_depth, depth_dp])

  generate_traj(traj_latents, images_dp, depths_dp)  →  dp_actions
  traj_to_actions(dp_actions)  →  local_actions (최대 MAX_LOCAL_STEPS=4개)

[action 반환]
  action = local_actions.pop(0)

[env.step(action)]
  step_id++, messages = []
```

---

## 케이스 2: 이산 액션 → LOOKDOWN 없이 이동

LLM이 화살표 문자열을 출력하는 경우

```
[스텝 시작 — action != LOOKDOWN]
  rgb_list.append(현재 forward 이미지)

  [look_down 관측 수집]  ← pixel_goal 예측 안 해도 항상 실행
    env.step(LOOKDOWN) × 2  → look_down_image, look_down_depth 캡처
    env.step(LOOKUP) × 2   → 수평 복원

[LLM inference]  model.generate()
  → llm_outputs = "↑↑←"  (숫자 없음 → 이산 액션)
  parse_actions() → action_seq = [1, 1, 2]
  (generate_latents 호출 안 됨)

[action 소비]
  스텝마다 action_seq.pop(0) → action 반환
  action_seq 소진되면 다시 LLM inference

[env.step(action)]
  step_id++, messages = []
```

---

## 케이스 3: 이산 액션 → LOOKDOWN 포함 → pixel_goal 예측 (action == LOOKDOWN)

LLM이 ↓(5)를 출력해 look_down 상태로 전환 후 재추론하는 경우

```
── 스텝 N ────────────────────────────────────────────────────────
[LLM inference]  model.generate()
  → llm_outputs = "↑↑↓"
  parse_actions() → action_seq = [1, 1, 5]

[스텝 N+1, N+2]  action_seq 소비
  action = 1(↑) → env.step(↑), step_id++, messages=[]
  action = 1(↑) → env.step(↑), step_id++, messages=[]

[스텝 N+3]  action = 5(↓) 소비  (line 783-785)
  env.step(LOOKDOWN) × 2  → 카메라 60° 아래
  step_id 증가 안 함, flag = True

── 스텝 N+4 — action == LOOKDOWN ─────────────────────────────────
[관측 수집 — action == LOOKDOWN 분기, line 550]
  카메라 이미 60° 아래 (추가 env.step 없음)
  look_down_image = 현재 obs["rgb"]  ← LOOKDOWN 상태의 이미지
  look_down_depth = preprocess_depth(현재 obs["depth"])

[LLM inference]  model.generate()  ← 카메라 60° 아래 상태
  input_images = [look_down_image]  (line 593)
  conversation 이어서:
    messages = [
      {role: assistant, content: "↑↑↓"},      ← 직전 llm_output 유지
      {role: user,      content: look_down_image + 텍스트}
    ]
  → llm_outputs = "310, 240"  (pixel_goal, look_down 이미지 기준)

[pixel_goal 처리]
  pixel_goal = [240, 310]  (look_down 이미지 기준)
  env.step(LOOKUP) × 2   → 카메라 60° → 0° 복원 ✓  (line 663-664의 의도)

  generate_latents()  →  traj_latents
  images_dp = stack([look_down_image, look_down_image])  ← goal=current 동일
  generate_traj()  →  local_actions

[env.step(action)]
  step_id++, messages=[]
```

---

## NavDP 재실행 (local_actions 소진 시)

pixel_goal이 설정된 상태에서 local_actions가 비면, NavDP만 재실행:

```
[local_actions 소진]
  image_dp = 현재 스텝 look_down_image  ← 이번 스텝의 새로운 look_down
  images_dp = stack([pix_goal_image, image_dp])   ← goal은 pixel_goal 당시 이미지 고정
  depths_dp = stack([pix_goal_depth, depth_dp])
  generate_traj(traj_latents, images_dp, depths_dp)  →  local_actions
  (generate_latents / LLM 재실행 없음)
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
⑤ LLM forward(output_hidden_states=True)
⑥ hidden_states[-1][:, -N_QUERY:, :]  →  traj_latent [1, N_QUERY, D]
```

---

## generate_traj 내부  ← internvla_n1.py:349

```
입력: traj_latents [1,N,D],  rgbs [1,2,224,224,3],  depths [1,2,224,224,1]
      (rgbs[0]=pix_goal_image, rgbs[1]=현재 look_down_image)

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
[매 스텝 시작: action != LOOKDOWN 이면]
  LOOKDOWN×2 → look_down_image/depth 캡처 → LOOKUP×2
        ↓
[LLM inference]  model.generate()  (forward 이미지 기준)
        ↓
llm_outputs
 ├─ 숫자 포함 (pixel_goal)
 │    generate_latents() → traj_latents
 │    generate_traj(look_down_image) → local_actions (4개)
 │    action = local_actions.pop(0)
 │    local_actions 소진 → generate_traj만 재실행 (traj_latents 재사용)
 │    forward_action > MAX_STEPS(8) → pixel_goal 리셋
 │
 └─ 숫자 없음 (이산 액션)
      action_seq 소비 (스텝마다 1개)
      ├─ 일반 액션 (↑←→) → env.step() 직접
      └─ ↓(5, LOOKDOWN)
               env.step(LOOKDOWN)×2
               다음 스텝: action==LOOKDOWN → look_down obs 수집
               LLM 재추론 (conversation 이어서)
               → pixel_goal → generate_latents + generate_traj
```

---

## H1 (Isaac Lab) 과의 핵심 차이

| 항목 | Habitat `dual_system` | H1 `partial_async` |
|---|---|---|
| **S1/S2 분리** | 없음 — 단일 모델, 단일 스레드 | S1/S2 별도 클래스, S2 백그라운드 스레드 |
| **look_down 수집 타이밍** | 매 스텝 LOOKDOWN×2/LOOKUP×2 항상 실행 | S2가 ↓(5) 출력 시에만 트리거 |
| **LLM이 보는 이미지** | 기본: forward 이미지 / LOOKDOWN 상태일 때만 look_down | H1 케이스2: look_down 이미지로 LLM 재추론 |
| **pixel_goal 좌표 기준** | forward 이미지 기준 (케이스 1) 또는 look_down 기준 (케이스 3) | look_down 이미지 기준 (케이스 2) |
| **NavDP 첫 번째 프레임** | pix_goal_image = pixel_goal 당시 look_down | rgb_memory = pixel_goal 당시 이미지 |
| **conversation 리셋** | env.step() 실행 후 messages=[] 초기화 | look_down 시 이전 assistant 응답 유지 |
| **action 공간** | 이산 (STOP/FWD/LEFT/RIGHT/LOOKUP/LOOKDOWN) | 연속 velocity trajectory |

---

## 핵심 규칙

| 항목 | 내용 | 코드 위치 |
|---|---|---|
| look_down 수집 조건 | action != LOOKDOWN인 매 스텝마다 항상 수행 | `habitat_vln_evaluator.py:566-585` |
| LLM 입력 이미지 | forward 이미지 (action==LOOKDOWN일 때만 look_down) | `habitat_vln_evaluator.py:550-564` |
| pixel_goal 좌표 기준 | LLM에 입력된 이미지 기준 (forward 또는 look_down) | `habitat_vln_evaluator.py:657-659` |
| generate_latents 호출 조건 | llm_outputs에 숫자 있을 때만 | `habitat_vln_evaluator.py:655` |
| pix_goal_image 고정 | pixel_goal 예측 당시 look_down_image를 저장, 이후 재사용 | `habitat_vln_evaluator.py:675-676` |
| NavDP 재실행 조건 | local_actions 소진 시 (traj_latents 재사용, LLM 재실행 없음) | `habitat_vln_evaluator.py:711-731` |
| messages 리셋 조건 | env.step() 실행 후 (LOOKDOWN 스텝은 리셋 안 함) | `habitat_vln_evaluator.py:789-791` |
| forward_action 리셋 조건 | MAX_STEPS(8) 초과 또는 action==STOP 시 | `habitat_vln_evaluator.py:736-751` |
| ⚠️ debugpy 블로킹 | wait_for_client() 항상 실행 — 디버거 없으면 무한 대기 | `habitat_vln_evaluator.py:336-337` |
