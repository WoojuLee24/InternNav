# S2 입력 파이프라인 (internvla_n1_policy.py: s2_step)

## 5단계 변환 흐름

```
sources → content → conversation_history → text → inputs
(텍스트 템플릿)  (텍스트+이미지 인터리브)  (multi-turn 대화 버퍼)  (직렬화 str)  (모델 입력 텐서)
```

---

## 1. `sources`
- **형식**: `[{"from": "human", "value": str}, {"from": "gpt", "value": str}]`
- `self.conversation` 템플릿을 deep copy해서 instruction과 history image placeholder(`<image>\n`)를 채운 텍스트 조립용 임시 구조체
- look_down 시엔 빈 템플릿(`value: ""`), 이후 버려짐

## 2. `content`
- **형식**: `[{"type": "text"/"image", "text": str / "image": PIL}, ...]`
- `sources[0]["value"]`를 `<image>` 기준으로 split한 뒤, 텍스트 조각과 실제 PIL 이미지 객체를 인터리브한 리스트
- 이후 `conversation_history`에 user turn으로 append됨

## 3. `self.conversation_history`
- **형식**: `[{"role": "user"/"assistant", "content": [...]}, ...]`
- LLM에 넘길 multi-turn 대화 누적 버퍼
- **일반 step**: 매번 `[]`로 초기화 → user turn 1개만 존재
- **look_down step**: 이전 assistant 응답을 먼저 append → `[assistant, user]` 2턴 구조

### role 구분
| role | 누가 | 내용 |
|---|---|---|
| `user` | 로봇/시스템 | 이미지 + 텍스트 질문 |
| `assistant` | LLM 이전 응답 | `self.llm_output` (예: `"310, 240"`) |

## 4. `text`
- **형식**: `str`
- `processor.apply_chat_template(conversation_history, tokenize=False)`로 생성한 단일 직렬화 문자열
- conversation_history의 role/content 구조를 Qwen-VL special token 형식으로 변환
- 예: `<|im_start|>user\n...<|vision_start|><|image_pad|><|vision_end|>...<|im_end|>\n<|im_start|>assistant\n`

## 5. `inputs`
- **형식**: `BatchFeature` (dict-like tensor 묶음)
- `processor(text=[text], images=self.input_images)`로 text와 이미지를 동시에 수치화
- `model.generate(**inputs)`에 직접 전달

| 키 | 형식 | 내용 |
|---|---|---|
| `input_ids` | `LongTensor [1, seq_len]` | text를 토큰 ID 정수로 변환한 배열. 이미지 위치는 `IMAGE_TOKEN_INDEX(151655)` placeholder로 채워짐 |
| `attention_mask` | `Tensor [1, seq_len]` | 패딩 마스크 |
| `pixel_values` | `Tensor [N_patches, C*patch_h*patch_w]` | 이미지 패치 텐서. forward 내부에서 image_token 위치에 주입됨 |
| `image_grid_thw` | `Tensor [N_images, 3]` | 이미지별 (T, H, W) 그리드 정보 |

### `input_ids` 구조 예시
```
text:     "<|im_start|>user\n당신은 네비게이션... <|vision_start|><|image_pad|>...<|vision_end|> ...<|im_end|>\n<|im_start|>assistant\n"
              ↓ tokenize
input_ids: [ 151644, 872, 198, ..., 151655, 151655, ..., 151645, 198, 151644, 77091, 198 ]
                                      ↑↑↑↑↑
                                IMAGE_TOKEN_INDEX (151655) — 이미지 위치 placeholder
```

---

## 6. `output_ids` / `self.llm_output`

### `output_ids`
- **형식**: `LongTensor [1, seq_len + new_tokens]`
- `model.generate(...).sequences` 결과 — `input_ids` 뒤에 새로 생성된 토큰 ID가 이어붙은 전체 시퀀스

```
output_ids: [ ...input_ids 전체... | 843, 11, 220, 17, 19, 15 ]
             ←── input_ids.shape[1] ──→ ←── 새로 생성 ──→
```

### `self.llm_output`
- **형식**: `str`
- `output_ids[0][input_ids.shape[1]:]` 를 decode한 문자열
- 예: `"310, 240"` (pixel_goal) 또는 `"↑↑←"` (action)
- `generate_latents(output_ids, ...)` 에서 output_ids 전체를 재사용 — IMAGE_TOKEN_INDEX 위치를 다시 찾아 image embedding으로 교체 후 trajectory latent 추출

### 세 변수 관계
| | 내용 | 범위 |
|---|---|---|
| `input_ids` | prompt 전체를 토큰화한 정수 배열 | `[0, seq_len)` |
| `output_ids` | input_ids + 생성된 응답 토큰 | `[0, seq_len + new_tokens)` |
| `self.llm_output` | `output_ids[seq_len:]` 를 decode한 문자열 | — |

---

## look_down 동작 차이

**look_down의 의미**: S2가 `↓` 액션(`action == 5`)을 출력 → 로봇이 카메라를 아래로 내려다봄 → 같은 질문을 이어서 다시 추론 (독립 추론이 아닌 연속 대화)

```
[일반 step]
user: "앞을 보고 있어. 어디로 가야 해?" + 현재 이미지
  → LLM: "310, 240" → floor 잘 안 보임 → action=↓ → look_down=True

[look_down step]
assistant: "310, 240"          ← 직전 응답 컨텍스트 유지
user: "아래를 봤어. 다시 확인해." + look_down 이미지
  → LLM: 정확한 pixel_goal 재출력
```

### look_down 시 달라지는 점
- `conversation_history` 초기화 안 함 (이전 문맥 유지)
- look_down 이미지를 `rgb_list`에 추가 안 함 (히스토리 오염 방지)
- `episode_idx` 증가 안 함 (rgb_list 인덱스 정합성 유지)

---

## 관련 파일
- `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`: `s2_step()` (line 110~)
- `internnav/agent/internvla_n1_agent.py`: look_down 트리거 (line 292)
