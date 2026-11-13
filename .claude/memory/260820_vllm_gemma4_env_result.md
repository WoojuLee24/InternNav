# gemma-4-31B vLLM 환경 확인 결과 (2026-08-20)

## 결론: 설치할 필요 없다 — 이미 서버가 떠 있고, 이 컨테이너에서 바로 호출된다

다른 user가 띄운 vLLM 서버가 **host network를 통해 internnav apptainer 컨테이너 안에서 그대로 접근 가능**하다.
apptainer는 기본적으로 host network namespace를 공유하므로 추가 bind/포트포워딩이 필요 없다.

| 항목 | 값 |
|---|---|
| endpoint | `http://127.0.0.1:8000` (0.0.0.0 bind → `http://10.77.16.191:8000` 로도 접근 가능) |
| model id | `cyankiwi/gemma-4-31B-it-AWQ-4bit` |
| vLLM | **0.24.0** (`system_fingerprint: vllm-0.24.0-5a2839c3`) |
| max_model_len | 4096 |
| GPU | GPU **3** 이 90.8GB 점유 (TP=1 로 보임. GPU 0,1,2 는 free) |
| API key | 없음 (그대로 호출됨) |
| 검증된 route | `/health` `/version` `/v1/models` `/v1/chat/completions` `/v1/completions` `/tokenize` |
| 없는 route | `/v1/embeddings` `/pooling` (generate-only 로 기동됨) |
| 멀티모달 | **동작 확인됨** — 224×224 PNG data URI 입력 → "빨간색 원입니다." (이미지 1장 ≈ 263 tokens) |

호스트: `OnVLM2-A` (10.77.16.191 / 10.77.32.231). 포트 8080 에는 다른 user의 웹앱(OnVLM2-A UI)이 떠 있다.

## 검증 명령 (실제 실행해서 통과함)
```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"cyankiwi/gemma-4-31B-it-AWQ-4bit","messages":[{"role":"user","content":"한 문장으로 자기소개 해줘."}],"max_tokens":64,"temperature":0}'
```
이미지 입력은 `messages[].content` 를 `[{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}},{"type":"text","text":"..."}]` 로 준다.

클라이언트 패키지도 추가 설치 불필요 — `urllib`/`curl` 로 충분. `openai` SDK 를 쓰고 싶으면
isaacsim python 에 `openai` 만 넣으면 되고(순수 python 패키지, torch 안 건드림), `base_url="http://127.0.0.1:8000/v1"`, `api_key="EMPTY"`.

## 사용 시 주의
- **GPU3 은 남의 것**이다. 서버 프로세스를 죽이거나 재시작하면 안 된다 (컨테이너 PID namespace가 달라서 `ps` 에도 안 보인다).
- 언제든 내려갈 수 있는 서비스다. InternNav 코드에서 의존할 거면 `/health` 체크 + 실패 fallback 을 두자.
- `max_model_len=4096` 고정이라 이미지 1장(≈263 tok) 기준 최대 ~15장. 더 긴 컨텍스트가 필요하면 내 서버를 따로 띄워야 한다.

## 직접 띄워야 할 경우 (참고)
- 이 서버 컨테이너(`internnav` sandbox: torch 2.7.0+cu128, transformers 4.51.0, python 3.11)에는 **절대 설치하지 말 것** — vllm 0.24.0 은 `torch==2.11.0`, 0.27.1 은 `torch==2.13.0` + `transformers>=5.5.3` 을 강제해서 IsaacLab/InternNav 가 깨진다.
- 서버에 공유된 유일한 apptainer 이미지: `/data/apptainer_images/pytorch-2.12.0-cuda13.2-cudnn9-devel.sif`
  (6.7GB, `docker.io/pytorch/pytorch:2.12.0-cuda13.2-cudnn9-devel` 기반, apptainer 1.5.1, 2026-06-16 빌드, world rx)
  - 단 torch 2.12.0 은 어떤 vllm 버전 pin 과도 안 맞음(0.24~0.26 → 2.11.0, 0.27.x → 2.13.0). 이미지의 torch 는 재사용 못 하고 venv 에서 새로 받게 된다. base(CUDA 13.2 런타임)로만 유용.
  - 다른 user 홈(`/data/users/{gs,ccsmm,yongseop,admin}`)은 0700 이라 그쪽 venv 내용은 확인 불가.
- 이 컨테이너 안에는 `apptainer` CLI 가 없다 → 새 인스턴스는 host 셸에서 띄워야 함.
- 하드웨어는 충족: RTX PRO 6000 Blackwell ×4 (96GB, sm_120), driver 580.173.02 (CUDA 13 wheel 요구 ≥580 OK), `/dev/shm` 126G.
- 디스크: 컨테이너 `/`(overlay) 9.1G 뿐 → venv/HF 캐시는 `/ws` 또는 `/data`(3.7T) 에 둘 것. 모델 weight 약 21GB.

## 모델 사실 확인 (HF config)
- arch `Gemma4ForConditionalGeneration` → vllm `gemma4_mm` (0.24.0 에 이미 존재)
- **이름만 AWQ**. 실제로는 compressed-tensors `pack-quantized` int4, group_size 32, asymmetric. `--quantization awq` 주면 안 됨(자동 감지).
- dtype float16, vision tower + audio token id 보유(멀티모달), weight 총 ~21GB → 96GB 1장으로 TP=1 충분.
