# nav3d 07 — instruction_labeling(b0) 연결 smoke: GT · nav3d 생성 · 노은역 (2026-10-16)

리포트 https://claude.ai/artifact/7zJ4D81tKuKp2ARtW7s4Eu · `git pull` → 5b442e5 (labeling 정지 단서 커밋, 로컬 변경과 겹침 없음).

## 환경 (이 PC)
- Gemma-4-31B/vLLM 없음, GPU 1 장 → **Qwen3-VL-8B-Instruct** (`checkpoints/Qwen/`) 를 transformers 4.57 (`.venv_label/site`, PYTHONPATH 로만) 로 띄운
  OpenAI 호환 서버 `nav3d/07_vlm_server.py` (:8200, 16.3 GiB, ~12 s/ep). model cfg `configs/model/qwen3vl8b-local.yaml`.
- labeling 코드는 안 고침. `nav3d/07_label_paths.py` 가 item(raw R2R 형식 ep, instruction 없음)만 만든다:
  gt = 릴리스 + R2R reference_path · nav3d_wp/sg = 05b 렌더 + nav3d 03 경로 RDP(0.3 m) · noeun = 06 렌더(rig 60cm_15deg) + 04 경로(habitat 좌표 변환, yaw→quat φ = θ − 90°).
- 같은 GT ep 는 source 가 달라도 같은 episode_id → 같은 VLM seed.

## 결과 (MP3D 28 ep · 노은 10 ep, 오류 0)
| | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|
| 사람 (형제 상한) | .056 | .130 | .281 | .096 |
| label gt | .030 | .141 | .280 | .095 |
| label gt seed1 | .029 | .138 | .264 | .076 |
| label wp | .000 | .137 | .263 | .077 |
| label sg | .053 | .139 | .278 | .118 |
- L1 실행 PASS · L2 생성=GT PASS (wp 가 seed 잡음 폭 안) · N1 노은 PASS (품질 9/10, 역 어휘 사용).
- 표본 28 이라 지표가 흔들림 (사람 상한 < label). 상대 비교로만.
- **label 끼리**: seed 만 다름 BLEU-4 .61 vs wp .39 / sg .40. 원인 = **경로 단어** 회전·계단 순서 일치 wp 36 % · sg 39 %:
  nav3d 경로는 곡선이라 RDP 꺾인 점이 R2R waypoint 와 다르다 (회전 합쳐짐 / 굽이가 "bear slightly" 여러 개).
- 다음: 생성 경로용 경로 단어 (창 안 누적 방향 변화로 회전 검출) 를 경로 단어 일치율로 먼저 고르기.

## 함정
- 컨테이너가 만든 파일은 root 소유 → 호스트에서 고칠 때 `docker exec ... chown -R 1000:1000`.
- pycocoevalcap 첫 실행이 Stanford CoreNLP (384 MB) 를 받는다. SPICE 는 lmdb 오류로 skip.
