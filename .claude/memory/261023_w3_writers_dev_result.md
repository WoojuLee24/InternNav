# 작성자 3명(w3-v1) dev 결과 — 2026-10-23 18:40 KST

목적: 형제 3문장을 서로 다른 "작성자"가 쓰게 해 GT 처럼 다양하게 (GT 통계 사용 없음). 이동 정보는 숨기지 않고 그림으로 준다.

| 작성자 | 이름 | 입력 |
|---|---|---|
| A | `b1` | 사진 + 번호 목록 + 방 순서 (기존) |
| B | `b1_overlay-arrow-v1_motion-none` | waypoint 도착 사진(돌기 전) 위에 진행 방향 화살표(아래 흰 원) + 바닥 초록선, 번호 목록 없음, 방 순서 줄 유지 |
| C | `b1_select-topdown-v1_motion-none` | .house 방 다각형 평면도 + 경로(출발 방향이 위) + 출발·도착 사진, 번호 목록 없음, 방 순서 줄 유지 |
| 섞음 | `w3-v1` (아직 파일 없음) | 경로마다 A·B·C 를 형제 3개에 무작위 배정 (seed `0:<traj>:w3`) |

코드: `label/visual.py` (e2e.build_request 가 guard clause 로 위임), `label/viz_request.py` (요청 이미지·프롬프트 저장),
config `frames.overlay`, `frames.select=topdown-v1`, `route.motion` (기본값이면 cfg None → 기존 gen_hash 불변 확인: b0 aba44a796514, b1 a69461a93ffb).

명령:
- `/usr/bin/python scripts/dataset_converters/instruction_labeling/label/annotate.py --method b1 --set frames.overlay=arrow-v1 --set route.motion=none --split val_unseen --subset dev`
- `/usr/bin/python scripts/dataset_converters/instruction_labeling/label/annotate.py --method b1 --set frames.select=topdown-v1 --set route.motion=none --split val_unseen --subset dev`
- 시각화: `label/viz_request.py --method b1 --set <위와 같음> --n 4` → `logs/instruction_labeling/viz_request/<이름>/ep_*.jpg`

dev 303 episode (101 trajectory):

| 세트 | 첫머리 이동/방나감/회전 % | 끝 stop/wait/없음 % | 단어 | 회전 덜 말함 | 도착방 | 형제 첫3 같음 | 형제 내용어 겹침 | 어휘 |
|---|---|---|---|---|---|---|---|---|
| GT | 56/17/17 | 53/29/16 | 27.2±11.3 | 31 % | 34 % | 5.9 % | 0.20 | 491 |
| A b1 | 30/6/64 | 98/0/2 | 26.4±3.0 | 0 % | 94 % | 77.6 % | 0.70 | 233 |
| B 화살표 | 36/45/17 | 97/0/3 | 26.4±3.2 | 49 % | 92 % | 48.5 % | 0.64 | 244 |
| C 지도 | 49/23/27 | 95/0/5 | 23.9±3.2 | 47 % | 99 % | 42.6 % | 0.64 | 185 |
| **w3 섞음** | 39/24/36 | 96/0/4 | 25.6±3.4 | 31 % | 96 % | **17.5 %** | **0.45** | 253 |

- 작성자끼리 같은 episode 내용어 겹침 A-B 0.49, A-C 0.49, B-C 0.46 (모델만 바꾼 Qwen 0.46 수준).
- 출발 회전 ≥90° 에서 회전으로 시작: GT 24 %, A 100 %, B 17 %, C 44 %.
- 안 바뀐 것: 끝맺음 stop 96 %, 도착 방 92~99 %, 길이 편차 3 (다음 단계 과제).
- 남은 확인: B·C 가 회전을 덜 말하는 것이 정보 손실인지 → val_unseen 전체 생성 + 정보량(GT→label) 검사.

## 진행 (2026-10-23 KST)
- C 의 평면도를 처음엔 pyplot 으로 그려 64 스레드에서 그림이 섞일 위험 → Figure API + lock 으로 고치고 C 전체 재생성 (위 표는 재생성 기준).
- 전체 생성: B·C 각각 train 10,684 / val_unseen 1,839, 오류 0, 품질검사 실패 B 14·33, C 27·109 (val·train).
- commit 3d106fc. node2 큐 (`sleep 1800` 앞, 백업 node2.txt.bak_1023_w3info): 준비 → 정보량 B → 정보량 C. 예상 완료 약 23:50 KST.
