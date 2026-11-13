# 노은역 랜드마크 .house 이어받기 — 5090 (2026-11-13)

- 코드: `noeun-landmark-house` 06298fb (5090 로컬 미커밋 수정은 stash → checkout → pop, 충돌 없음).
- 꾸러미 sha256 OK. navmesh·glb·verify.json 은 기존 것과 md5 동일. holes.csv 는 minor 1행만 다름 (CRITICAL 197 동일), 기존 것 백업 후 덮어씀.
- 3. 준비 확인: load_objects 690/690.
- (가) `.house` → DATA/scene_data/noeun_station/noeun_station.house (물체 690 · 범주 77 · region 3). corrections.json 없이.
- (나) 랜드마크 515 → 471 → 목적지 1,412 (기대값과 같음). logs/goal_candidates/goals.json.
- (다) **멈춤**: `22_select_paths.py` 는 goals.json 을 안 읽는다 (출발·도착을 navmesh 위에서 무작위로 뽑음 + 03 층이동 경로).
  그대로 돌리면 랜드마크 목적지와 무관한 경로가 나오고, 기본 out_dir 의 기존 noeun_station_paths.json (10-16) 을 덮어쓴다.

## 최종 랜드마크 지도 (사람 필터 반영)
- 사람이 고친 것 = 리포트(d7945fda) db `corrections/latest` (11/13 15:36 KST). 편집기(8mxw3GgS) `edits/latest` (13:46) 의 버림 204 는 전부 여기에 포함.
- 적용: 812 -> **277** (34 종). 저장 `logs/landmark_final/{corrections.json, scene_landmarks/, viz/landmark_map.png}`.
- 71 수정: landmark_map 저장에 bbox_inches='tight' (긴 이름이면 범례 잘림). 한글 폰트 = `.venv_label/fonts/NanumGothic-Regular.ttf` (google/fonts).
- 주의: 277 x 3 자리 = 최대 831 < 필요 1,280. `.house`·goals 는 아직 보정 전 (70·73 --corrections 같이 돌려야 함).

## 노은역 랜드마크 걸어 보기 도구 (Isaac 검증용)
- teleop_view.py (수정 없음) = RBQ10 걷기·렌더 :8765 · noeun_landmark_walk.py = 페이지 서버 :8766 (teleop 전달 + 랜드마크·.house·수정)
- 지도 배경: tools/noeun_walk_bg.py -> nav3d/logs/noeun_walk/noeun_station/paint_{bg,meta}_<층>.{png,json}
- 화면: 지도(로봇 점·부채꼴·랜드마크 .house 타원) · 카메라(랜드마크 되찍기, 메쉬 광선으로 가린 것 숨김) · 오른쪽 .house 정보 수정 + 69 스캔 사진
- 수정은 logs/landmark_final/corrections.json 에 저장 (처음 저장 때 .bak_<시각>). 새 키: include_ids · common_ids · override_ids (corrections.py, 70 이 반영)
- 서버 숫자 = 70 과 같음 (812 · 남김 277 · .house 242). 분할 mask 는 69 가 저장 안 해서 없음 (검출 점만)
- SAM mask: 69 --save_masks (기본 끔) 로 검출 단계만 다시 (--stage detect, names/vocab 복사) -> logs/scene_landmarks_masks/{detections.json, masks/*.png 4,996 장 21 MB}. 5090 611 초.
  원래 4,998 건과 거의 같음 (종류별 차이 ±2). 랜드마크 812 전부에 mask 검출이 붙음 (70 과 같은 규칙: 원래 이름 · 1.5 m 안).
  noeun-isaac 컨테이너에 embreex pip 설치 (없으면 광선 계산이 매우 느림). transformers 4.51 (6000pro 는 5.22).
- 걸어 보기 도구에 분할 칸 추가 (mask 둘레 확대, 누르면 전체 사진 + 상자).
- 실시간 분할: tools/noeun_seg_server.py (noeun-isaac, :8768, SAM vit-base). 페이지 서버가 /seg/* 를 전달.
  .house 상자 모드 = .house 중심·축·반지름 8 꼭짓점 투영 -> 상자 -> SAM. 실측 약 110~125 ms/장 (약 8 장/s, Isaac 렌더와 같은 GPU).
  검출기 모드 = DINO 어휘 2 덩어리 + SAM, 약 0.5 s (첫 호출은 모델 올림 3 s). GPU 합계 약 13.5 GB.
  예: 벤치 앞에서 .house 상자의 bench 는 다리·기둥만 칠함 -> .house 상자가 의자 면과 어긋남 (도구로 확인할 대상)

## 2026-11-13 걸어 보기 도구 — 화면 눌러 물체 추가 (SAM2 · GroundingDINO · Gemma · 직접 입력)
- 사용: 페이지 "+ 화면 멈추고 추가" → 카메라 화면 누르기(점)/Shift(빼는 점)/드래그(상자) → 분할 → 3D(메쉬 광선) → 이름(추천 또는 직접) → 추가 저장.
- 모델 고르기: 분할 = SAM2.1 hiera base+ (`checkpoints/sam2.1-hiera-base-plus`, 첫 클릭 1.2 s, 이후 12~17 ms) / SAM vit-base (0.27 s).
  추천 = GroundingDINO (어휘 100, 약 3 s, 붙은 라벨은 어휘 단어로 다시 쪼갬) / Gemma (OpenAI 형식 `--vlm_url :8200`, 서버 꺼져 있으면 "Gemma 서버 꺼짐").
- 저장: corrections.json `add` [{id ≥100000, name, xyz, points(≤400), source}] + alias_ids/common_ids. corrections.apply 가 lms 에 제자리로 덧붙임 → 70·73·goal_rules·페이지가 `points` 로 상자 계산.
- 검증(복사본 corrections, 포트 8767): 페이지 812/277/242 그대로, 추가 1 → 70 물체 243 (load_objects 243), 73 추가 물체 목적지 3 곳. 실제 corrections.json 은 안 건드림.
- sam2 패키지: noeun-isaac 에 `pip install --no-deps git+facebookresearch/sam2` (SAM2_BUILD_CUDA=0) + hydra-core iopath. torch 2.7 그대로.
- Gemma: `checkpoints/gemma-4-31B-it-qat-q4_0-gguf` (google 공식 QAT q4_0, 17.7 GB + mmproj 1.2 GB) 내려받기만. 사용자: CPU 로 넘겨 돌리지 말 것.
  GPU 남은 메모리 ~19 GB < 모델 ~19 GB + KV → teleop(7.7 GB)·seg(4.5 GB) 켠 채로는 못 띄움. llama.cpp 빌드는 자동 권한에서 막힘(외부 코드).
- 남은 테스트 찌꺼기(root 소유라 못 옮김): logs/_walk_test_house, logs/_walk_test_goals, logs/_walk_test_corr.json.bak_*.

## 2026-11-13 상자 다시 맞추기 (refit)
- 랜드마크 고르고 "화면에서 다시 맞추기" → 화면 클릭/상자 → SAM2 → 3D 점 → "이 모양으로 상자 저장".
  corrections.json override_ids[id] = {points(≤400), refit_at, source}. 손으로 넣은 center/radii 는 지운다 (그 뒤 손으로 다시 고치면 덧씌움).
  corrections.apply 가 override points 를 l['points'] 로 -> 70/73/goal_rules/페이지가 그 점으로 축·반지름·회전까지 다시 계산.
- "상자 원래대로" = center·radii·points 모두 지움 (검출 점 상자로 복귀) — 테스트에서 원래 값과 같음 확인.
- 고친 버그: 분할 픽셀 600 개를 저장 때 앞 400 개로 잘라 상자가 치우침 → 분할 서버가 처음부터 고르게 400 개.
- 사용자가 실제로 'poster' (#100000) 를 추가해 둠 (11:13). 실제 페이지 813/278/243.

## 2026-11-13 "라벨링한 상자가 .house 에서 더 위에" — 원인·조치
- 원인: 'poster' #100000 은 600→앞 400 자르기 버그가 고쳐지기 전(11:13)에 저장 → 위쪽 점만 남아 상자 중심 1.30 m (전체 중앙값 1.03 m), 약 28 cm 위. 버그는 이미 고침 (분할 서버가 고르게 400 개). 기존 항목은 다시 맞춰야 함.
- 같이 고침: 도구들이 카메라 자세를 yaw 만 썼음 (렌더는 몸 회전 전체). teleop_view /state 에 'R'(3x3) 추가 → seg 서버·페이지 서버 광선·페이지 되찍기가 R 사용 (없으면 yaw). 실측 기울기 1.7° 에서 차이 ~1 cm, 서 있을 때 4.4° 까지 봄.
- .house 상자 모드: 상자가 화면 위에 붙으면 이름표가 화면 밖 → 상자 안에 그리게.
- teleop 재시작함 (로봇 위치 되돌려 놓음).

## 2026-11-13 고른 물체의 .house 상자·분할 보기
- 랜드마크를 고르면 카메라 화면이 `/seg/house.jpg?id=N` 로 바뀜: 그 물체 .house 3D 상자 선 12 개(자홍) + 그 상자로 SAM 분할. 약 110 ms/장.
  가려짐 = 상자만 (SAM 안 돌림), 화면 밖·카메라 뒤 = 제목에 표시. 버튼 "고른 물체 .house 상자·분할" 로 끔/켬. 분할 서버 꺼지면 5 s 맨 화면.
- 다시 맞추기: 멈춘 화면에 지금 .house 상자(청록 선)를 그림 → 새 SAM2 모양(자홍)과 비교.
- 확인: #100000 poster 상자가 왼쪽 세움 간판 위에 그려짐, 아래쪽이 비어 있음 (예전 400 자르기 버그 흔적).

## 2026-11-13 지도에서 로봇 놓기 (카메라 시점 키는 사용자가 원치 않아 되돌림)
- 지도: 빈 곳 누르고 끌기 = 그 자리·방향 / Shift+끌기 또는 "로봇 놓기" 버튼(Esc 끔) = 랜드마크 위라도 / 로봇 점 끌기 = 제자리 방향만.
  끄는 동안 놓을 자리 미리보기 (부채꼴·화살표·각도). 서버는 기존 /teleport (x, y, yaw, tag) 그대로.
- teleop_view /state 에는 R (몸 회전) · C (실제 카메라→세계 4x4) 만 남김. 도구들은 C 우선, 없으면 R, 없으면 yaw.
- 추가한 물체도 '빼기'로 통일 (사용자 결정). 삭제 버튼은 만들었다가 지움. 빼기/되살리기가 추가한 물체에도 동작 (복사본 확인).

## 2026-11-13 사람이 지정한 점의 상자 계산 바꿈 (어긋남 원인)
- 원인: 3D 점은 맞음 (원래 화면에 되찍으면 지정 상자와 일치). 점->상자 pca_box (중앙값 ± 90%, PCA 축) 가 가까운 쪽 픽셀 몰림에 쏠리고 축이 기움.
- 70 `points_box(P, lo=1, hi=99)`: 위아래 축 = 수직, 가로 = 평면 SVD, 축마다 1~99 백분위 범위의 가운데·반폭. 사람이 지정한 점(add · override points)에만 사용 — 70 / goal_rules / 페이지 같게. 69 검출 상자는 그대로.
- 저장된 24 개로 재 본 원래 화면 오차(상자 2D vs 점 2D 최대 px): pca 중앙 15 · 최대 173 → points_box(1~99) 중앙 9 · 최대 89. 문(doorway)은 열린 곳 너머를 맞혀 여전히 큼.
- 사용자 의견: stairs · yellow tactile paving 은 상자(instance)보다 semantic segmentation 이 맞아 보인다 → 제안·질문 중.

## 2026-11-13 계단 = 위·아래 끝 · 점자블록 전부 제외 (사용자 결정)
- corrections.json: "stairs": "ends" + exclude_types 에 'tactile paving' (백업 corrections.json.bak_*_before_stairs).
- 새 stair_ends.py: navmesh 계단 덩어리(NavScene.stairs_tri)마다 기울기 방향 → 가장 낮은/높은 꼭짓점 = 입구.
  입구 바깥 1.2 m 안(입구 5 곳 × 12 방향)에 계단 아닌 바닥이 있고 그 높이가 navmesh 층(levels 6 개) 0.5 m 안이면 끝.
  층 사이 높이 = 참(계단 중간) → 끝 아님. 결과: 덩어리 12 → 끝 17 (아래 9 · 위 8). selftest 있음.
- inject(): 69 의 'stairs'(이름 고친 뒤, escalator·stair treads 포함) 55 개 빼고 끝 랜드마크 id 200000+2·덩어리+(위=1),
  xyz = 입구+0.3 m, points = 입구 + 1.5 m 안 계단 꼭짓점. 70 · 73 · 페이지 같은 함수.
- 73: 계단 끝 목표 = 입구, 자리는 입구 바깥 쪽 · 바닥 높이 ±0.25 m 만 (STAIR_SIDE). part = stairs_top/stairs_bottom.
  navmesh 계단 라벨이 디딤판 일부를 놓쳐 자리가 계단 위에 생기던 것 막음.
- 73 다시 돌림 (logs/goal_candidates_final): 목적지 402, 계단 물체 13 (끝 17 중 16 에 자리). .house 최종(70)은 아직 안 만듦.
- 페이지: 랜드마크 844 · 남김 216 · .house 206. 걸을 수 없는 에스컬레이터 등 navmesh 에 없는 계단은 빠짐.

## 2026-11-13 가는 물체(이젤) 상자가 판에만 생기던 것
- 원인 3 개: (1) 마스크 가장자리 2 px 깎기 → 다리 거의 사라짐 (400 점 중 다리 14), (2) 다리 옆 광선이 뒤 바닥에 맞아 거리 밖 → 버림,
  (3) SAM2 가 그린 상자 밖에 부스러기 (상자 끝 486, 마스크 625) → 상자 늘어남.
- 고침: 분할 서버 = 깎기 없애고 **줄마다 같은 수**로 400 점 · 픽셀마다 '안쪽'(가장자리 3 px 넘게) 표시 · 상자 밖 자르기 · 2 % 미만 떨어진 조각 지우기.
  페이지 광선 = 물체 거리 범위를 안쪽 픽셀로만 재고, 가는 부분·가장자리는 물체 거리 ±0.15 m 일 때만 맞은 점 사용, 아니면 그 광선 위 물체 거리(중앙값)에 둠.
- 재현 확인 (저장된 자세·상자로 다시): 이젤 #486 · #196 다리 끝까지, 자판기 #100002 · 카운터 #308 그대로 맞음.
- 이미 저장된 이젤들(#486 196 586 100011 100012 100013)은 예전 계산 → 다시 맞춰야 함.

## 2026-11-13 스크린도어 = 줄마다 물체 하나, 열리는 부분 = 'screen door doorway' (사용자 결정)
- 새 merge_lines.py (selftest 있음). corrections "merge_lines": {"screen doors": {gap_m 12, perp_m 1.5, openings "doorway", open_perp_m 1.0}}.
  줄 방향 = 이웃 방향 텐서 주축 (전체 PCA 는 평행 두 줄에서 9° 기욺). 같은 층 · 줄 방향 ≤ 12 m · 직각 ≤ 1.5 m 를 union-find.
  합친 물체 id 300000+, 점 = 구성원 검출 점 + 사람 점 → points_box (줄 따라 똑바른 상자).
- (사용자: 열리는 부분 17 개는 이름 그대로 doorway — 줄로 잇기만. openings 이름 바꾸기 코드는 지움)
  (.house 다른 이름은 범주마다라 doorway 전체에 'screen doors' 가 묻던 것 막음)
- 결과: 스크린도어 36 → 줄 3 (B2 북 10 · 남 24, B1 미닫이 2), 열리는 부분 17. 70 시험 물체 188 · 범주 40. 73 다시 (399).
  백업 corrections.json.bak_*_before_lines. B2 지도 그림 확인: 두 줄 상자가 선로 쪽 가장자리를 따라 얇게.

## 2026-11-13 스크린도어 높이 · 화살표 목적지 제외
- 스크린도어 줄 상자 높이가 바닥 위 0.1~1.5 m 뿐이었음 (검출 xyz = 마스크 중앙값 하나). merge_lines "mask_points": true →
  구성원 근처 69 마스크(logs/scene_landmarks_masks) 픽셀을 snaps c2w(640x480, hfov 68.7)로 메쉬에 쏜 점, 줄 면 ±0.5 m 만.
  결과 B2 북 -0.1~1.8 m · 남 -0.3~2.2 m (메쉬 실측 스크린도어 마스크 95% 높이 1.5~1.9). B1 미닫이 2 개는 마스크 점 적어 0.7 m.
- corrections "goal_exclude_types": ["arrow"] → 73 에서만 뺌 (.house 에 arrow 5 개 남음). 목적지 384.
- 백업 corrections.json.bak_*_before_arrow_maskpts.

## 2026-11-13 카메라 위치 옮기기 (로봇은 그대로)
- teleop_view: shared cam_off [앞, 왼쪽, 위] m (범위 제한 없음 — 사용자 결정), POST /cam {d:[..]} 또는 {reset:true}. c2w 위치에만 더함 (앞·왼쪽 = 로봇 yaw 기준 수평, 위 = 세계). /state pose 에 cam_off.
  C 에 반영되므로 되찍기·분할·3D 광선이 옮긴 카메라 그대로. /visible 에 cx,cy,cz (카메라 자리) 추가.
- 페이지: ↑/↓ 앞뒤 · ←/→ 좌우 0.3 m · PgUp/PgDn 높이 0.2 m · C 되돌리기, 조종 칸 버튼. 지도 부채꼴은 카메라 자리에서.
- 시험: 위 1 m · 앞 0.6 m → 카메라 (−0.58, 0.19, 1.01) 이동, 로봇 그대로, 되돌리기 후 원위치.

## 2026-11-13 다른 이름 · 공통 class 채움 (사용자 라벨링 대강 끝)
- 표 /tmp 에서 만든 40 종류: 종류별 corrections aliases·common + 기존 alias_ids 125 개에 더하기만 (사용자가 쓴 것은 그대로, 'air' 같은 잘린 이름만 뺌).
  공통 class 묶음: door · seat · cone · board · gate · cabinet · plant · poster · banner · symbol · fire equipment · bin · machine 등.
- 오타 'eemergency supply cabinet' -> 'emergency supply cabinet'. wide fare gate 공통 'fare gate' 는 사용자 값이라 둠.
- stair_ends.inject 가 corrections 의 'stairs' 다른 이름·공통 class 를 끝 랜드마크에 붙이게.
- 빈 물체 0. 73 다시 (공통 class 가 '같은 종류' 기준이라 결과 바뀜). 백업 corrections.json.bak_*_before_names.

## 2026-11-13 5090 마무리 · a6000 인계 (다음 단계부터 a6000)
현재: .house 들어갈 랜드마크 204 (69 130 · 화면 추가 54 · 계단 끝 17 · 스크린도어 줄 3) / 39 종류. 목적지 랜드마크 129 · 목적지 384.
다음 단계:
1. .house 최종 — 70 --corrections logs/landmark_final/corrections.json (아직 안 만듦; 시험본 logs/_stairs_test/house).
2. 73 목적지 — logs/goal_candidates_final 최신 (384). 학습 1,200 + 평가 80 → 목적지당 경로 약 3~4.
3. 74 경로 — 작게 시험 → 고칠 것 2: (a) 같은 랜드마크 평가 경로끼리 출발점 겹침, (b) 다듬은 경로 끝점이 계단에 걸려 버려짐(GOAL_ON_STAIRS).
   계단 끝 목적지(part stairs_top/bottom) · 스크린도어 줄(긴 물체 양 끝) · arrow 목적지 제외 확인. 전체는 물어보고.
4. (GPU) Gemma 문장 → 학습 400/1200 → 평가 80 — 시작 전 보고.
옮길 것: 코드(커밋 안 됨 — 70·71·73·74·69·corrections.py 수정, goal_rules · stair_ends · merge_lines · tools/noeun_* 새 파일),
  데이터(logs 는 git 무시): logs/landmark_final (corrections.json!) · logs/scene_landmarks · logs/scene_landmarks_masks · logs/goal_candidates_final · logs/scene_scan (≈75 MB).
