# 2dloader_vlnce — 리포트 목록

시야각 내 RGB/depth 한 장만으로 GT path와 pixel goal을 다시 만드는 파이프라인의 검증 리포트.
게이트는 **W0 → W6** 순서로 읽는다. 각 리포트 맨 앞에 "이 리포트가 검증하는 것"(무엇을/왜/통과 기준)이 있다.

- 로컬: `logs/embodiment_augment2d/<stage>/report.html` (브라우저에서 바로 열기)
- 발행본: 아래 Artifact 주소 (이미지가 base64로 인라인된 self-contained HTML)
- 재생성 명령: [`command_2dloader.md`](command_2dloader.md)
- 상세 결과·설계 근거: `.claude/memory/260817_2dloader_vlnce_result.md`

기준 씬은 **17DRP5sb8fy ep0**, 일반화 확인 씬은 **s8pcmisQ38h ep0**. 둘 다 preset `125cm_0_30`
(FPV = pitch_1 0°, 룩다운 = pitch_2 30°), 프레임 6개(W6은 4개), `r_b ∈ {0.10, 0.20, 0.35, 0.50}` m.

---

## 주 리포트 (17DRP5sb8fy ep0)

| # | 리포트 | 무엇을 확인하나 | 결과 | Artifact |
|---|---|---|---|---|
| 1 | **W0 · 좌표 규약·항등** | depth 1장으로 만든 robot-centric 격자가 (A) 중력 정렬, (B) **학습 GT 궤적과 같은 좌표계**, (C) 저장된 pixel goal 재현, (D) 원본 GT가 점유칸을 안 밟음 | **4/4 PASS**<br>중력 4.0e-5 · 학습프레임 **3.6e-8 m** · goal ≤ 2.25 px · 점유칸 0 | https://claude.ai/code/artifact/45f4346b-a5bc-474d-bfe6-a2a671fdabd1 |
| 2 | **W1 · 2D vs 3D 오라클** | depth 1장 맵과 씬 mesh 오라클을 겹쳐 본 곳의 일치도(IoU)와 **못 본 장애물 비율** | 측정용(임계 없음)<br>IoU **0.502** · 미관측 **85.4%** | https://claude.ai/code/artifact/b685b1e2-757f-4678-a62b-9768a4857c0a |
| 3 | **W2 · 장애물 합성 정합** | 3D 박스 하나가 FPV RGB·룩다운 RGB·depth·BEV **네 곳에 같은 자리로** 들어가는가 + 합성 후 재계획 경로 | **4/4 PASS**<br>footprint 안 100% · 표면 0.00 mm · 교차 rig ≥ 99.8% | https://claude.ai/code/artifact/a41460be-6a99-4021-987f-0b991433655d |
| 4 | **W3 · e가 GT path를 바꾸는가** | `r_b`만 바꿔 원본 GT 통행성이 뒤집히는지 + **A\*가 실제로 쓴 17.9 cm 격자**를 r_b별로 표시 | **2/2 PASS**<br>단조 6/6 · 통행 불가 12/24 · 이탈>10 cm 2/24 | https://claude.ai/code/artifact/a73835fb-8974-4b03-bcea-5b839c5d9e54 |
| 5 | **W4 · pixel goal 라벨 갱신** | S2 학습 라벨이 `e`를 따라가는가, 그리고 **바뀌면 안 될 때 안 바뀌는가** (mesh 미사용) | **3/3 PASS** (V0·V2·V3)<br>unchanged 15 / adjusted 2 / 기각 7 | https://claude.ai/code/artifact/a7871393-1a70-4b9f-8107-96836878bc5e |
| 6 | **W5 · on-the-fly 예산** | 프레임당 비용 단계별 분해 + **torch 스레드 절벽** 원인 규명 | **FAIL**<br>20.7 ms (목표 20 ms) · 이미지 IO 7.8 ms의 2.7배 | https://claude.ai/code/artifact/b9af0fb2-4f26-468e-a8d9-f20680f17b9e |
| 7 | **W6 · paired counterfactual** | 같은 관측 1장이 `e`별로 **통과 / 우회 / 기각**으로 갈리는가 | 정성 · 유효 11/16 | https://claude.ai/code/artifact/4f372536-f207-4bd0-9f5f-d5ccb125048f |

## 일반화 확인 (s8pcmisQ38h ep0)

| 리포트 | 결과 | Artifact |
|---|---|---|
| W0 — s8pcmisQ38h | 4/4 PASS | https://claude.ai/code/artifact/ab8cc961-7325-41e9-a2f1-278878606603 |
| W1 — s8pcmisQ38h | IoU 0.365 · 미관측 95.9% | https://claude.ai/code/artifact/05151f7a-6820-447f-ba39-d20b3f1f5823 |
| W2 — s8pcmisQ38h | 4/4 PASS | https://claude.ai/code/artifact/ed1e9b03-1653-49da-a3b7-488a31ad5957 |
| W3 — s8pcmisQ38h | 2/2 PASS · 통행 불가 11/24 · 이탈 0 (넓은 복도라 **기각**으로 e 효과가 나타남) | https://claude.ai/code/artifact/6d074fdc-8ac3-4d9d-b3ea-eb430e20c11c |
| W4 — s8pcmisQ38h | 3/3 PASS · unchanged 14 / adjusted 3 / 기각 7 | https://claude.ai/code/artifact/f8a2b081-bb98-4c66-9af1-10b1d2c62c4e |
| W6 — s8pcmisQ38h | 유효 10/16 | https://claude.ai/code/artifact/aee91df0-5a65-448d-a250-69a6fda4aa0b |

## 관련 리포트 (3dloader_vlnce)

| 리포트 | 내용 | Artifact |
|---|---|---|
| **R2 속도 정정** | 기존 "depth→BEV 165 ms/frame"이 **torch 스레드 설정**(스레드 수 == 코어 수) 문제였음을 재측정으로 확인 — 1스레드에서 1.8 ms/frame (100배). `with_bev=False`는 유지하되 이유가 "느려서"가 아니라 "중복이라서"로 바뀜 | https://claude.ai/code/artifact/55795bd9-9db1-4ad8-b08d-9a558bc7c800 |

---

## 그림에 공통으로 쓰는 색

리포트마다 범례 표가 들어 있지만, 지도 색은 전부 아래와 같다.

| 색 | 뜻 |
|---|---|
| 회색 | **A\*가 실제로 쓸 수 있었던 칸** (17.9 cm 조립 격자 한 칸 = 4×4 확대라 네모가 굵다) |
| 어두운 빨강 | 비어 있다고 관측됐지만 **이 로봇에겐 여유가 부족하거나** 조립 칸에 미관측이 섞여 제외된 곳 — `r_b`가 커질수록 넓어진다 |
| 빨강 | 점유 = 높이 밴드 `(h_nav, h_b]`에 depth 반환이 있는 칸. `r_b`와 무관 |
| 검정 | 미관측 |
| 노랑 | 원본 GT path |
| 초록 / RB 색 | 재계획 path |
| 자홍 | pixel goal (빈 원 = 저장된 원본, 채운 원 = 갱신) |
| 주황 | 합성 장애물 밑면 |
| 흰 점 | 로봇 원점 (지도 중심, 전방이 위쪽) |

## 알아둘 것

- **mesh를 쓰는 리포트는 W1 하나뿐**이고, 거기서도 mesh는 *비교 대상 오라클*일 뿐 파이프라인 입력이 아니다.
  W0·W2~W6은 그 프레임 depth 한 장으로만 돈다.
- 지도의 기준 카메라는 **룩다운(pitch_2)** 이다 — pose·depth·pixel goal 라벨·BEV가 모두 이 rig 기준.
  FPV(pitch_1)는 System 2가 보는 이미지이고 여기엔 장애물 합성만 한다.
- W5만 FAIL이다. 목표 20 ms는 스스로 정한 값이고, 실제로 문제는 augmentation(20.7 ms)이
  프레임당 이미지 읽기(7.8 ms)보다 커서 **IO에 묻히지 않는다**는 점이다. 남은 병목은
  바닥 free carving 3.3 ms + 장애물 합성 4.1 ms.
