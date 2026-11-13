# 검증 스크립트 (한 번씩 돌려 사실을 확인한 것들)

본 파이프라인에 들어가지는 않지만, **설계 판단의 근거가 된 측정**이라 재현할 수 있게 남긴다.

| 파일 | 무엇을 확인했나 | 결과 |
|---|---|---|
| `cmp2.py` | 메시 광선 vs Isaac depth | 80 점 중 96 % 가 0.3 m 안 · 상관 0.90. Isaac 은 `max_depth=10 m` 에서 잘린다 |
| `cmp_depth_mesh.py` | 같은 ep·step 으로 대조 (초판) | step 이 두 렌더에서 안 맞아 `cmp2.py` 로 바꿈 |
| `cmp_pose.py` | 기존 렌더 vs 고속 렌더의 카메라 자세 | 위치 중앙 0.82 m · 방향 중앙 4.8° 차이 — **고속 렌더가 목적지에 정확**(0.01 m) |
| `coord_probe.py` | Gemma 가 쓰는 좌표계 | **1000 기준**("range: 1000x750"). 640 으로 읽으면 36 % 가 화면 밖 -> 비율(0~1)로 받아야 함 |
| `fair_cmp.py` | Gemma 점 vs DINO+SAM (같은 메시 광선) | 좌표 고친 뒤 차이 중앙 **0.09 m** · 22 중 19 가 0.5 m 안 · 상관 0.983 |
| `gemma_dist_test.py` | Gemma 의 **거리 추정** | 절대 오차 중앙 0.76 m — 사진만 보고 어림. 기하 계산을 못 이긴다 |
| `vocab_probe.py` | Gemma 가 사진에서 이름 뽑기 | 22 장 -> 75 종류. 내 추측 어휘에 없던 `platform screen doors` 등 발견 |
| `vocab_verify.py` | 그 이름을 DINO 가 찾는가 | 75 중 확실 64 · 약함 9 · 안됨 2 (`trash bin`·`light fixture`) |
| `viz38.py` | 검출 상자 눈으로 보기 | tiny 는 기둥을 `trash` 로 오인 -> **base 로 교체** |
| `floor_outline.py` | 층별 걸을 수 있는 영역 | 편집 도구 평면도 배경 |
| `spots_test.py` | 격자 간격별 설 자리 수 | 3 m 432 · 5 m 167 · 8 m 68 |

모두 habitat 컨테이너에서 돈다: `scripts/apptainer/noeun_apptainer.sh habitat python3 <파일>`
(`gemma_*`·`fair_cmp` 는 `vlm_server.py` 가 :8200 에 떠 있어야 한다)
