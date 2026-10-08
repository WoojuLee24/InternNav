# #03 재작성 — "벽 뚫기"는 데이터셋의 지도에서 재현되지 않는다

2026-08-20 · `03_verify_navmesh_gt.py` · 리포트 `logs/embodiment_augment/s1_navmesh/report.html` ·
[Artifact](https://claude.ai/code/artifact/330c32e4-4b25-4f51-96c6-5fc54f1756a0)

## 왜 다시 썼나

사용자: "너무 복잡하고 이해가 안 된다." 실제로 447줄에 게이트가 8개(A/B/C/D/D2/E/F/G)였고 절반은
게이트가 아니라 측정값이었으며, **헤드라인 결론이 이미 한 번 철회**된 상태였다
([[260820_clearance_gt_vs_findpath_result]]).

복잡함의 주범은 **`clearance` 분석 한 덩어리**였다. navmesh는 이미 `agent_radius` 0.10만큼 깎인
configuration space라 물어야 할 것은 "칸 안이냐"뿐이고, 거기에 `clearance ≥ r_b`를 또 요구하면
**반경을 두 번 센다**(`navmesh_grid.esdf_from_mask`의 `+ r_b` 트릭이 그 증거). **통째로 삭제했다.**

## 무엇이 달라졌나

| | 이전 | 이후 |
|---|---|---|
| 줄 수 | 447 | **593** ⚠️ 안 줄었다 (아래) |
| 게이트 | 8 (A/B/C/D/D2/E/F/G) | **4** (A 파라미터 / B 벽뚫기 / C leg 도달 / D 캐시) |
| 경로 | GT · find_path (+ 흰색·주황) | ⓐ 원본 정답(노랑) · ⓑ waypoint 연결(주황) + **위반점 빨강** |
| "뚫기" 판정 | `clearance ≥ r_b` (반경 이중 계산) | `is_navigable` + **깊이 1칸 술어** |
| 리포트 파일 | 갤러리 전체를 `summary.html`로 (발행기와 이름 충돌) | `report.html` + `summary.html`·`body.html` 조각 (`02` 규약) |
| 의존성 | habitat만 | 동일 (우리 플래너·우리 맵은 범위 밖 → #04/#05) |

⚠️ **줄 수는 줄지 않았다** (447 → 593: docstring 50 · 리포트 HTML 191 · 분석 코드 352).
`clearance`를 지운 만큼 3중 판정(`judge` / `off_depth` / 깊이 술어)과 skip 버킷이 늘었다.
참고로 같은 스타일의 `00`은 217줄, `02`는 264줄이다 — 03이 여전히 가장 크다.
**줄어든 것은 읽어야 할 개념 수다**: 게이트 절반, 철회된 분석 삭제, 분포 표 하나를 반증 가능한 술어로 대체.

## 실행 명령

```bash
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/03_verify_navmesh_gt.py --scenes 17DRP5sb8fy,s8pcmisQ38h --r_bs 0.10,0.15,0.20,0.30,0.45 --episodes 14 --out_dir logs/embodiment_augment/s1_navmesh
```

| 인자 | 의미 |
|---|---|
| `--scenes` | 집(scan) 목록 |
| `--r_bs` | 시험할 로봇 반경 목록[m]. **`0.10`이 원본 데이터셋 값** |
| `--episodes` | 집당 에피소드 수 |
| `--mpp` | topdown 래스터 해상도[m], 기본 0.05 — 우리 격자와 같게 |
| `--max_z_spread` | GT 높이 변동이 이보다 크면 **계단으로 보고 제외** (기본 0.30, `05`와 같은 값) |
| `--navmesh_cache` | `r_b`별 navmesh 저장 위치. **#04의 default 맵이자 #05 W6 오라클이 읽는다** |

발행: `/usr/bin/python scripts/dataset_converters/3dloader_vlnce/publish_all.py --only s1_navmesh`

## 결과 — 게이트 (두 씬 전부 통과)

| 게이트 | 17DRP5sb8fy | s8pcmisQ38h |
|---|---|---|
| A 파라미터 | 불일치 **없음** | 불일치 **없음** |
| B 벽 뚫기 (5 cm로 채운 점) | ⓐ **100.00%** · ⓑ **100.00%** | ⓐ **99.43%** · ⓑ **100.00%** |
| C leg `find_path` ≤0.5 m | **69/69** | **47/47** |
| D `recompute_navmesh` 재현 | 52.04 vs 52.03 (+0.02%) | 184.01 vs 183.73 (−0.15%) |

waypoint 자체도 **474/474 · 531/531** navigable · `snap_point` 이동 **0.0000 m**.
s8은 14 에피소드 중 **5개가 계단**으로 제외됐다(높이 변동 1.0~2.1 m).

## 결과 — 원래 증상의 답

**"벽을 뚫는" 문제는 데이터셋의 지도에서 재현되지 않는다.**

| 씬 | ⓐ 점 수 | J1 밖 (진짜 뚫음) | 그 점의 snap 거리 max | J1통과·J2밖 | 깊이 1칸 | 2칸 | ≥3칸 |
|---|---|---|---|---|---|---|---|
| 17DRP | 2465 | **0** (0.00%) | 0.0000 m | **0**/2465 | 0 | 0 | 0 |
| s8 | 2450 | **14** (0.57%) | **0.0371 m** | **58**/2436 | **58** | **0** | **0** |

- **J1** = `pf.is_navigable(p, 0.5)` — habitat 직접 답 = 진실
- **J2** = `pf.get_topdown_view(0.05, fz)` 래스터 — **우리가 그리고 계획할 때 보는 것**
  (`navmesh_grid.load_mask`가 이걸 샘플링한다)

**반증 가능한 술어로 냈다**: "J1 통과·J2 밖인 점이 전부 깊이 1칸이면 원인은 격자 반올림." 2칸 이상이
하나라도 나오면 반증인데 **나오지 않았다**. J1 밖인 14점도 **최대 3.7 cm = 한 칸(5 cm) 미만**이라
관통이 아니라 경계에 걸친 이산화 오차다. #04가 "66/8459점, 최대 깊이 정확히 1셀"로 통계로 냈던 것을
**참/거짓 술어**로 바꾼 것이다.

→ 원 증상("우리 맵에서 GT가 `clearance ≥ r_b` 위반, r_b=0.2에서 4/7")은 **우리 쪽 문제**였다.
그 숫자는 `Z_OFFSET_M = 0.20` 정렬 버그 + 데이터셋이 쓴 적 없는 `r_b = 0.2`에서 나왔고, 버그 정정 후
#05 W5 identity는 69/69 · 47/47 PASS · GT 거리 0.00 cm다.

## 시각화가 반증한 주장 — **길이비 ≈ 1은 "같은 경로"의 증거가 아니다**

| 씬 | 에피소드 | 대응점 거리 median | max | 길이비 (ⓑ/ⓐ) |
|---|---|---|---|---|
| 17DRP | 14 | **12.9 cm** | 112.6 cm | **1.001** |
| s8 | 9 | **56.4 cm** | 195.9 cm | **1.002** |

리포트 초안에 "길이비 ≈ 1이면 같은 경로이고 남는 차이는 코너 절단"이라고 썼다. **그림을 보고 틀린 것을
발견했다** — `s8pcmisQ38h_ep008.jpg`에서 노랑(GT)과 주황(find_path)이 어두운 섬을 **위/아래로 갈라져**
지나간다. 두 우회로 길이가 비슷하면 측지 최단선이 GT와 **다른 쪽**으로 도는 것이고, 그때도 길이비는 1.002다.

즉 어긋남에 **두 종류**가 섞여 있고 길이비는 둘을 구분하지 못한다:
1. **코너 절단·궤적 여유** — GT는 `0.25 m/15°` 이산 액션으로 넓게 돌고 `find_path`는 자른다 (17DRP, 12.9 cm)
2. **장애물의 반대편으로 갈라짐** — (s8, 56.4 cm, 길이비 1.002)

→ **회랑 중심선은 `find_path`가 아니라 GT 서브궤적이어야 한다**(현 `follow_waypoints`가 이미 그렇게 한다).
반대편으로 갈라지는 경우가 있으니 **선택이 아니라 필수**다.

**이 세션에서 세 번째로 반복된 패턴이다**: 요약 통계 하나가 실체를 가렸다.
(1회 `clearance min`을 전형값으로 → 2회 평균이 20 cm z-offset을 가림 → 3회 길이비가 경로 갈라짐을 가림)

## 같이 고친 실제 버그 2건

1. **`chain_find_path`가 가짜 벽 뚫기를 만들었다.** 실패한 leg를 `continue`로 버리고 남은 조각을
   `vstack`해서, 그려진 주황 선이 **떨어진 두 구간 사이를 직선으로 건너뛰며 벽을 통과**했다. 게이트 B가
   그 가짜 직선에서 실패한다. 또 `reach`(L개)와 `pieces`가 어긋났다.
   → 기본값 있는 optional 인자 **`per_leg=False`** 추가. `True`면 `[array|None] * L` 반환.
   기존 호출부(`05_validate_waypoints.py:61`은 `[1]`만 씀)는 **무수정**.
2. **`summary.html` 이름 충돌.** 옛 03은 **갤러리 문서 전체**를 `summary.html`로 썼는데
   `publish_artifact_report.py:198`은 그 파일을 **카드 조각**으로 인라인한다.
   → `02` 규약대로 `report.html`(문서) + `summary.html`·`body.html`(조각)로 분리.
3. **발행본에 모순되는 범례 2개.** `publish_all.py`의 `legend='floorplan,path'`에서 공용 `path` 범례가
   "**흰**=원본 GT · **주황**=우회 지점"인데 #03은 "**노랑**=원본 GT · **주황**=waypoint 연결"이다.
   발행된 페이지에 두 범례가 같이 실려 있었다(발행본을 WebFetch로 열어보고 발견 — 로컬 `report.html`만
   보면 안 보인다). → #03의 키를 **`'floorplan'`**으로 축소. 공용 `path` 텍스트는 #05 등 6개 stage가
   공유하므로 **수정하지 않았다**.

## 하류 계약 (깨면 죽는다 — 유지 확인됨)

- 모듈 최상단이 **habitat·open3d 없이 import** 가능. `02:51` · `04:51` · `05:37`이 5개 심볼을 가져간다:
  `sf2mesh` · `mesh_to_habitat_floor` · `load_poses` · `topdown` · `chain_find_path`. 시그니처 동일
  (`chain_find_path`만 기본값 있는 인자 추가).
- `<navmesh_cache>/<scene>_rb{r:.2f}.navmesh` 계속 기록 — `navmesh_grid.load_mask:58`이 assert 의존.

## 검증 (전부 통과)

| 검증 | 결과 |
|---|---|
| import 안전성 (habitat·open3d 미로드 + 심볼 5개) | ✅ |
| `navmesh_grid.py` self-check | ✅ 7/7 |
| `04_calibrate_map.py --selfcheck` | ✅ 8/8 |
| `03` 본체 2씬 × 14 에피소드 | ✅ A/B/C/D × 2씬 |
| `02_verify_depth_sources.py` 회귀 (03에서 `sf2mesh` import) | ✅ median 0.02 mm |
| `05_validate_waypoints.py --map_source navmesh` 회귀 (03의 캐시 읽음) | ✅ W5 identity 10/10 PASS · W6 거짓승인 0 |
| `publish_all.py --only s1_navmesh` | ✅ artifact 1.4 MB · 그림 33장 인라인 |

## 고친 산문

`reports.md`(색인 행 L29 + §#03 전체 + #04의 L260 상호참조) · `publish_all.py`(`REPORTS['s1_navmesh']`
7-tuple 전부) · `README.md`(표 행 · 명령 · "확인된 사실" 3행) · `command_embodiment_augment.md`(L77) ·
`04_calibrate_map.py` docstring(철회된 주장이 #04의 **동기**로 적혀 있었다 — 코드는 무수정) ·
[[260819_map_align_rb_ladder_result]] · [[260820_clearance_gt_vs_findpath_result]]

## 남은 것

- **s8의 J1 밖 14점을 개별로 보지 않았다.** 최대 3.7 cm라 한 칸 미만이지만, `Δy`(층 오차)인지
  `Δxz`(평면 오차)인지 분해하지 않았다.
- 씬 2채 · s8은 계단으로 5/14 제외. **다층 에피소드는 이 리포트가 아예 다루지 못한다** —
  `get_topdown_view`가 단일 높이 절단이기 때문이고, 층별로 잘라 합치는 것은 미구현이다.
- `is_navigable`의 **수평 1 cm · 수직 0.5 m 맹점**. 게이트 B는 "지도를 벗어나지 않는다"가 아니라
  "평면상 navmesh 폴리곤 안, ±1 cm"다.
- 게이트 D는 자기가 **쓰는** `r_b > 0.10` 지도가 옳은지 검증하지 않는다 — #05 W6의 일.
- **우리 플래너(`follow_waypoints`)를 이 지도 위에서 돌려보지 않았다.** 사용자가 범위에서 제외했다.
  "우리 경로가 원본과 일치하는가"의 플래너 판은 여전히 #05 W5(occ 맵)에만 있다.
