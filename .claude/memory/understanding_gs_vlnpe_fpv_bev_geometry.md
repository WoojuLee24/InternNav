# gs_vlnpe 기하 레퍼런스 (현재 상태)

`scripts/dataset_converters/gs_vlnpe/geometry_utils.py` 기준. **04/06 등 이후 스크립트는 이 모듈을
import해서 쓸 것** — 직접 재구현하면 아래 "과거 오진" 표의 실수를 반복한다.

## 1. 좌표 규약

| 기호 | 의미 |
|---|---|
| `A_t` | parquet `action[t]` (4,4) |
| `F` | `CAM_CV_TO_GL = diag(1,−1,−1,1)`, `F = F⁻¹` |
| `P_t` | 진짜 camera-to-world = `A_t · F` |

`unproject_to_camera_frame`은 **OpenCV** 프레임(x=right, y=down, z=forward) 점을 만드는데
`action[t]`의 회전은 **OpenGL/USD** 프레임(y=up, z=backward) 기준이라 `F`가 필요하다.

```python
action_to_c2w(pose, 'cam2world_gl')  #  pose @ CAM_CV_TO_GL   ← 정답
action_to_c2w(pose, 'cam2world')     #  pose                  (기각, mesh 0.42 m)
action_to_c2w(pose, 'world2cam')     #  inv(pose)             (기각, mesh 0.96 m)
```

**`action`을 직접 행렬 연산에 쓰지 말 것.** 반드시 `action_to_c2w`를 거친다.

## 2. `camera_extrinsic` — 에피소드별 로봇 파라미터

에피소드 **내**에서는 상수지만 **에피소드마다 다르다**. 0행이 `[1,0,0]`(순수 x축 회전)이라:

```python
h_b, pitch_down = decompose_camera_extrinsic(E)   # E[2,3],  90° − atan2(E[2,1], E[2,2])
E               = compose_camera_extrinsic(h_b, pitch_down)
floor_z         = cam_z − h_b                      # mesh 추정 불필요
```

| | 실측 (347 ep / 4개 씬) | NavDP 논문 §3 |
|---|---|---|
| `h_b` (로봇 키) | 0.251 ~ 1.493 (mean 0.871, 균등) | (0.25, 1.25) — 실제 상한 1.5 |
| pitch_down | 0.00 ~ 30.00° | (−30°, 0°) |
| `floor_z` | 씬 내 상수 (std 2.4e−08) | — |

03은 `h_b`로 통과 가능 높이를 정하고, 04는 카메라를 배치하며, 05는 이 필드를 정답지와 1:1 대조한다.
씬별 분포는 `target_schema.json`의 `robot_params`.

## 3. 정합 확인 3종

### 상대 검증 2개 — 같은 상대 변환, 그리는 격자만 다름

앞 8개 인자(`rgb_t0, depth_t0, rgb_t1, depth_t1, pose_t0, pose_t1, k, convention`)와 반환 키
(`metric_name`/`metric_value`/`coverage_frac`/`viz_paths`)가 동일하다.

```python
# 공통: cam(t0) -> cam(t1)
t_rel = compute_relative_transform(pose_t0, pose_t1, convention)   # inv(c2w(t1)) @ c2w(t0)

check_alignment_fpv(...)  # t1의 색을 t0의 카메라 픽셀 격자에 backward-warp (정면 시점)
check_alignment_bev(...)  # 같은 변환 후 top-down 격자에 rasterize (위에서 본 시점)
```

- **FPV 비교 대상은 항상 `rgb_t0`** — 출력이 t0 격자라서. `rgb_t1`과 비교하면 다른 시점을 비교하는 셈.
- BEV는 앵커(t1)의 pose를 두 점군에 똑같이 적용해 격자를 중력 정렬한다(비교 결과에는 영향 없음).
- 무효 픽셀은 검정(0)이 아니라 `mark_invalid_as_checkerboard`로 회색 체크무늬.

### 절대 검증 1개 — `check_against_scene_mesh`

`vln_n1`의 depth는 씬 mesh를 렌더한 것이므로, world로 올린 점의 **mesh 표면거리**가 절대 판정이 된다.

```python
world_pts = unproject_to_world_frame(depth_m, k, action[t], convention)[valid]
dist      = compute_mesh_distance(world_pts, mesh)      # trimesh ProximityQuery
passed    = median(dist) < MESH_ANCHOR_TOL_M            # 1e-3
```

mesh 경로: `find_scene_mesh` → `<mesh_root>/<scene>/matterport_mesh/<hash>/<hash>.obj`
(해시 폴더명이 씬마다 달라 glob, `isaacsim_*` 변환 사본은 제외). `import trimesh`가 함수 안에 있는
이유는 trimesh 없는 환경에서도 모듈 import가 되게 하려는 것.

### 검출 능력 (이게 셋 다 필요한 이유)

| 오류 종류 | FPV | BEV | mesh |
|---|---|---|---|
| 상대 자세 오류 | O | O | O |
| 축 컨벤션 오류 | 조합에 따라 상쇄 | 조합에 따라 상쇄 | **O** |
| **전역 강체변환** | **X 원리적 불가** | **X 원리적 불가** | **O** |

관측끼리 비교하는 한 전역 강체변환은 검출 불가 — 좌표계 전체가 틀어져도 모든 관측이 똑같이 틀어져
서로는 맞기 때문이다. FPV는 수학적으로 정확히 무감각하다:
`inv(G·A₁F)·(G·A₀F) = F·A₁⁻¹·G⁻¹G·A₀·F = F·A₁⁻¹A₀·F` (G 완전 소거).
실측: 모든 pose에 30° 회전을 씌워도 BEV err 1.88→2.08 불변, mesh 거리 0.0000→0.2199 m.

**02/03이 mesh 좌표계에서 경로를 만들고 04가 절대 pose를 기록하므로 mesh 검사는 필수다.**

## 4. 회귀 게이트 — `python geometry_utils.py`

```
[1/5] identity self-transform      [2/5] CAM_CV_TO_GL involution
[3/5] camera_extrinsic 왕복        [4/5] mesh anchor < 1mm
[5/5] GT camera_extrinsic pitch 대조 (fwd_z == −sin(pitch))
```

pose 규약이나 분해식을 건드리면 여기서 먼저 깨진다.

## 5. 현재 수치 (`17DRP5sb8fy` ep0)

```
detection(gap=15)  cam2world_gl  err=2.076  mesh_dist=0.000029 m   ← 채택
                   cam2world     err=41.606 mesh_dist=0.418680 m
                   world2cam     err=46.754 mesh_dist=0.957407 m
visual(gap=3)      FPV err=2.347  BEV err=2.646 (coverage 0.178)
```

`s8pcmisQ38h`도 `cam2world_gl` mesh_dist=0.000028 m로 동일 결론.

## 6. 잔차 해석 — err가 0이 아닌 이유

규약이 맞아도 gap=15에서 err≈2가 남는다. **기하 오차가 아니라 JPEG 노이즈 + 가려짐**이다.

| 픽셀 집합 (pair 56→71) | mean | median |
|---|---|---|
| 전체 valid | 1.30 | 1.00 |
| **가려진 픽셀만 (0.4%)** | **24.44** | 20.00 |
| 안 가려짐 + depth 평탄부 | 0.97 | 1.00 |

median이 모든 gap에서 정확히 **1.00**(= JPEG 양자화 1단계)이다. mean은 상위 5% 픽셀이 32%를 만드는
heavy-tail이라 나쁜 요약치다.

가려짐 판정: t0의 점을 t1로 투영한 예측 depth `z1`을 실제 `depth_t1`과 비교해 `z1 > depth_t1 + 0.10`.

> **`err=2.08`은 "2픽셀 어긋남"이 아니다.** 기하 정확성의 근거는 err가 아니라 mesh 표면거리
> (0.00003 m)와 `matchTemplate` offset(0px)이다.
> **06에서 재투영 임계를 정할 때**: mean이 아니라 median을 쓰거나 가려짐 픽셀을 먼저 제외할 것.

---

## 과거 오진 목록 (같은 함정 재발 방지용)

| # | 당시 진단 | 실제 원인 | 왜 못 잡았나 |
|---|---|---|---|
| 1 | `action`을 그대로 c2w로 사용 | `CAM_CV_TO_GL` flip 누락 | photometric error(상대 지표)로만 판별 — 축 오류가 상쇄됨 |
| 2 | "`compute_relative_transform`의 t0/t1 순서 버그" → `inv(t0)@t1`로 수정 | flip 누락을 부분 상쇄한 것뿐. 교과서 공식 `inv(t1)@t0`가 맞음 | 좁은 baseline(gap=3)에서 정답/오답이 err 1.93 vs 1.91로 구분 불가 |
| 3 | "프레임 독립 unproject하면 정렬 안 됨(0.24 m)" → BEV에 anchor 우회 추가 | 같은 flip 누락. 우회는 증상만 가림 | 관측끼리 비교라 절대 오류를 볼 수 없음 |
| 4 | "`camera_extrinsic` = 고정 마운트 오프셋" | 에피소드마다 다름 (= `h_b` + pitch) | 00이 **한 에피소드만** 읽는 설계 |
| 5 | 01의 `floor_z`를 mesh 정점에서 추정 | `cam_z − h_b`가 정확 (오차 0.06~0.40 m) | 러그·문턱·가구를 바닥으로 주움 |
| 6 | 01의 스케일 검증을 "층고 2~4 m"로 | 다층 건물이 섞여 있음 (Z extent 12.24 m) | 씬 1개로만 검증 |

**일반 원리 3가지**

1. **좌표 규약을 상대 지표로 확정하지 말 것** — photometric error·matchTemplate은 두 프레임에
   똑같이 걸린 축 오류를 상쇄한다. 절대 앵커(mesh, GT 포인트클라우드)에 대고 재야 자릿수로 갈린다.
   이 프로젝트에서 그 mesh는 세 세션 내내 로컬에 있었는데 아무도 쓰지 않았다.
2. **"에피소드 내 상수"를 "데이터셋 상수"로 확대하지 말 것** — 한 에피소드만 읽는 스크립트는
   구조적으로 이 구분을 못 한다.
3. **씬 1개에서 잡은 임계값을 일반화하지 말 것** — 2번째 씬에서 두 번(#5, #6) 깨졌다.

## 관련 문서

- `docs/execution-staged.md` — 전체 파이프라인 설계
- `target_schema.json` — 파이프라인 canonical 스키마 (`pose_convention`, `robot_params`)
- 시각화 비교: https://claude.ai/code/artifact/6446f9e3-a51d-479b-a80e-bc67abffb08b
