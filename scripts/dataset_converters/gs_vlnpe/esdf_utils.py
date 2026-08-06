"""M1.2~M1.3 — mesh를 "어디로 갈 수 있는가" 맵(ESDF)으로 바꾸고 그 위에서 경로를 만드는 공용 모듈.

    00 : vln_n1 데이터 -> "어떻게 읽고 쓰나"  -> target_schema.json      (전체에 1개)
    01 : 씬 mesh/USD   -> "이 씬 써도 되나"    -> scene_meta/<scene>.json (씬마다 1개)
    이 모듈: mesh      -> "어디로 갈 수 있나"  -> 02가 esdf/<scene>.npz로 저장
             ESDF      -> "어떤 경로로 가나"   -> 03이 paths/<scene>.json으로 저장

`geometry_utils.py`가 카메라/재투영 기하를 담당한다면 이 모듈은 **주행 가능 영역과 경로**를 담당한다.
NavDP 논문 §3 Trajectory Generation 전체(1~7단계)를 구현하고, 재현도를 재는 지표와 논문 단계별
역검증까지 포함한다. 스크립트(02/03/03b/03c)는 파일 입출력·검증·리포트만 한다.

## 처리 단계별 데이터

```
mesh (삼각형)
  | sample_surface(3M) -> 격자 스캐터                     voxelize_surface()
(1) occupancy    (Nx, Ny, Nz) bool        <- h_b와 무관. 02가 npz로 저장하는 것은 여기까지
  | 밴드 (floor_z+h_nav, floor_z+h_obs] 에 점유가 있는 열  derive_obstacle_2d()   논문 2
(2) obstacle_2d  (Ny, Nx) bool
  | distance_transform_edt(~obstacle) * cell             compute_esdf_2d()      논문 3
(3) esdf         (Ny, Nx) float32 [m]
  | esdf >= r_b                                          truncate_navigable()   논문 4
(4) navigable    (Ny, Nx) bool            0.05 m
  | 4x4 블록 any()                                       downsample_navigable() 논문 5
(5) nav_coarse   (Ny/4, Nx/4) bool        0.2 m
  | 8-이웃 A*                                            astar()                논문 5
(6) waypoints    (N,2) 셀중심 -> world xy
  | 국소 greedy 탐색 (0.05 m 원본 ESDF)                   greedy_refine()        논문 6
(7) wp_refined   (N,2) world xy           <- r_b 제약은 여기에 걸린다
  | 호길이 0.8 m 간격 솎기                                thin_waypoints()       (논문에 없음)
  | cubic spline (또는 bezier)                           smooth_cubic_spline()  논문 7
(8) trajectory   (M,2) world xy
```

**(2)~(8)은 로봇 키 `h_b`에 의존하므로 파일로 저장하지 않는다.** 03이 에피소드마다 (1)에서
도출한다 — 328x166 격자에서 `distance_transform_edt`가 수 ms라 비용이 무의미하다.

## `h_nav` / `h_obs`가 무엇인가

로봇을 높이 `h_b`, 반경 `r_b`인 원통으로 보면 충돌 구간은 바닥 위 `(h_nav, h_obs]` 밴드다:

  - `h_nav` 아래 : 밟고 넘어가는 지면 (문턱, 러그, 작은 단차)
  - `(h_nav, h_obs]` : **충돌** — 이 구간에 뭐가 있으면 그 (x,y)는 장애물
  - `h_obs`(= `h_b`) 위 : 밑으로 통과 (천장, 높은 선반, 테이블 상판)

`h_obs`가 `h_b`에 의존하는 이유가 여기 있다 — 키 0.25 m 로봇은 의자 밑을 지나가지만 키 1.25 m
로봇은 못 지나간다(논문이 말하는 embodiment-aware planning). `h_b`는 `camera_extrinsic`에서
읽는다(`geometry_utils.decompose_camera_extrinsic`). 바닥 높이도 거기서 온다:
`floor_z = cam_z - h_b` (01이 계산해 `scene_meta.json`에 넣는다; mesh 추정 아님).

**글자 그대로 "h_obs 초과가 장애물"로 읽으면 안 된다** — 실내 천장 때문에 씬 전체가 장애물이
되어 경로 생성이 0/20이었다. 밴드 해석만 성립한다.

## 좌표 규약

- 3D occupancy는 `(Nx, Ny, Nz)`, 2D 파생물은 **`(Ny, Nx)`** (numpy 이미지 관례).
- 2D 인덱싱은 `arr[iy, ix]`이고 **y를 뒤집지 않는다**. 화면 표시용 뒤집기는
  `viz_utils.floorplan_canvas` 안에서만 한다.
- 격자 원점은 **월드 0.2 m 격자에 스냅**한다(`voxelize_surface(align_to_m=)`) — 아래 참고.

## 실측 기준값 (self-check 게이트가 쓰는 것)

`h_nav = 0.10`, `h_obs = h_b`, `r_b = 0.25`일 때 GT 궤적 clearance median이 **씬마다 0.46~0.68 m**
(6개 씬 실측, 넓은 씬일수록 큼), 최소 0.18~0.32. **단일 값으로 게이트를 잡으면 안 된다** —
`17DRP5sb8fy`의 0.500을 기준선으로 박았다가 `s8pcmisQ38h`(0.680)를 오판한 이력이 있다.

self-check: `python esdf_utils.py`  (게이트 11개)
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------------------------
# 상수 — 전부 NavDP 논문 §3 또는 실측 근거
#
# **원칙: 논문이 숫자로 준 값(voxel 0.05, A* 0.2)은 고정한다. 논문이 안 준 값(격자 원점,
# h_nav, r_b, refine 반경·해석, knot 간격·배치, 다운샘플 방식)만 조정 대상이다.**
# 이 원칙이 "저자가 하지 않았을 최적화"를 자동으로 배제한다. 실측으로 A* 격자를 0.1 m로 줄이면
# 모든 지표가 개선되고 실행시간도 같지만(이 규모에서 "efficiency" 근거가 무의미) 채택하지 않는다.
# ---------------------------------------------------------------------------

VOXEL_SIZE_M = 0.05          # 논문: "voxel size of 0.05m"
ASTAR_CELL_M = 0.20          # 논문: "downsampled to 0.2m resolution to facilitate efficient A*"
DOWNSAMPLE_FACTOR = int(round(ASTAR_CELL_M / VOXEL_SIZE_M))   # 4

# 지면 여유고 — "바닥 위 이 높이까지는 밟고 넘는 지면으로 본다".
#
# **0.15로 두면 GT path 재현이 깨진다.** 낮은 가구·문턱이 장애물에서 빠져 GT에 없는 지름길이 열린다.
# 실측(`17DRP5sb8fy` 20 에피소드, GT의 h_b/start/goal 복사해 재현):
#
#   h_nav       ep11   ep17   ep18   | 같은 루트 | chamfer
#   0.05~0.12   0.42   0.29   0.39   |   20/20   | 0.094
#   0.15        1.88   1.73   1.75   |   17/20   | 0.095
#
# 전환점이 0.12~0.15 사이라 작동 범위(0.05~0.12)의 중앙인 0.10을 쓴다.
# 논문은 "`h_nav`와 `h_obs`는 씬마다 다르고 로봇 키 `h_b`에 의존"이라고 한다. 비례형(0.15*h_b)도
# 20/20을 주지만 고정 0.10과 결과가 같아 기본은 고정으로 두고, 비례형은 `--h_nav_ratio`로 지원한다.
H_NAV_M = 0.10

ROBOT_RADIUS_M = 0.25        # 논문 r_b. InternNav `agent_radius` 기본값과 동일.
                             # 스윕 실측: 0.15는 하드충돌 88개, 0.30은 GT가 맵 밖으로 밀려남(4~6/20).

SURFACE_SAMPLES = 3_000_000  # 0.05 m 격자를 메우기에 충분. 씬당 0.7초.
                             # 3x3 closing이 navigable을 0셀도 안 바꾼다 = 얇은 장애물 구멍 없음.

# 스무딩 전 waypoint를 솎아낼 호길이 간격. A* 전점을 통과시키면 GT보다 2.8배 많이 회전한다 —
# 0.8 m에서 GT 스무드니스와 일치하고, 03b의 knot 복원이 GT에서 역추정한 간격(0.79~0.90 m)과도 맞다.
WAYPOINT_SPACING_M = 0.8

# GT 궤적 clearance 게이트 범위. **단일 값으로 잡으면 안 된다** — 씬마다 다르다(넓은 씬일수록 큼).
# 6개 씬 실측: median 0.461 / 0.495 / 0.501 / 0.602 / 0.650 / 0.680, 전체 min 0.206~0.320.
# 상한이 필요한 이유: 밴드가 비면(예: h_nav > h_b) 장애물이 사라져 clearance가 무한대가 된다.
GT_CLEARANCE_MEDIAN_RANGE_M = (0.30, 1.00)
GT_CLEARANCE_MIN_M = 0.15                     # 궤적 최소 clearance 하한 (실측 최소 0.180)
GT_CLEARANCE_TYPICAL_RANGE_M = (0.40, 0.80)   # 이 밖이면 경고만 (판정 아님)

# 논문 6단계 반경 역추정용 (03b의 ⑤). `estimate_refine_radius` 참고.
REFINE_PROBE_RADII_M = (0.05, 0.10, 0.20, 0.30, 0.40)
REFINE_CANDIDATES_M = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)

DOWNSAMPLE_MODES = ('any', 'majority', 'all')


# ---------------------------------------------------------------------------
# mesh 로드
# ---------------------------------------------------------------------------

def load_scene_usd(usd_path):
    """`fixed.usd`의 Mesh prim들을 하나의 trimesh로 합쳐 반환.

    **Stage 1의 기본 입력은 obj다** — 논문이 raw scene mesh를 썼고(BlenderProc + Matterport3D,
    USD는 InternNav가 붙인 것), GT도 그 mesh에서 렌더됐기 때문이다. USD는 표면이 obj와 동일하고
    (양방향 표면거리 median 0.00000 m) collision Plane 2장(바닥 구멍 패치, face 4개)만 더 있는데,
    최종 navigable 맵 차이가 0.4%(103셀)에 불과하다.
    **Stage 2(Isaac)로 넘어갈 때** 시뮬레이터가 실제로 충돌 판정하는 자산이므로 이걸 쓴다.
    """
    import trimesh
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    verts, faces, offset = [], [], 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != 'Mesh':
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts:
            continue
        xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        verts.append(np.array([[*xform.Transform(Gf.Vec3d(*p))] for p in points], dtype=np.float64))
        cursor = 0
        for count in counts:                      # n각형 -> 삼각형 fan
            poly = list(indices[cursor:cursor + count])
            cursor += count
            for j in range(1, count - 1):
                faces.append([poly[0] + offset, poly[j] + offset, poly[j + 1] + offset])
        offset += len(points)
    if not verts:
        raise ValueError(f'USD에 Mesh prim이 없다: {usd_path}')
    return trimesh.Trimesh(np.vstack(verts), np.asarray(faces), process=False)


# ---------------------------------------------------------------------------
# (1) 3D occupancy
# ---------------------------------------------------------------------------

def voxelize_surface(mesh, cell_m: float = VOXEL_SIZE_M, samples: int = SURFACE_SAMPLES,
                     seed: int = 0, align_to_m: float = None):
    """mesh 표면을 샘플링해 3D occupancy로 만든다.

    정점만 쓰면 큰 삼각형 내부가 비므로 `sample_surface`로 면적 비례 샘플링한다.

    `align_to_m`을 주면 원점을 그 간격의 **월드 격자에 스냅**한다(내림). 논문은 격자 원점을
    언급하지 않는데, `mesh.bounds[0]`을 그대로 쓰면 격자 위상이 **씬마다 제멋대로**가 된다
    (실측: 월드 0.2 m 격자선까지 y로 0.087 / 0.082 m 어긋남). A\\*가 출력하는 좌표는 셀중심이므로
    이 위상이 경로를 그만큼 옮긴다. 월드 정렬은 **자유 파라미터가 0개인 씬 독립 규칙**이고,
    `s8pcmisQ38h` ep17의 루트 불일치를 해결했다(같은 루트 39/40 -> **40/40**,
    chamfer 1.036 -> 0.140, 위상 면적 1.745 -> 0.000 m²).

    02의 `--grid_align` 기본값이 `ASTAR_CELL_M`이라 실제로는 항상 스냅된다. `0`을 주면 이전 동작.
    내림이므로 `origin <= mesh.bounds[0]`이 유지된다(02의 bounds sanity 검사가 그대로 성립).

    Returns:
        occupancy: (Nx, Ny, Nz) bool
        origin:    (3,) float64 — grid 원점
    """
    import trimesh

    points, _ = trimesh.sample.sample_surface(mesh, samples, seed=seed)
    origin = np.asarray(mesh.bounds[0], dtype=np.float64)
    if align_to_m:
        origin = np.floor(origin / align_to_m) * align_to_m
    shape = ((np.asarray(mesh.bounds[1]) - origin) / cell_m).astype(int) + 1

    idx = ((points - origin) / cell_m).astype(int)
    np.clip(idx, 0, shape - 1, out=idx)
    occupancy = np.zeros(tuple(shape), dtype=bool)
    occupancy[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occupancy, origin


def detect_floor_levels(occupancy: np.ndarray, origin: np.ndarray, cell_m: float = VOXEL_SIZE_M,
                        min_share: float = 0.15, headroom_m=(0.3, 1.2)):
    """바닥 후보 z를 점수 높은 순으로 반환 (다층 씬 대응).

    **Stage 1에서는 쓰지 않는다** — 01이 `scene_meta.json`에 `floor_z`를 정확히 기록하고
    (`cam_z - camera_extrinsic[2,3]`), GT 전 20 에피소드가 바닥 하나를 공유함을 확인했다
    (표준편차 0.0000 m). mesh에서 독립 검증한 결과도 반셀 안이다(+0.039 / -0.024 m).
    이 함수는 **GT가 없는 씬(Stage 3의 GS-map)용 fallback**이다.

    점수 = (그 z의 점유 셀 수) x (그 셀들 위 `headroom_m` 구간이 비어 있는 비율).
    "위가 비어 있는 넓은 수평면"이 바닥이라는 뜻이다. 점유 수만 세면 가구 상판·천장까지
    잡혀 쓸 수 없다(실측: `s8pcmisQ38h`에서 13개가 나왔고 1순위가 2.64 m로 틀렸다).

    Returns: (F,) float32 — world z 값, 점수 높은 순
    """
    from scipy.signal import find_peaks

    n_z = occupancy.shape[2]
    lo, hi = int(headroom_m[0] / cell_m), int(headroom_m[1] / cell_m)
    counts = occupancy.sum(axis=(0, 1)).astype(np.float64)
    if counts.max() <= 0:
        return np.empty((0,), dtype=np.float32)

    free_above = np.zeros(n_z)
    for k in np.flatnonzero(counts):
        here = occupancy[:, :, k]
        above = occupancy[:, :, min(k + lo, n_z - 1):min(k + hi, n_z)].any(axis=2)
        free_above[k] = 1.0 - above[here].mean()

    score = counts * free_above
    if score.max() <= 0:
        return np.empty((0,), dtype=np.float32)
    peaks, _ = find_peaks(score, height=score.max() * min_share,
                          distance=max(1, int(0.5 / cell_m)))   # 층 간 최소 0.5 m
    peaks = peaks[np.argsort(-score[peaks])]
    return (origin[2] + peaks * cell_m).astype(np.float32)


# ---------------------------------------------------------------------------
# (2)~(5) 2D 파생물 — 전부 h_b에 의존하므로 호출 때마다 계산한다
# ---------------------------------------------------------------------------

def compute_scan_coverage_mask(occupancy: np.ndarray) -> np.ndarray:
    """(Nx,Ny,Nz) occupancy -> (Ny,Nx) bool. 그 (x,y) 열에 **어느 높이든** mesh 표면 샘플이 하나라도
    있으면 True — "이 칸이 스캔된 건물 범위 안에 있는가"를 뜻한다.

    `derive_obstacle_2d`+`truncate_navigable`만으로는 이걸 구분 못 한다 — `(h_nav,h_obs]` 밴드에
    장애물이 없으면 navigable로 치는데, 밴드에 장애물이 없는 이유가 "정말 뚫린 공간"인지 "mesh가
    아예 없는 스캔 밖 허공"인지 구별하지 않기 때문이다. 실측(`s8pcmisQ38h`): "navigable"로 분류된
    셀의 **~40%**가 전체 높이에서 mesh 점유가 0인 칸이었다 — GT는 항상 건물 안 좌표만 쓰므로 이
    문제를 안 겪지만, C2(`--mode random`)처럼 navigable 전체에서 균등하게 뽑으면 건물 밖 허공을
    "갈 수 있는 곳"으로 뽑아버린다. `navigable = truncate_navigable(...) & compute_scan_coverage_mask(occupancy)`
    로 호출부에서 추가로 걸러야 한다.
    """
    return occupancy.any(axis=2).T   # (Nx,Ny) -> (Ny,Nx), derive_obstacle_2d와 같은 축 규약


def derive_obstacle_2d(occupancy: np.ndarray, origin: np.ndarray, floor_z: float, h_obs: float,
                       cell_m: float = VOXEL_SIZE_M, h_nav: float = H_NAV_M) -> np.ndarray:
    """논문 2단계 — 바닥 위 `(h_nav, h_obs]` 밴드에 점유가 있는 (x,y) 열 -> (Ny, Nx) bool.

    `h_obs`에는 그 에피소드의 로봇 키 `h_b`를 넣는다(모듈 docstring의 밴드 해석 참고).
    """
    assert h_obs > h_nav, f'h_obs({h_obs})는 h_nav({h_nav})보다 커야 한다'
    n_z = occupancy.shape[2]
    k_lo = int(np.ceil((floor_z + h_nav - origin[2]) / cell_m))
    k_hi = int(np.floor((floor_z + h_obs - origin[2]) / cell_m))
    k_lo, k_hi = max(k_lo, 0), min(k_hi, n_z - 1)
    if k_lo > k_hi:
        return np.zeros(occupancy.shape[1::-1], dtype=bool)   # (Ny, Nx)
    return occupancy[:, :, k_lo:k_hi + 1].any(axis=2).T       # (Nx,Ny) -> (Ny,Nx)


def compute_esdf_2d(obstacle_2d: np.ndarray, cell_m: float = VOXEL_SIZE_M) -> np.ndarray:
    """논문 3단계 — (Ny,Nx) bool obstacle -> (Ny,Nx) float32 ESDF[m]."""
    if not obstacle_2d.any():
        return np.full(obstacle_2d.shape, np.inf, dtype=np.float32)
    return (distance_transform_edt(~obstacle_2d) * cell_m).astype(np.float32)


def truncate_navigable(esdf: np.ndarray, r_b: float = ROBOT_RADIUS_M) -> np.ndarray:
    """논문 4단계 — clearance < `r_b`인 셀을 navigable에서 **제외**한다 -> (Ny,Nx) bool.

    원문 "voxels with distance values lower than the robot radius rb are truncated"는
    **값 clipping이 아니라 집합 축소**다. 값을 자르는 것으로 읽으면 `r_b` 미만 셀이 navigable에
    남아 A*가 벽에 붙어 지나간다.
    """
    return esdf >= r_b


def downsample_navigable(navigable: np.ndarray, factor: int = DOWNSAMPLE_FACTOR,
                         mode: str = 'any') -> np.ndarray:
    """논문 5단계 앞부분 — 0.05 m navigable -> 0.2 m A* 격자.

    **보수적(`all()`)으로 하면 안 된다** — 문틈이 지워져 GT start/goal이 연결되지 않는다.
    실측(12 에피소드, start/goal이 같은 연결성분인 비율):

        nav all()  2/12(4-conn) 5/12(8-conn)      nav any()  12/12  12/12
        esdf min   2/12         5/12              esdf mean  10/12
                                                  esdf center 11/12

    논문은 이 다운샘플을 "to facilitate efficient A* path planning"이라고만 한다 — 안전 목적이
    아니다. 안전은 0.05 m에서의 `r_b` truncate와 refinement(6단계)가 담당하고, 낙관적 A*가
    너무 좁은 틈을 지나는 위험은 `check_path_navigable`로 잡는다.

    `mode`(기본 `any` = 기존 동작): `majority`는 블록의 절반 이상이 navigable일 때만 통과시킨다 —
    현재 파이프라인 실측으로 연결성은 `any`와 같고(19/20, 20/20) navigable만 16% 줄어드는데,
    GT 재현은 씬1만 개선되고 씬2는 악화해 **채택하지 않았다**(옵션으로만 유지).
    `all`은 연결성이 붕괴한다(씬1 4/20).
    """
    h, w = navigable.shape
    hc, wc = h // factor, w // factor
    if hc == 0 or wc == 0:
        return np.zeros((max(hc, 1), max(wc, 1)), dtype=bool)
    block = navigable[:hc * factor, :wc * factor].reshape(hc, factor, wc, factor)
    if mode == 'any':
        return block.any(axis=(1, 3))
    if mode == 'majority':
        return block.mean(axis=(1, 3)) >= 0.5
    if mode == 'all':
        return block.all(axis=(1, 3))
    assert False, f'unreachable mode={mode!r}'


# ---------------------------------------------------------------------------
# (5)~(8) 경로 계획 — 논문 5~7단계
# ---------------------------------------------------------------------------

_NEIGHBORS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
              (1, 1, np.sqrt(2.0)), (1, -1, np.sqrt(2.0)),
              (-1, 1, np.sqrt(2.0)), (-1, -1, np.sqrt(2.0))]


def astar(navigable: np.ndarray, start_ij, goal_ij, connectivity: int = 8,
         clearance: np.ndarray = None, clearance_weight: float = 0.0):
    """격자 A* (논문 5단계). `navigable`은 (Ny,Nx) bool, 좌표는 `(ix, iy)`.

    8-이웃(기본), 대각 비용 sqrt(2), 유클리드 heuristic(admissible). 저장소에 격자 A*가 없어 신규 구현.

    `connectivity=4`면 축 방향 이동만 허용한다(대각 코너컷 체크는 대각 이동이 없으니 자연히
    발동하지 않는다 — 분기 추가 불필요). 04c 실측: 8-이웃이 대각선 지도에서 더 짧은 경로를 낸다.

    `clearance`(navigable와 같은 shape, float, 보통 ESDF)와 `clearance_weight`(기본 0.0)는
    **동일 비용 경로들 사이의 tie-break**을 벽에서 먼 쪽으로 살짝 미는 옵션이다.
    `edge_cost = max(step - clearance_weight * clearance[ny,nx], 1e-6)` — 가중치 0이면 `clearance`를
    줘도 완전한 no-op이라 기존 호출부(`astar(nav, start, goal)`)는 100% 그대로 동작한다.

    Returns: (N,2) int `(ix, iy)` 경로, 또는 실패 시 `None`
    """
    import heapq

    assert connectivity in (4, 8), f'unreachable connectivity={connectivity!r}'
    neighbors = _NEIGHBORS if connectivity == 8 else _NEIGHBORS[:4]

    h_grid, w_grid = navigable.shape
    start, goal = (int(start_ij[0]), int(start_ij[1])), (int(goal_ij[0]), int(goal_ij[1]))
    for name, (ix, iy) in (('start', start), ('goal', goal)):
        if not (0 <= ix < w_grid and 0 <= iy < h_grid) or not navigable[iy, ix]:
            return None
    if start == goal:
        return np.array([start], dtype=int)

    def heuristic(node):
        return float(np.hypot(node[0] - goal[0], node[1] - goal[1]))

    open_heap = [(heuristic(start), 0.0, start)]
    came_from = {}
    best_cost = {start: 0.0}
    closed = set()
    while open_heap:
        _, cost, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        if node == goal:
            path = [node]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            return np.array(path[::-1], dtype=int)
        closed.add(node)
        ix, iy = node
        for dx, dy, step in neighbors:
            nx, ny = ix + dx, iy + dy
            if not (0 <= nx < w_grid and 0 <= ny < h_grid) or not navigable[ny, nx]:
                continue
            # 대각 이동은 양옆이 모두 열려 있어야 한다 — 벽 모서리를 대각으로 통과하는 것을 막는다.
            if dx and dy and not (navigable[iy, nx] and navigable[ny, ix]):
                continue
            edge_cost = step
            if clearance is not None and clearance_weight:
                edge_cost = max(step - clearance_weight * float(clearance[ny, nx]), 1e-6)
            neighbor = (nx, ny)
            new_cost = cost + edge_cost
            if new_cost < best_cost.get(neighbor, np.inf):
                best_cost[neighbor] = new_cost
                came_from[neighbor] = node
                heapq.heappush(open_heap, (new_cost + heuristic(neighbor), new_cost, neighbor))
    return None


def greedy_refine(waypoints_xy: np.ndarray, esdf: np.ndarray, origin: np.ndarray,
                  cell_m: float = VOXEL_SIZE_M, radius_m: float = 0.5,
                  fix_endpoints: bool = True) -> np.ndarray:
    """논문 6단계 — 각 waypoint를 `radius_m` 국소 영역에서 ESDF가 최대인 위치로 옮긴다.

    "a greedy search is performed in a local area of the original ESDF map to refine the position
    by maximizing the distance to nearby obstacles". **원본(0.05 m) ESDF를 쓴다** — A*는 0.2 m
    격자에서 돌지만 보정은 원해상도에서 한다.

    `fix_endpoints=True`면 start/goal은 그대로 둔다(reproduce 모드에서 GT 값을 유지하기 위함).

    **이 단계는 정규화(normalization) 연산이다** — 어떤 루트든 복도 중심선(ESDF 능선)으로 모은다.
    GT도 refine을 거쳐 능선 근처에 있으므로 우리 A*의 계단 모양이 씻겨나가며 GT와 수렴한다.
    덜 움직이게 바꾸면(`refine_min_move`) 기준선은 좋아지지만 **실제 경로는 나빠진다** — 그 비교는
    `03c_compare_refine.py` 참고.
    """
    ij = world_to_cell(waypoints_xy, origin, cell_m)
    h_grid, w_grid = esdf.shape
    radius = int(round(radius_m / cell_m))
    out = ij.copy()
    for n, (ix, iy) in enumerate(ij):
        if fix_endpoints and n in (0, len(ij) - 1):
            continue
        x0, x1 = max(ix - radius, 0), min(ix + radius + 1, w_grid)
        y0, y1 = max(iy - radius, 0), min(iy + radius + 1, h_grid)
        window = esdf[y0:y1, x0:x1]
        if window.size == 0:
            continue
        dy, dx = np.unravel_index(np.argmax(window), window.shape)
        out[n] = (x0 + dx, y0 + dy)
    return cell_to_world(out, origin, cell_m)


def refine_min_move(waypoints_xy: np.ndarray, esdf: np.ndarray, origin: np.ndarray,
                    cell_m: float = VOXEL_SIZE_M, radius_m: float = 0.30,
                    fix_endpoints: bool = True, r_b: float = ROBOT_RADIUS_M) -> np.ndarray:
    """논문 6단계의 **대안 해석** — "최대한 밀어낸다"가 아니라 "`r_b`를 만족하는 만큼만 옮긴다".

    clearance가 이미 `r_b` 이상이면 **움직이지 않고**, 미달일 때만 창 안에서 `esdf >= r_b`인
    **가장 가까운** 셀로 옮긴다(창 안에 없으면 argmax로 폴백 — 안전 우선).

    실측 — **기준선은 좋아지지만 실제 경로는 나빠진다**(20 에피소드 median, 씬1/씬2):

        방식        기준선 chamfer      실제 경로 chamfer
        argmax      0.080 / 0.109       0.103 / 0.134   <- 채택
        min_move    0.058 / 0.062       0.142 / 0.165

    `argmax`가 루트를 복도 중심선으로 **정규화**하는데, 기준선은 GT 자신의 셀 시퀀스에서 출발하므로
    안 움직이는 쪽이 유리할 뿐이고 그 이점은 GT 루트를 모르는 실제 생성에서 쓸 수 없다.
    게이트 "refine이 clearance를 올렸는가"도 통과하지 못한다. `--refine_mode min_move` 옵션으로만 유지.

    `radius_m` 기본이 `greedy_refine`보다 큰 0.30인 이유: 최소 이동이라 창을 넓게 줘도 멀리 가지
    않는다(반경 0.15/0.30/0.60에서 수치 동일). 오히려 좁으면 `r_b`를 채울 셀을 못 찾아 폴백한다.
    """
    ij = world_to_cell(waypoints_xy, origin, cell_m)
    h_grid, w_grid = esdf.shape
    radius = max(1, int(round(radius_m / cell_m)))
    out = ij.copy()
    for n, (ix, iy) in enumerate(ij):
        if fix_endpoints and n in (0, len(ij) - 1):
            continue
        if 0 <= ix < w_grid and 0 <= iy < h_grid and esdf[iy, ix] >= r_b:
            continue                                   # 이미 충족 -> 그대로 둔다 (핵심 차이)
        x0, x1 = max(ix - radius, 0), min(ix + radius + 1, w_grid)
        y0, y1 = max(iy - radius, 0), min(iy + radius + 1, h_grid)
        window = esdf[y0:y1, x0:x1]
        if window.size == 0:
            continue
        ok = np.argwhere(window >= r_b)
        if len(ok) == 0:
            dy, dx = np.unravel_index(np.argmax(window), window.shape)   # 폴백: 최대점
        else:
            d = (ok[:, 0] + y0 - iy) ** 2 + (ok[:, 1] + x0 - ix) ** 2
            dy, dx = ok[int(np.argmin(d))]
        out[n] = (x0 + dx, y0 + dy)
    return cell_to_world(out, origin, cell_m)


# 논문 6단계의 두 해석. 기본은 `argmax`(논문 문장 그대로) — 03의 `--refine_mode`로 고른다.
REFINERS = {'argmax': greedy_refine, 'min_move': refine_min_move}


def thin_waypoints(waypoints_xy: np.ndarray, spacing_m: float) -> np.ndarray:
    """waypoint를 호길이 `spacing_m` 간격으로 솎아낸다 (양 끝은 유지). **논문에 없는 추가 단계.**

    A*가 0.2 m 격자에서 45도 단위로만 꺾이는데 cubic spline이 그 점을 **전부** 통과해 지그재그가
    남는다. 스무딩 **전에** 솎으면 GT 수준이 된다 (`17DRP5sb8fy` 20 에피소드):

        간격    회전(도/m)  배율   chamfer  같은루트  spline min clearance
        0.2      99.8       2.80x  0.094    20/20    0.250      <- 솎지 않음
        0.4      64.8       1.9x   0.103    20/20    0.200
        0.8      35.0       1.03x  0.102    20/20    0.158      <- 채택
        GT       33.7        -       -        -      0.206

    `r_b` 제약은 **솎기 전** `wp_refined`에 걸리므로 안전 판정은 영향받지 않는다.

    **Douglas-Peucker(RDP)는 쓰면 안 된다** — 기하 편차만 보고 장애물을 확인하지 않아 코너를
    관통해 잘라낸다(eps 0.05/0.10/0.20 전부 spline min clearance가 0.000으로 붕괴).

    03b의 knot 복원이 GT에서 역추정한 간격이 0.79~0.90 m라 이 값과 맞는다 — **논문 표기
    `[(x0,y0),...,(xk,yk)]`의 k가 작다는 뜻이고, 이 단계는 일탈이 아니라 복원일 수 있다.**
    """
    pts = np.asarray(waypoints_xy, dtype=np.float64)
    if len(pts) < 3 or spacing_m <= 0:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep, last = [0], 0.0
    for n in range(1, len(pts) - 1):
        if s[n] - last >= spacing_m:
            keep.append(n)
            last = s[n]
    keep.append(len(pts) - 1)
    return pts[keep]


def smooth_cubic_spline(waypoints_xy: np.ndarray, step_m: float = VOXEL_SIZE_M) -> np.ndarray:
    """논문 7단계 (기본) — 누적거리로 파라미터화한 cubic spline. **waypoint를 통과한다.**

    `internnav/dataset/navdp_lerobot_dataset.py:472`의 파라미터화 패턴과 같다.
    급코너에서 오버슈트할 수 있어 clearance가 떨어질 수 있다 — 그게 싫으면 `smooth_bezier`.
    """
    from scipy.interpolate import CubicSpline

    pts = np.asarray(waypoints_xy, dtype=np.float64)
    if len(pts) < 3:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-9])      # 중복점 제거 (CubicSpline은 t가 증가해야 한다)
    pts, t = pts[keep], t[keep]
    if len(pts) < 3:
        return pts.copy()
    n_out = max(2, int(t[-1] / step_m) + 1)
    ts = np.linspace(t[0], t[-1], n_out)
    return np.stack([CubicSpline(t, pts[:, 0])(ts), CubicSpline(t, pts[:, 1])(ts)], axis=1)


def smooth_bezier(waypoints_xy: np.ndarray, step_m: float = VOXEL_SIZE_M) -> np.ndarray:
    """옵션 — 합성 quadratic Bézier. waypoint를 통과하지 않고 **convex hull 안에 머문다.**

    연속 삼중점 `(p0, p1, p2)`마다 중점 `m0=(p0+p1)/2`, `m1=(p1+p2)/2`를 양 끝으로,
    `p1`을 제어점으로 하는 2차 Bézier를 잇는다. 양 끝점은 그대로 유지한다.

    실측 (현재 파이프라인, 씬1 / 씬2) — 오버슈트가 없어 **clearance와 헤딩이 낫지만** GT보다
    **짧아지고**(길이비 0.955) **GT의 실제 회전까지 씻어낸다**(①이 1 미만 = 과도 스무딩):

        스무딩   chamfer         Fréchet        ①회전       ③헤딩      traj min clr  길이비
        cubic    0.103 / 0.134   0.266 / 0.333  1.01 / 1.30  9.0 / 11.5  0.100 / 0.200  0.993 / 1.010
        bezier   0.112 / 0.133   0.274 / 0.322  0.67 / 0.86  8.5 / 8.8   0.141 / 0.180  0.955 / 0.985

    기본값은 논문대로 `cubic`. 좁은 씬·큰 `r_b`·04의 방향 정합이 중요할 때 쓸 옵션이다.
    """
    pts = np.asarray(waypoints_xy, dtype=np.float64)
    if len(pts) < 3:
        return pts.copy()
    out = [pts[0]]
    for p0, p1, p2 in zip(pts[:-2], pts[1:-1], pts[2:]):
        m0 = p0 if len(out) == 1 else (p0 + p1) / 2.0
        m1 = (p1 + p2) / 2.0
        n = max(2, int(np.linalg.norm(m1 - m0) / step_m) + 1)
        u = np.linspace(0.0, 1.0, n)[:, None]
        out.append(((1 - u) ** 2) * m0 + 2 * u * (1 - u) * p1 + (u ** 2) * m1)
    out.append(pts[-1][None, :])
    return np.vstack([np.atleast_2d(o) for o in out])


# 논문 7단계의 두 스무딩. 기본은 `cubic`(논문 문장 그대로) — 03의 `--smooth`로 고른다.
# **모든 호출부가 이 registry를 써야 한다** — 어느 한 곳이 cubic을 하드코딩하면 기준선이 다른
# 스무딩으로 계산돼 배수 비교가 무의미해지고, 03b의 F1(결정성)이 통째로 실패한다(실제로 겪음).
SMOOTHERS = {'cubic': smooth_cubic_spline, 'bezier': smooth_bezier}


def check_path_navigable(path_xy: np.ndarray, esdf: np.ndarray, origin: np.ndarray,
                         cell_m: float = VOXEL_SIZE_M, r_b: float = ROBOT_RADIUS_M,
                         sample_step_m: float = None) -> dict:
    """경로가 실제로 통행 가능한지 — 점뿐 아니라 **점 사이 선분까지** 확인한다.

    낙관적 다운샘플(`downsample_navigable`)로 A*가 너무 좁은 틈을 지날 수 있으므로, 연속 점 사이를
    `sample_step_m` 간격으로 촘촘히 찍어 clearance를 잰다. 점만 보면 사이를 놓친다.

    두 종류를 구분한다:
      - **하드 충돌** `hard_*` : clearance == 0. 장애물 voxel 안이라 질점 로봇도 못 지난다 =
        맵/플래너 버그. 0이 아니면 무조건 실패로 봐야 한다.
      - **몸통 침범** `segments_ok`: clearance < `r_b`. 질점은 지나지만 로봇 반경이 겹친다.
        GT 궤적도 이걸 위반하므로(씬1 5/20, 씬2 4/20 에피소드) 하드 제약으로 쓰지 않는다.

    기본 `sample_step_m`은 셀의 1/5 — 셀 크기로 찍으면 좁은 틈을 스치는 구간을 놓친다.
    """
    step = sample_step_m or cell_m / 5.0
    pts = np.asarray(path_xy, dtype=np.float64)
    at_points = sample_esdf_at(esdf, pts, origin, cell_m)

    dense = [pts[0][None, :]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(2, int(np.linalg.norm(b - a) / step) + 1)
        dense.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    dense = np.vstack(dense)
    along = sample_esdf_at(esdf, dense, origin, cell_m)
    return {
        'points_min_m': float(at_points.min()) if len(at_points) else float('nan'),
        'points_median_m': float(np.median(at_points)) if len(at_points) else float('nan'),
        'segments_min_m': float(along.min()) if len(along) else float('nan'),
        'length_m': float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()),
        'r_b': r_b,
        'points_ok': bool(len(at_points) and at_points.min() >= r_b),
        'segments_ok': bool(len(along) and along.min() >= r_b),
        'n_samples': int(len(along)),
        'hard_n': int((along <= 1e-9).sum()),
        'hard_ok': bool(len(along) and (along > 1e-9).all()),
    }


# ---------------------------------------------------------------------------
# 경로 비교 지표 (GT 재현 검증용)
#
# **절대값만 보면 안 된다** — 경로가 정말 같아도 지표는 0이 아니다. `pipeline_floor_path`가 주는
# 기준선 ⓒ("A*가 GT 루트를 정확히 골랐을 때 우리 파이프라인이 내는 값")와 함께 읽을 것.
# ---------------------------------------------------------------------------

def resample_by_arclength(path_xy: np.ndarray, n: int = None, step_m: float = None) -> np.ndarray:
    """경로를 **호길이 등간격**으로 리샘플. `n`(점 개수) 또는 `step_m`(간격) 중 하나를 준다.

    GT는 시간축 샘플이라 회전 구간에서 점이 촘촘하다(스텝 비율 0.71~0.97). 점 분포가 다른 두 경로를
    대응점끼리 비교하려면 반드시 호길이로 다시 뽑아야 한다.
    """
    p = np.asarray(path_xy, dtype=np.float64)
    if len(p) < 2:
        return p.copy()
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    if s[-1] <= 0:
        return p[:1].repeat(max(n or 2, 2), axis=0)
    count = n if n else max(3, int(s[-1] / (step_m or VOXEL_SIZE_M)) + 1)
    t = np.linspace(0.0, s[-1], count)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)


def path_smoothness(path_xy: np.ndarray, step_m: float = VOXEL_SIZE_M) -> dict:
    """**지표 ①** 경로가 얼마나 완만한가 — 미터당 회전각[도/m]과 곡률.

    chamfer/Fréchet가 **둘 다 통과했는데도** 우리 경로가 GT보다 2.8배 지그재그였다(100 vs 34 도/m).
    위치는 맞고 **모양**이 틀린 경우를 잡는 유일한 지표다.

    03의 게이트는 **양쪽**이다(0.5~2.0). 상한은 A*의 45도 격자 꺾임이 spline에 새는 것을,
    하한은 스무딩이 GT의 실제 회전까지 씻어내는 것을 잡는다(bezier가 0.67을 내면서 필요성이 드러났다).

    Returns: {'turn_per_m_deg', 'total_turn_deg', 'kappa_median', 'kappa_p95', 'length_m'}
    """
    q = resample_by_arclength(path_xy, step_m=step_m)
    if len(q) < 4:
        return {'turn_per_m_deg': 0.0, 'total_turn_deg': 0.0,
                'kappa_median': 0.0, 'kappa_p95': 0.0, 'length_m': 0.0}
    seg = np.diff(q, axis=0)
    length = float(np.linalg.norm(seg, axis=1).sum())
    theta = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    d_theta = np.abs(np.diff(theta))
    kappa = d_theta / np.maximum(np.linalg.norm(seg, axis=1)[1:], 1e-9)
    total = float(np.degrees(d_theta.sum()))
    return {'turn_per_m_deg': total / max(length, 1e-9), 'total_turn_deg': total,
            'kappa_median': float(np.median(kappa)), 'kappa_p95': float(np.percentile(kappa, 95)),
            'length_m': length}


def compare_clearance_profile(ours_xy, gt_xy, esdf, origin, cell_m: float = VOXEL_SIZE_M) -> dict:
    """**지표 ②** 같은 ESDF에서 두 경로의 clearance 분포를 비교한다.

    `median_diff_m`이 양수면 우리 경로가 GT보다 벽에서 멀다 = refinement가 과하다는 뜻이다
    (실측으로 +0.032 m 편향을 이 방식으로 찾아 `refine_radius`를 0.15 -> 0.10으로 바꿨다).
    우리 값만 기록하면 과/부족을 알 수 없다.
    """
    co = sample_esdf_at(esdf, np.asarray(ours_xy), origin, cell_m)
    cg = sample_esdf_at(esdf, np.asarray(gt_xy), origin, cell_m)
    if not len(co) or not len(cg):
        return {'median_diff_m': float('nan')}
    return {
        'ours_median_m': float(np.median(co)), 'gt_median_m': float(np.median(cg)),
        'median_diff_m': float(np.median(co) - np.median(cg)),
        'ours_p10_m': float(np.percentile(co, 10)), 'gt_p10_m': float(np.percentile(cg, 10)),
        'ours_min_m': float(co.min()), 'gt_min_m': float(cg.min()),
    }


def heading_alignment(ours_xy, gt_xy, n: int = 200) -> dict:
    """**지표 ③** 호길이 정렬 후 진행 방향(헤딩) 차이 [도].

    04가 이 경로로 카메라를 배치하므로 **위치보다 방향이 관측에 더 큰 영향**을 준다 —
    위치 5 cm 어긋남보다 헤딩 10도 어긋남이 렌더 결과를 더 바꾼다.

    **하한이 0이 아니다** — 0.2 m 격자에 얹는 순간 45도 꺾임이 생겨 기준선이 이미 6도다.
    실측 8.0 / 10.3도를 "8도 틀렸다"로 읽으면 안 되고 "하한 대비 2~4도 초과"로 읽어야 한다.
    """
    a, b = resample_by_arclength(ours_xy, n=n), resample_by_arclength(gt_xy, n=n)
    if len(a) < 2 or len(b) < 2:
        return {'median_deg': float('nan')}
    ta, tb = np.diff(a, axis=0), np.diff(b, axis=0)
    na, nb = np.linalg.norm(ta, axis=1), np.linalg.norm(tb, axis=1)
    ok = (na > 1e-9) & (nb > 1e-9)
    if not ok.any():
        return {'median_deg': float('nan')}
    cos = np.sum(ta[ok] * tb[ok], axis=1) / (na[ok] * nb[ok])
    deg = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return {'median_deg': float(np.median(deg)), 'p90_deg': float(np.percentile(deg, 90)),
            'max_deg': float(deg.max())}


def chamfer_distance(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    """양방향 최근접거리 평균 [m] — **순서를 보지 않는다**. 주 지표."""
    a, b = np.asarray(a_xy, dtype=np.float64), np.asarray(b_xy, dtype=np.float64)
    if not len(a) or not len(b):
        return float('nan')
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))


def discrete_frechet_distance(a_xy: np.ndarray, b_xy: np.ndarray, max_points: int = 400) -> float:
    """이산 Fréchet 거리 [m] — **순서를 본다**. "같은 루트인가" 판정에 쓴다(임계 = 문 폭 0.8 m).

    O(NM)이라 `max_points`로 리샘플해서 잰다.
    """
    a = resample_by_arclength(a_xy, n=min(len(np.asarray(a_xy)), max_points))
    b = resample_by_arclength(b_xy, n=min(len(np.asarray(b_xy)), max_points))
    if not len(a) or not len(b):
        return float('nan')
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    acc = np.full_like(dist, np.inf)
    acc[0, 0] = dist[0, 0]
    for i in range(1, len(a)):
        acc[i, 0] = max(acc[i - 1, 0], dist[i, 0])
    for j in range(1, len(b)):
        acc[0, j] = max(acc[0, j - 1], dist[0, j])
    for i in range(1, len(a)):
        for j in range(1, len(b)):
            acc[i, j] = max(min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]), dist[i, j])
    return float(acc[-1, -1])


# ---------------------------------------------------------------------------
# 논문 단계별 역검증 — "GT가 그 단계의 산출물로서 성립하는가"
#
# chamfer/Fréchet는 **최종 곡선**만 본다. 그래서 "chamfer 0.103 m가 좋은 값인가?"에 답할 수 없고,
# 논문이 값을 안 준 파라미터를 지표 스윕으로 고르면 순환논증이 된다. 아래 함수들은 논문 5·6·7단계를
# **GT에 거꾸로 적용**해 그걸 해결한다. 03b가 이걸 쓴다.
# ---------------------------------------------------------------------------

def pipeline_floor_path(gt_xy: np.ndarray, esdf: np.ndarray, origin: np.ndarray,
                        cell_m: float = VOXEL_SIZE_M, coarse_cell_m: float = ASTAR_CELL_M,
                        refine_radius_m: float = 0.10, spacing_m: float = WAYPOINT_SPACING_M,
                        smooth_step_m: float = VOXEL_SIZE_M, with_refine: bool = True,
                        refine_mode: str = 'argmax', smooth_mode: str = 'cubic') -> np.ndarray:
    """**지표의 기준선 ⓒ** — "A\\*가 GT 루트를 정확히 골랐다면 우리 파이프라인이 내는 경로".

    GT 루트를 0.2 m 격자 셀 시퀀스로 얹어 A\\*의 산출물 자리에 놓고, 그 뒤 논문 6~7단계
    (refine -> `thin_waypoints` -> 스무딩)를 **실제와 똑같이** 돌린다 — `refine_mode`/`smooth_mode`를
    03이 쓴 값과 반드시 일치시켜야 배수 비교가 성립한다.

    이게 필요한 이유: **루트가 완벽해도 chamfer는 0이 아니다**(실측 0.072/0.100 m). 기준선을 모르고
    실측 0.103을 보면 "많이 틀렸다"로 오독한다 — 실제로는 기준선의 1.45배다.
    단계별 기여(chamfer 누적, 씬1/씬2):

        s0 GT (리샘플만)          0.011 / 0.011
        s1 0.2 m 셀중심 스냅       0.045 / 0.045   (+0.035 / +0.034)  <- A*가 낼 수 있는 좌표뿐
        s2 + 양끝 GT 고정          0.044 / 0.043   (-0.001 / -0.002)
        s3 + greedy_refine        0.072 / 0.101   (+0.028 / +0.058)  <- GT를 안 보고 밀어낸다
        s4 + thin_waypoints       0.069 / 0.093   (-0.003 / -0.008)  <- 격자 꺾임 제거로 오히려 개선
        s5 + cubic spline = ⓒ     0.079 / 0.108   (+0.011 / +0.015)

    `with_refine=False`면 격자+스무딩만의 기여를 분리해 볼 수 있다(기준선 ⓑ).

    **`metric_floors`(F2/F3)보다 이걸 기준선으로 쓸 것.** F3는 격자 스냅만 재서 씬마다 크게
    흔들린다(0.105 vs 0.063) — 같은 상태의 두 씬을 "한계 도달 / 여지 있음"으로 갈라 보이게 했다.
    """
    gt = np.asarray(gt_xy, dtype=np.float64)
    if len(gt) < 2:
        return gt.copy()
    cells = path_cell_sequence(gt, origin, coarse_cell_m)
    wp = cell_to_world(cells, origin, coarse_cell_m)
    wp[0], wp[-1] = gt[0], gt[-1]          # 03과 같게 start/goal은 GT 값을 유지한다
    if with_refine:
        # **기준선도 03과 같은 refiner를 써야** 비교가 성립한다 (모드가 어긋나면 배수가 무의미해진다)
        wp = REFINERS[refine_mode](wp, esdf, origin, cell_m, refine_radius_m, fix_endpoints=True)
    return SMOOTHERS[smooth_mode](thin_waypoints(wp, spacing_m), smooth_step_m)


def metric_floors(gt_xy: np.ndarray, coarse_cell_m: float = ASTAR_CELL_M,
                  smooth_step_m: float = VOXEL_SIZE_M) -> dict:
    """오차 **원인별 분해** — 격자(F3)와 점분포(F2)가 각각 얼마를 만드는지 본다.

    **기준선으로는 `pipeline_floor_path`를 쓸 것.** 여기 F3는 격자 스냅만 재서 씬마다 크게
    흔들린다(실측 0.105 vs 0.063). 이 함수는 원인 분해 용도로만 남긴다.

      - **F2 점분포**: GT를 호길이 등간격으로 재샘플한 것과 원래 GT를 비교. GT가 시간축 샘플
        (회전 시 감속 -> 코너에 점이 몰림)이라는 사실만으로 생기는 오차. 실측 0.011 m로 작다 =
        점 분포는 주원인이 아니다.
      - **F3 격자**: GT를 0.2 m 셀중심에 스냅한 것과 원래 GT를 비교.

    F1(자기재현 = 파이프라인 결정성)은 재계획이 필요해 여기 없다 — 03b가 담당한다.
    """
    gt = np.asarray(gt_xy, dtype=np.float64)
    if len(gt) < 2:
        return {k: float('nan') for k in
                ('f2_chamfer_m', 'f2_frechet_m', 'f3_chamfer_m', 'f3_frechet_m', 'f3_aligned_p95_m')}

    # F2: 같은 곡선, 점 분포만 다르게 (우리 궤적의 샘플 간격과 같게 맞춘다)
    f2 = resample_by_arclength(gt, step_m=smooth_step_m)

    # F3: 0.2 m 셀중심으로 스냅 -> 연속 중복 제거 (A*가 낼 수 있는 최선의 GT 근사).
    # 셀중심 점만 두면 점이 0.2 m 간격이라 **점 희소성**이 격자 오차에 섞인다 — 우리 궤적은 스무딩으로
    # 촘촘하므로, 스냅한 꺾은선을 같은 간격으로 리샘플해 격자 성분만 남긴다.
    ij = world_to_cell(gt, np.zeros(3), coarse_cell_m)
    keep = np.concatenate([[True], (np.diff(ij, axis=0) != 0).any(axis=1)])
    f3_cells = cell_to_world(ij[keep], np.zeros(3), coarse_cell_m)
    f3 = resample_by_arclength(f3_cells, step_m=smooth_step_m)

    a = resample_by_arclength(f3, n=200)
    b = resample_by_arclength(gt, n=200)
    return {
        'f2_chamfer_m': chamfer_distance(f2, gt), 'f2_frechet_m': discrete_frechet_distance(f2, gt),
        'f3_chamfer_m': chamfer_distance(f3, gt), 'f3_frechet_m': discrete_frechet_distance(f3, gt),
        'f3_aligned_p95_m': float(np.percentile(np.linalg.norm(a - b, axis=1), 95)),
        'f3_n_cells': int(keep.sum()),
    }


def path_cell_sequence(path_xy: np.ndarray, origin: np.ndarray,
                       cell_m: float = ASTAR_CELL_M) -> np.ndarray:
    """경로가 지나는 격자 셀 시퀀스 `(ix, iy)` — **인접 셀만 남게** 끊김을 메운다.

    단순 스냅 + 중복 제거로는 셀을 건너뛴다(연속 GT 점이 0.2 m보다 멀면 대각조차 아닌 점프가 생김).
    A\\* 비용을 재려면 인접 셀 시퀀스여야 하므로, 점프 구간은 두 셀을 잇는 직선을 셀 크기의 1/4로
    촘촘히 찍어 채운다.
    """
    ij = world_to_cell(np.asarray(path_xy, dtype=np.float64), origin, cell_m)
    out = [tuple(ij[0])]
    for a, b in zip(ij[:-1], ij[1:]):
        if tuple(b) == out[-1]:
            continue
        if max(abs(b[0] - out[-1][0]), abs(b[1] - out[-1][1])) <= 1:
            out.append(tuple(b))
            continue
        pa = cell_to_world(np.atleast_2d(out[-1]), origin, cell_m)[0]
        pb = cell_to_world(np.atleast_2d(b), origin, cell_m)[0]
        n = int(np.linalg.norm(pb - pa) / (cell_m / 4)) + 2
        for p in pa + (pb - pa) * np.linspace(0, 1, n)[:, None]:
            c = tuple(world_to_cell(np.atleast_2d(p), origin, cell_m)[0])
            if c != out[-1] and max(abs(c[0] - out[-1][0]), abs(c[1] - out[-1][1])) <= 1:
                out.append(c)
    return np.array(out, dtype=int)


def astar_cost_gap(gt_xy: np.ndarray, our_path_ij: np.ndarray, nav_coarse: np.ndarray,
                   origin: np.ndarray, coarse_cell_m: float = ASTAR_CELL_M,
                   n_via: int = 8) -> dict:
    """**논문 5단계 역검증 (지표 ④)** — 루트가 갈렸을 때 "맵이 다른가 / 그냥 동점인가"를 가른다.

    논문 5단계가 A\\*이고 start/goal이 같으면 **같은 맵에서 두 경로의 비용은 같아야 한다.**
    GT를 따라가도록 강제한 A\\*(GT에서 뽑은 `n_via`개 경유점을 순서대로 잇는 A\\*)의 비용을
    우리 A\\* 비용과 비교한다.

      - `cost_ratio` ~ 1.0 인데 루트가 다르다  -> **동점**. tie-breaking 차이라 우리 잘못이 아니다
      - `cost_ratio` > 1.05                    -> GT 루트가 우리 맵에서 실제로 더 비싸다
                                                  = 우리 맵에 GT 맵엔 없던 장애물 (h_nav/r_b 재조사)
      - `n_blocked > 0`                        -> GT가 우리 navigable 밖을 지남 = 맵이 과대 장애물

    **GT 궤적의 셀 시퀀스 비용을 직접 세면 안 된다** — 매끄러운 곡선을 셀로 바꾸면 45도 구간이
    계단(1+1=2)으로 세어져 A\\*의 대각(√2=1.41)보다 비싸진다. 그 방식으로 재면 같은 루트에서도
    1.04~1.32가 나와(실측) 맵 차이와 구분되지 않는다. **양쪽 다 A\\* 비용이어야 비교가 성립한다.**

    **항등값이 0이 아니라 ~1.02다**(경유점 제약 자체의 비용). 우리 궤적을 GT 자리에 넣어도
    1.016 / 1.028이 나온다 — 실측 1.024 / 1.023이 이와 사실상 같으므로 GT 루트는 우리 맵에서 최적이다.

    `n_via`는 GT 루트를 얼마나 촘촘히 강제할지다. 1개면 그 점이 두 루트의 공통 구간에 있을 때
    차이가 안 보여 "동점"으로 오판한다(실제로 겪음). 늘리면 제약이 세져 비율이 커지므로
    **값을 함께 보고해야 하는 지표다.** 비용 단위는 셀 수(m는 `* coarse_cell_m`).
    """
    def cost_of(cells):
        d = np.abs(np.diff(np.asarray(cells, dtype=np.float64), axis=0))
        return float(np.where((d[:, 0] > 0) & (d[:, 1] > 0), np.sqrt(2.0), 1.0).sum())

    h_grid, w_grid = nav_coarse.shape
    gt_cells = path_cell_sequence(gt_xy, origin, coarse_cell_m)
    inside = ((gt_cells[:, 0] >= 0) & (gt_cells[:, 0] < w_grid)
              & (gt_cells[:, 1] >= 0) & (gt_cells[:, 1] < h_grid))
    blocked = int((~inside).sum())
    if inside.any():
        blocked += int((~nav_coarse[gt_cells[inside][:, 1], gt_cells[inside][:, 0]]).sum())

    # GT에서 경유 셀 뽑기 — navigable 아니면 이웃에서 가장 가까운 navigable로 옮긴다
    pick = np.unique(np.linspace(0, len(gt_cells) - 1, max(2, n_via)).astype(int))
    via, n_snapped = [], 0
    for ix, iy in gt_cells[pick]:
        if 0 <= ix < w_grid and 0 <= iy < h_grid and nav_coarse[iy, ix]:
            cand = (int(ix), int(iy))
        else:
            cand = None
            for r in (1, 2):
                ys, xs = np.mgrid[iy - r:iy + r + 1, ix - r:ix + r + 1]
                ok = ((xs >= 0) & (xs < w_grid) & (ys >= 0) & (ys < h_grid))
                ys, xs = ys[ok], xs[ok]
                free = nav_coarse[ys, xs]
                if free.any():
                    d = (xs[free] - ix) ** 2 + (ys[free] - iy) ** 2
                    j = int(np.argmin(d))
                    cand = (int(xs[free][j]), int(ys[free][j]))
                    n_snapped += 1
                    break
            if cand is None:
                continue
        if not via or cand != via[-1]:
            via.append(cand)

    gt_cost, n_failed = 0.0, 0
    for a, b in zip(via[:-1], via[1:]):
        leg = astar(nav_coarse, a, b)
        if leg is None:
            n_failed += 1
            continue
        gt_cost += cost_of(leg)
    our_cost = cost_of(np.asarray(our_path_ij))
    return {
        'cost_ratio': gt_cost / our_cost if our_cost > 0 and not n_failed else float('nan'),
        'gt_cost_cells': gt_cost, 'our_cost_cells': our_cost,
        'gt_cost_m': gt_cost * coarse_cell_m, 'our_cost_m': our_cost * coarse_cell_m,
        'n_blocked': blocked, 'n_gt_cells': int(len(gt_cells)),
        'n_via': len(via), 'n_via_snapped': n_snapped, 'n_via_legs_failed': n_failed,
    }


def refine_displacement_profile(waypoints_xy: np.ndarray, esdf: np.ndarray, origin: np.ndarray,
                                cell_m: float = VOXEL_SIZE_M, probe_radii_m=None) -> np.ndarray:
    """각 탐침 반경에서 `greedy_refine`을 **한 번 더** 걸었을 때의 median 이동거리 [m].

    이미 벽에서 멀리 밀려난 경로는 ESDF 평탄부에 있어 덜 움직인다. 그래서 이 곡선이
    "그 경로가 얼마나 refine 되어 있는가"의 지문이 된다 — `estimate_refine_radius`가 이걸 쓴다.

    **"refine 산출물은 국소최대다"로 검사하면 안 된다.** greedy refine은 한 번만 적용되고 옮긴 뒤
    창도 함께 이동하므로 산출물은 국소최대가 아니다 — 우리 `wp_refined`(반경 0.15로 만든 것)조차
    국소최대 비율이 14%에 불과했다(양성 대조 실패). 그 전제로는 아무 결론도 못 낸다.
    """
    wp = np.asarray(waypoints_xy, dtype=np.float64)
    radii = probe_radii_m if probe_radii_m is not None else REFINE_PROBE_RADII_M
    return np.array([float(np.median(np.linalg.norm(
        greedy_refine(wp, esdf, origin, cell_m, r, fix_endpoints=False) - wp, axis=1)))
        for r in radii])


def estimate_refine_radius(gt_waypoints_xy: np.ndarray, astar_waypoints_xy: np.ndarray,
                           esdf: np.ndarray, origin: np.ndarray, cell_m: float = VOXEL_SIZE_M,
                           candidates_m=None, probe_radii_m=None) -> dict:
    """**논문 6단계 역검증 (지표 ⑤)** — 논문이 "a local area"라고만 한 반경을 GT에서 역추정한다.

    후보 반경 `R_c`마다 **같은 A\\* waypoint를** refine해서 이동거리 프로파일을 만들고,
    GT 프로파일과 L1 거리가 최소인 `R_c`를 고른다. 현재 `refine_radius=0.10`은 chamfer 스윕으로
    골랐으므로(지표를 보고 지표에 맞춘 셈) 이 추정이 **독립 근거**가 된다. 실측 R\\* = 0.05
    (두 씬 일치) — `r_b` 제약이 만드는 하한(~0.10) 때문에 0.05를 그대로는 못 쓴다.

    **내장 대조군**: 후보 반경으로 만든 경로를 GT 자리에 넣으면 그 반경을 되찾아야 한다.
    `control_recovered`가 그 결과다 — 실측으로 0.10/0.15/0.25/0.40을 전부 정확히 복원했다.
    대조군이 실패하면 추정값을 신뢰할 수 없다(후보들이 구분되지 않는 에피소드에서 생긴다).
    """
    cands = list(candidates_m if candidates_m is not None else REFINE_CANDIDATES_M)
    probes = probe_radii_m if probe_radii_m is not None else REFINE_PROBE_RADII_M
    astar_wp = np.asarray(astar_waypoints_xy, dtype=np.float64)

    refined = {c: greedy_refine(astar_wp, esdf, origin, cell_m, c, fix_endpoints=False)
               for c in cands}
    cand_prof = {c: refine_displacement_profile(refined[c], esdf, origin, cell_m, probes)
                 for c in cands}
    gt_prof = refine_displacement_profile(gt_waypoints_xy, esdf, origin, cell_m, probes)

    def nearest(prof):
        l1 = {c: float(np.abs(cand_prof[c] - prof).sum()) for c in cands}
        return min(l1, key=l1.get), l1

    est, l1 = nearest(gt_prof)
    control = {c: nearest(cand_prof[c])[0] for c in cands}
    return {
        'estimate_m': est, 'l1_by_candidate': l1,
        'gt_profile': gt_prof.tolist(),
        'candidate_profiles': {c: cand_prof[c].tolist() for c in cands},
        'control_recovered': control,
        'control_ok': bool(all(control[c] == c for c in cands)),
        'probe_radii_m': list(probes),
    }


def recover_spline_knots(path_xy: np.ndarray, tol_m: float = 0.02, k_max: int = 40) -> dict:
    """**논문 7단계 역검증 (지표 ⑥)** — GT를 재현하는 최소 knot 수 `k*`와 그때의 waypoint 간격.

    7단계가 "refined waypoint를 cubic spline으로 스무딩"이면 GT는 **소수의 knot으로 재현된다.**
    호길이 등간격 k개 knot으로 cubic spline을 맞춰 GT와의 RMSE를 k = 3…`k_max`에서 재고,
    처음 `tol_m` 밑으로 떨어지는 `k_star`를 찾는다. `spacing_m = 길이 / (k*-1)`가 GT의 실제
    waypoint 간격 추정치다 — 우리 `--waypoint_spacing_m 0.8`의 **독립 근거**가 된다.

    **실측 결과: 40/40 에피소드에서 무릎을 찾았다**(k\\* median 7~9, 간격 0.79~0.90 m).
    즉 **GT는 논문 7단계 그대로 cubic spline이다** — 앞서 "GT는 컨트롤러 실행 궤적이라 점 분포를
    못 맞춘다"고 추정했던 것은 틀렸다.

    `has_knee=False`(어느 k에서도 tol 밑으로 안 감)이면 GT는 cubic spline이 아니라 컨트롤러 실행
    궤적이다. 그러면 점 단위 일치는 원리적으로 불가능하고, 남은 오차의 정체가 확정된다.

    **`k_star`는 상한이다** — 호길이 등간격 knot은 원래 knot 위치와 안 겹치므로 실제보다 크게 나온다
    (자체 테스트: 5-knot spline -> 11로 복원). 절대값보다 **경로 간 비교**와 `has_knee`가 신호다.
    """
    # **여기는 cubic 고정이다** — "GT가 논문 7단계(cubic spline)의 산출물인가"를 검정하는 것이므로
    # 우리 `--smooth` 선택과 무관하다. bezier로 바꾸면 검정의 의미가 사라진다.
    gt = resample_by_arclength(np.asarray(path_xy, dtype=np.float64), n=400)
    length = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    curve, k_star = {}, None
    for k in range(3, min(k_max, len(gt) - 1) + 1):
        knots = resample_by_arclength(gt, n=k)
        fit = resample_by_arclength(smooth_cubic_spline(knots, max(length / 400, 1e-4)), n=len(gt))
        rmse = float(np.sqrt(np.mean(np.sum((fit - gt) ** 2, axis=1))))
        curve[k] = rmse
        if k_star is None and rmse < tol_m:
            k_star = k
    return {
        'k_star': k_star, 'has_knee': k_star is not None,
        'spacing_m': length / (k_star - 1) if k_star and k_star > 1 else float('nan'),
        'length_m': length, 'rmse_curve': curve,
        'rmse_at_k_max': curve[max(curve)] if curve else float('nan'),
    }


def homotopy_obstacle_area(a_xy: np.ndarray, b_xy: np.ndarray, obstacle: np.ndarray,
                           origin: np.ndarray, cell_m: float = VOXEL_SIZE_M) -> float:
    """**지표 ⑦** 두 경로 사이에 갇힌 **장애물 면적**[m²] — "같은 통로인가"의 임계값 없는 판정.

    두 경로의 끝점을 이어 만든 닫힌 영역을 채우고, 그 안에 든 장애물 셀 면적을 잰다.
    같은 통로면 사이에 장애물이 없어 **정확히 0**이고, 다른 통로면 벽/가구가 끼어 0이 아니다.
    Fréchet 임계 0.8 m는 임의값인데 이건 0/1로 갈린다 — 실측에서 같은루트 39개 전부 0.000,
    다른루트 1개만 1.745 m²로 분리됐다(그 1개는 격자 원점을 월드 정렬로 바꿔 해결됐다).

    주 판정은 Fréchet 그대로 두고 이건 **보조 지표**로 함께 보고한다.

    경로 자체가 장애물을 지나면 도형이 납작해도 셀 겹침이 남아 면적이 잡힌다 — 의도된 동작이다.
    """
    import cv2

    poly = np.vstack([np.asarray(a_xy, dtype=np.float64)[:, :2],
                      np.asarray(b_xy, dtype=np.float64)[::-1, :2]])
    ij = world_to_cell(poly, origin, cell_m)
    h_grid, w_grid = obstacle.shape
    mask = np.zeros((h_grid, w_grid), dtype=np.uint8)
    cv2.fillPoly(mask, [ij.astype(np.int32)], 1)          # (ix, iy) 순서 = cv2의 (x, y)와 일치
    return float((mask.astype(bool) & obstacle).sum()) * cell_m * cell_m


# ---------------------------------------------------------------------------
# 좌표 변환 — 2D 격자는 arr[iy, ix]이고 y를 뒤집지 않는다 (모듈 docstring 참고)
# ---------------------------------------------------------------------------

def world_to_cell(xy: np.ndarray, origin: np.ndarray, cell_m: float = VOXEL_SIZE_M) -> np.ndarray:
    """(...,2) world xy -> (...,2) int (ix, iy)"""
    return ((np.asarray(xy, dtype=np.float64) - np.asarray(origin)[:2]) / cell_m).astype(int)


def cell_to_world(ij: np.ndarray, origin: np.ndarray, cell_m: float = VOXEL_SIZE_M) -> np.ndarray:
    """(...,2) int (ix, iy) -> (...,2) world xy (셀 중심)"""
    return (np.asarray(ij, dtype=np.float64) + 0.5) * cell_m + np.asarray(origin)[:2]


def sample_esdf_at(esdf: np.ndarray, xy: np.ndarray, origin: np.ndarray,
                   cell_m: float = VOXEL_SIZE_M) -> np.ndarray:
    """world xy 위치들의 clearance[m]를 뽑는다 (격자 밖은 0)."""
    ij = world_to_cell(xy, origin, cell_m)
    h, w = esdf.shape
    inside = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    out = np.zeros(len(ij), dtype=np.float32)
    out[inside] = esdf[ij[inside, 1], ij[inside, 0]]
    return out


# ---------------------------------------------------------------------------
# self-check — `python esdf_utils.py`  (게이트 11개)
#
# [1] 순수 로직  [1b] 경로 계획  [1d] 충돌 검사  [1f] refine 모드  [1g] A* 파라미터  [1c] 보조 지표
# [1e] 논문 역검증  [2] voxelize  [3] GT clearance  [4] 바닥 탐지  [5] USD vs obj
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pyarrow.parquet as pq
    from geometry_utils import action_to_c2w, decompose_camera_extrinsic, load_scene_mesh

    _DATA_ROOT = Path('data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i')
    _MESH_ROOT = Path('data/scene_data/mp3d_n1')
    _USD_ROOT = Path('data/scene_data/mp3d_pe')
    _SCENE = '17DRP5sb8fy'

    # ---- [1] 순수 로직 — 데이터 없이도 도는 것들 ----
    _obst = np.zeros((20, 20), dtype=bool)
    _obst[10, 10] = True
    _esdf = compute_esdf_2d(_obst, 0.05)
    assert _esdf[10, 10] == 0.0, 'obstacle 셀의 ESDF는 0이어야 한다'
    assert abs(_esdf[10, 14] - 0.20) < 1e-6, f'4셀 떨어진 곳은 0.20 m여야 한다 (얻은 {_esdf[10, 14]})'
    _nav = truncate_navigable(_esdf, 0.25)
    assert _nav.dtype == bool and not _nav[10, 10], 'truncate는 bool 집합을 반환해야 한다(값 clip 아님)'
    assert not _nav[10, 14], 'clearance 0.20 < r_b 0.25 이므로 non-navigable이어야 한다'
    _ds = downsample_navigable(np.ones((8, 8), dtype=bool), 4)
    assert _ds.shape == (2, 2) and _ds.all(), 'downsample shape'
    _one_bad = np.ones((8, 8), dtype=bool); _one_bad[0, 0] = False
    assert downsample_navigable(_one_bad, 4)[0, 0], \
        'any()는 낙관적이라 블록에 하나 막혀도 navigable이어야 한다 (문틈 보존)'
    _blk_bad = np.ones((8, 8), dtype=bool); _blk_bad[0:4, 0:4] = False
    assert not downsample_navigable(_blk_bad, 4)[0, 0], '블록 전체가 막히면 non-navigable'
    assert not downsample_navigable(_one_bad, 4, 'all')[0, 0], 'all()은 보수적이어야 한다'
    assert downsample_navigable(_one_bad, 4, 'majority')[0, 0], 'majority: 15/16이면 통과'
    print('[1/5] 순수 로직 (ESDF 거리 / truncate 의미 / downsample any·majority·all) OK')

    # ---- [1b] 경로 계획 ----
    _room = np.zeros((40, 40), dtype=bool)
    _room[1:19, 1:39] = True; _room[21:39, 1:39] = True     # 방 2개
    _room[19:21, 20] = True                                  # 유일한 문
    _p = astar(_room, (5, 5), (35, 35))
    assert _p is not None and tuple(_p[0]) == (5, 5) and tuple(_p[-1]) == (35, 35), 'A* 양끝'
    assert np.abs(np.diff(_p, axis=0)).max() <= 1, 'A* 경로에 점프가 있다'
    assert any(j == 30 and i == 20 for i, j in _p) or any(i == 20 for i, j in _p), \
        'A*가 유일한 문을 지나지 않았다'
    _room[19:21, 20] = False                                 # 문을 막으면 실패해야 한다
    assert astar(_room, (5, 5), (35, 35)) is None, '통로가 없는데 A*가 경로를 냈다'
    assert astar(np.ones((5, 5), dtype=bool), (0, 0), (99, 99)) is None, '격자 밖 goal은 None'

    # refine: 벽에 붙은 waypoint가 안쪽으로 밀려나 clearance가 올라가야 한다
    _obst = np.zeros((60, 60), dtype=bool); _obst[:, 0] = True
    _e = compute_esdf_2d(_obst, 0.05)
    _org0 = np.zeros(3)
    _wp = cell_to_world(np.array([[1, 10], [1, 30], [1, 50]]), _org0, 0.05)
    _rf = greedy_refine(_wp, _e, _org0, 0.05, radius_m=0.5, fix_endpoints=False)
    _c0 = sample_esdf_at(_e, _wp, _org0, 0.05); _c1 = sample_esdf_at(_e, _rf, _org0, 0.05)
    assert _c1.min() > _c0.min(), f'refine이 clearance를 못 올렸다 ({_c0.min():.3f} -> {_c1.min():.3f})'

    # 스무딩 2종: 길이가 원래 waypoint 길이와 비슷하고, Bézier는 오버슈트가 없다(convex hull 안)
    _corner = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    for _name, _fn in [('cubic', smooth_cubic_spline), ('bezier', smooth_bezier)]:
        _sm = _fn(_corner, 0.02)
        assert np.allclose(_sm[0], _corner[0]) and np.allclose(_sm[-1], _corner[-1]), f'{_name} 양끝'
        _len = np.linalg.norm(np.diff(_sm, axis=0), axis=1).sum()
        assert 0.5 * 2.0 < _len < 1.5 * 2.0, f'{_name} 길이 {_len:.2f}가 원래 2.0과 너무 다르다'
    _bz = smooth_bezier(_corner, 0.02)
    assert _bz[:, 0].max() <= 1.0 + 1e-9 and _bz[:, 1].min() >= -1e-9, 'Bézier가 convex hull을 벗어났다'
    assert smooth_cubic_spline(_corner, 0.02)[:, 1].min() < -1e-6, \
        'cubic spline은 이 코너에서 오버슈트해야 한다 (Bézier와의 차이를 확인하는 테스트)'

    # thin_waypoints: 간격 유지 + 양 끝 보존
    _line = np.stack([np.linspace(0, 4, 21), np.zeros(21)], axis=1)   # 0.2 m 간격 21점
    _th = thin_waypoints(_line, 0.8)
    assert np.allclose(_th[0], _line[0]) and np.allclose(_th[-1], _line[-1]), 'thin 양끝 보존'
    assert 5 <= len(_th) <= 7, f'0.2m x 21점을 0.8m로 솎으면 6개 근처여야 한다 (얻은 {len(_th)})'
    assert len(thin_waypoints(_line, 0.0)) == len(_line), 'spacing 0이면 그대로'

    # 비교 지표
    _a = np.stack([np.linspace(0, 1, 50), np.zeros(50)], axis=1)
    assert chamfer_distance(_a, _a) < 1e-9 and discrete_frechet_distance(_a, _a) < 1e-9, '자기 거리 0'
    _b = _a + np.array([0.0, 0.3])
    assert abs(chamfer_distance(_a, _b) - 0.3) < 0.05, 'chamfer 평행이동 0.3'
    assert abs(discrete_frechet_distance(_a, _b) - 0.3) < 1e-6, 'Fréchet 평행이동 0.3'
    print('[1b/5] 경로 계획 (A* / refine / cubic·bezier / thin / chamfer·Fréchet) OK')

    # ---- [1d] 충돌 검사: 하드 충돌은 장애물을 관통할 때만 (점만 보면 사이를 놓친다) ----
    _obs_w = np.zeros((40, 40), dtype=bool); _obs_w[18:22, 0:20] = True      # 왼쪽에서 뻗은 벽
    _ew = compute_esdf_2d(_obs_w, 0.05)
    _thru = np.array([[0.25, 0.6], [0.25, 1.4]])                             # 벽을 관통
    _around = np.array([[0.25, 0.6], [1.3, 0.6], [1.3, 1.4], [0.25, 1.4]])   # 벽 끝을 돌아감
    assert not check_path_navigable(_thru, _ew, _org0, 0.05)['hard_ok'], '관통 경로를 못 잡았다'
    assert check_path_navigable(_around, _ew, _org0, 0.05)['hard_ok'], '우회 경로를 충돌로 오판했다'
    assert check_path_navigable(_thru, _ew, _org0, 0.05, sample_step_m=10.0)['hard_ok'], \
        '샘플 간격이 크면 관통을 놓친다 — 기본값(cell/5)이 촘촘해야 하는 이유'
    print('[1d/5] 충돌 검사 (선분 관통 탐지 / 샘플 간격 민감도) OK')

    # ---- [1f] refine 모드: "필요한 만큼만" 옮긴다 ----
    _mm_wp = cell_to_world(np.array([[2, 10], [10, 30], [2, 50]]), _org0, 0.05)  # 0.10/0.50/0.10 m
    _mm = refine_min_move(_mm_wp, _e, _org0, 0.05, 0.30, fix_endpoints=False, r_b=0.25)
    _mv = np.linalg.norm(_mm - _mm_wp, axis=1)
    assert _mv[1] < 1e-9, f'r_b를 이미 만족하는 waypoint를 움직였다 ({_mv[1]:.4f} m)'
    assert sample_esdf_at(_e, _mm, _org0, 0.05).min() >= 0.25 - 1e-9, 'min_move 후 r_b 미달이 남았다'
    _am = greedy_refine(_mm_wp, _e, _org0, 0.05, 0.30, fix_endpoints=False)
    assert np.linalg.norm(_am - _mm_wp, axis=1).sum() > _mv.sum(), \
        'min_move가 argmax보다 더 움직였다 (이름과 반대)'
    _mm_narrow = refine_min_move(_mm_wp, _e, _org0, 0.05, 0.05, fix_endpoints=False, r_b=0.25)
    assert np.linalg.norm(_mm_narrow - _mm_wp, axis=1)[0] > 0, '좁은 창에서 폴백이 안 걸렸다'
    assert set(REFINERS) == {'argmax', 'min_move'}, 'REFINERS registry 키가 바뀌었다'
    assert set(SMOOTHERS) == {'cubic', 'bezier'}, 'SMOOTHERS registry 키가 바뀌었다'
    print('[1f/5] refine 모드 (min_move 최소이동 / argmax 대비 / r_b 달성 / 폴백) OK')

    # ---- [1g] A* 파라미터: connectivity / clearance tie-break 가중치 ----
    _diag_room = np.ones((10, 10), dtype=bool)
    _p8 = astar(_diag_room, (0, 0), (9, 9), connectivity=8)
    _p4 = astar(_diag_room, (0, 0), (9, 9), connectivity=4)
    assert len(_p8) < len(_p4), '8-이웃이 4-이웃보다 대각선을 써서 더 짧아야 한다'
    assert all(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) == 1
              for a, b in zip(_p4[:-1], _p4[1:])), '4-이웃 경로에 대각 이동이 섞였다'

    # 장애물 하나를 위(row1)/아래(row3)로 피해가는 대칭 지도 — clearance 편향의 no-op/효과를 검증
    _bias_room = np.ones((5, 11), dtype=bool)
    _bias_room[2, 5] = False
    _clr = np.zeros((5, 11), dtype=np.float32)
    _clr[1, :], _clr[3, :] = 100.0, 0.0
    _p_nobias = astar(_bias_room, (0, 2), (10, 2))
    _p_bias0 = astar(_bias_room, (0, 2), (10, 2), clearance=_clr, clearance_weight=0.0)
    assert np.array_equal(_p_nobias, _p_bias0), 'clearance_weight=0이면 clearance=None과 완전히 같아야 한다'
    _p_bias = astar(_bias_room, (0, 2), (10, 2), clearance=_clr, clearance_weight=5.0)
    assert not any(int(iy) == 3 for ix, iy in _p_bias), 'clearance bias가 낮은 row(3)를 여전히 지난다'
    assert any(int(iy) == 1 for ix, iy in _p_bias), 'clearance bias가 높은 row(1)로 안 갔다'
    print('[1g/5] A* 파라미터 (connectivity 4/8-이웃 / clearance tie-break no-op·효과) OK')

    # ---- [1h] compute_scan_coverage_mask: 전체 높이에서 점유가 0인 열만 False여야 한다 ----
    _occ_cov = np.zeros((4, 3, 5), dtype=bool)   # (Nx=4, Ny=3, Nz=5)
    _occ_cov[0, 0, 2] = True    # (ix=0,iy=0) 열은 z=2에만 점유 -> covered
    _occ_cov[2, 1, :] = False   # (ix=2,iy=1) 열은 전체 높이 무점유 -> not covered
    _cov = compute_scan_coverage_mask(_occ_cov)
    assert _cov.shape == (3, 4), f'compute_scan_coverage_mask shape 오류: {_cov.shape}'
    assert bool(_cov[0, 0]) is True, '점유가 있는 열이 covered=False로 나왔다'
    assert bool(_cov[1, 2]) is False, '전체 높이 무점유 열이 covered=True로 나왔다'
    print('[1h/5] compute_scan_coverage_mask (전체 높이 무점유 열 판별) OK')

    # ---- [1c] 보조 지표 ①②③ ----
    _dense = np.stack([np.linspace(0, 3, 300), np.zeros(300)], axis=1)
    _sparse = np.stack([np.array([0.0, 0.1, 2.9, 3.0]), np.zeros(4)], axis=1)
    assert np.allclose(resample_by_arclength(_dense, n=50), resample_by_arclength(_sparse, n=50),
                       atol=1e-6), '리샘플이 점 분포에 의존한다'
    assert path_smoothness(_dense)['turn_per_m_deg'] < 1e-6, '직선 회전각 0'
    _zig = np.stack([np.arange(41) * 0.1, 0.1 * (np.arange(41) % 2)], axis=1)
    _sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])            # 90도 코너 1개 / 총 2 m
    assert path_smoothness(_zig)['turn_per_m_deg'] > 10 * path_smoothness(_sq)['turn_per_m_deg'], \
        '지그재그가 완만한 코너보다 회전각이 커야 한다'
    assert abs(path_smoothness(_sq)['total_turn_deg'] - 90.0) < 5.0, '90도 코너 1개면 총 회전 90도'
    assert abs(path_smoothness(_sq)['turn_per_m_deg'] - 45.0) < 3.0, '90도 / 2 m = 45 도/m'
    assert abs(compare_clearance_profile(_wp, _wp, _e, _org0, 0.05)['median_diff_m']) < 1e-9, \
        '같은 경로 clearance 차이 0'
    assert compare_clearance_profile(_rf, _wp, _e, _org0, 0.05)['median_diff_m'] > 0, \
        'refine된 쪽이 벽에서 더 멀어야 한다(양수)'
    assert heading_alignment(_dense, _dense)['median_deg'] < 1e-6, '같은 경로 헤딩 0도'
    assert abs(heading_alignment(_dense, _dense[::-1])['median_deg'] - 180.0) < 1e-3, '역주행 180도'
    _diag = np.stack([np.linspace(0, 3, 300), np.linspace(0, 3, 300)], axis=1)
    assert abs(heading_alignment(_dense, _diag)['median_deg'] - 45.0) < 1e-3, '45도 어긋남'
    print('[1c/5] 보조 지표 ① 스무드니스 / ② clearance 프로파일 / ③ 헤딩 OK')

    # ---- [1e] 논문 역검증 ----
    # 바닥값 F3 = 격자 양자화. **셀 중심 위의 직선은 0**, 셀 경계 위의 직선은 반셀(0.1 m)이다.
    _line_center = np.stack([np.linspace(0, 4, 200), np.full(200, 0.1)], axis=1)
    _fl_center = metric_floors(_line_center)
    assert _fl_center['f3_chamfer_m'] < 0.02, \
        f"셀 중심 위 직선의 F3는 0이어야 한다 ({_fl_center['f3_chamfer_m']:.3f})"
    _fl_edge = metric_floors(np.stack([np.linspace(0, 4, 200), np.zeros(200)], axis=1))
    assert abs(_fl_edge['f3_chamfer_m'] - 0.1) < 0.02, \
        f"셀 경계 위 직선의 F3는 반셀(0.10)이어야 한다 ({_fl_edge['f3_chamfer_m']:.3f})"
    assert _fl_center['f2_chamfer_m'] < VOXEL_SIZE_M / 2, '등간격 경로의 F2는 샘플 간격 절반 미만'
    _t3 = np.linspace(0, 1, 200) ** 3
    assert metric_floors(np.stack([_t3 * 4, np.full(200, 0.1)], axis=1))['f2_chamfer_m'] \
        > _fl_center['f2_chamfer_m'], '불균등 샘플에서 F2가 커져야 한다'

    # 기준선 ⓒ: 격자 위 직선은 파이프라인을 통과해도 그대로, refine을 끼우면 벽에서 밀려난다
    _fcor = np.zeros((60, 60), dtype=bool); _fcor[:, 0] = _fcor[:, 40] = True
    _efc = compute_esdf_2d(_fcor, 0.05)
    _mid = cell_to_world(np.stack([np.full(40, 20), np.arange(10, 50)], axis=1), _org0, 0.05)
    _fl_b = pipeline_floor_path(_mid, _efc, _org0, 0.05, 0.2, with_refine=False)
    assert chamfer_distance(_fl_b, _mid) < 0.11, '복도 중심선이 반셀 이상 벗어났다'
    _near = cell_to_world(np.stack([np.full(40, 4), np.arange(10, 50)], axis=1), _org0, 0.05)
    _fl_c = pipeline_floor_path(_near, _efc, _org0, 0.05, 0.2, refine_radius_m=0.30)
    assert sample_esdf_at(_efc, _fl_c, _org0, 0.05).mean() > \
        sample_esdf_at(_efc, _near, _org0, 0.05).mean(), 'refine을 끼운 기준선이 벽에서 안 멀어졌다'
    assert chamfer_distance(_fl_c, _near) > chamfer_distance(
        pipeline_floor_path(_near, _efc, _org0, 0.05, 0.2, with_refine=False), _near), \
        'refine이 기준선을 GT에서 더 멀어지게 해야 한다 (실측 0.052 -> 0.080)'
    # 기준선이 스무딩/refine 모드를 실제로 반영하는가 — 하드코딩되면 배수 비교가 무의미해진다
    _fl_bz = pipeline_floor_path(_near, _efc, _org0, 0.05, 0.2, smooth_mode='bezier')
    _n = min(len(_fl_bz), len(_fl_c))
    assert not np.allclose(_fl_bz[:_n], _fl_c[:_n]), 'smooth_mode=bezier가 기준선에 반영되지 않았다'
    _fl_mm = pipeline_floor_path(_near, _efc, _org0, 0.05, 0.2, refine_mode='min_move')
    _n = min(len(_fl_mm), len(_fl_c))
    assert not np.allclose(_fl_mm[:_n], _fl_c[:_n]), 'refine_mode=min_move가 기준선에 반영되지 않았다'

    # 셀 시퀀스: 인접 셀만 남아야 한다 (점프를 메우는지)
    _cs = path_cell_sequence(np.array([[0.0, 0.0], [3.0, 0.0]]), _org0, 0.2)
    assert np.abs(np.diff(_cs, axis=0)).max() <= 1, '셀 시퀀스에 점프가 남았다'
    assert len(_cs) >= 15, f'0.2 m 셀로 3 m면 15칸 이상이어야 한다 (얻은 {len(_cs)})'

    # ④ A* 비용 갭: 우회를 강제하는 벽에서 GT가 직선이면 막힌 것으로 잡혀야 한다
    _nav_g = np.ones((20, 20), dtype=bool); _nav_g[8:12, 0:10] = False
    _our = astar(_nav_g, (2, 5), (2, 15))
    assert _our is not None, '테스트 맵에서 A*가 우회로를 찾아야 한다'
    _gap = astar_cost_gap(cell_to_world(np.stack([np.full(40, 2), np.linspace(5, 15, 40)], axis=1),
                                        _org0, 0.2), _our, _nav_g, _org0, 0.2)
    assert _gap['n_blocked'] > 0, '벽을 관통하는 GT를 막힌 것으로 못 잡았다'
    _gap_same = astar_cost_gap(cell_to_world(_our, _org0, 0.2), _our, _nav_g, _org0, 0.2)
    assert _gap_same['n_blocked'] == 0 and abs(_gap_same['cost_ratio'] - 1.0) < 0.05, \
        f"자기 자신과의 비용비는 1이어야 한다 ({_gap_same['cost_ratio']:.3f})"
    # 매끄러운 45도 대각선도 1에 가까워야 한다 — 셀 시퀀스 비용을 직접 세면 여기서 1.41이 나온다
    _diag_nav = np.ones((30, 30), dtype=bool)
    _diag_our = astar(_diag_nav, (2, 2), (25, 25))
    _diag_gt = cell_to_world(np.stack([np.linspace(2, 25, 300)] * 2, axis=1), _org0, 0.2)
    assert abs(astar_cost_gap(_diag_gt, _diag_our, _diag_nav, _org0, 0.2)['cost_ratio'] - 1.0) < 0.05, \
        '45도 대각선의 비용비가 1이 아니다 — 계단 비용을 세고 있다'

    # ⑤ refine 반경 역추정: 알려진 반경으로 만든 경로에서 그 반경을 되찾는가 (내장 대조군)
    _cor = np.zeros((60, 60), dtype=bool); _cor[:, 0] = _cor[:, 20] = _cor[:, 50] = True
    _ecor = compute_esdf_2d(_cor, 0.05)
    _wall_wp = cell_to_world(np.stack([np.full(24, 3), np.arange(18, 42)], axis=1), _org0, 0.05)
    _cands = (0.10, 0.20, 0.40)
    for _r in _cands:
        _made = greedy_refine(_wall_wp, _ecor, _org0, 0.05, _r, fix_endpoints=False)
        _est = estimate_refine_radius(_made, _wall_wp, _ecor, _org0, 0.05, candidates_m=_cands)
        assert _est['estimate_m'] == _r, \
            f"반경 {_r}로 만든 경로에서 {_est['estimate_m']}를 추정했다 (대조군 실패)"
    assert _est['control_ok'], f"내장 대조군이 자기 반경을 못 되찾았다 ({_est['control_recovered']})"
    _p10 = refine_displacement_profile(
        greedy_refine(_wall_wp, _ecor, _org0, 0.05, 0.10, False), _ecor, _org0, 0.05)
    _p40 = refine_displacement_profile(
        greedy_refine(_wall_wp, _ecor, _org0, 0.05, 0.40, False), _ecor, _org0, 0.05)
    assert _p40[-1] < _p10[-1], f'넓게 refine한 경로가 더 움직인다 ({_p40[-1]:.3f} vs {_p10[-1]:.3f})'

    # ⑥ knot 복원: 등간격 knot이라 k_star는 상한이다 — 무릎의 존재와 경로 간 순서를 본다
    _rec_line = recover_spline_knots(np.stack([np.linspace(0, 4, 200), np.zeros(200)], axis=1))
    assert _rec_line['k_star'] == 3, f"직선은 최소 knot(3)으로 재현된다 (얻은 {_rec_line['k_star']})"
    _rec_arc = recover_spline_knots(smooth_cubic_spline(
        np.array([[0.0, 0.0], [2.0, 0.4], [4.0, 0.0]]), 0.01), tol_m=0.02, k_max=30)
    _rec_s = recover_spline_knots(smooth_cubic_spline(
        np.array([[0.0, 0.0], [1.0, 0.8], [2.0, 0.0], [3.0, 0.8], [4.0, 0.0]]), 0.01),
        tol_m=0.02, k_max=30)
    assert _rec_arc['has_knee'] and _rec_s['has_knee'], 'cubic spline 입력인데 무릎을 못 찾았다'
    assert _rec_arc['k_star'] < _rec_s['k_star'], \
        f"완만한 곡선이 적은 knot으로 재현돼야 한다 ({_rec_arc['k_star']} vs {_rec_s['k_star']})"
    assert 0.1 < _rec_s['spacing_m'] < 4.0, f"knot 간격이 경로 길이 범위 안 ({_rec_s['spacing_m']:.3f})"
    _rng = np.random.default_rng(0)
    _jag = np.stack([np.linspace(0, 4, 200), _rng.normal(0, 0.15, 200)], axis=1)
    assert not recover_spline_knots(_jag, tol_m=0.02, k_max=30)['has_knee'], \
        '난잡한 경로에서 무릎이 잡혔다 — 이 검사가 spline 여부를 구분하지 못한다는 뜻'

    # ⑦ 위상: 같은 경로면 0, 벽을 사이에 두고 갈리면 > 0
    _obs_h = ~_nav_g
    _left = cell_to_world(np.stack([np.full(40, 2), np.linspace(5, 15, 40)], axis=1), _org0, 0.2)
    _right = cell_to_world(_our, _org0, 0.2)
    assert homotopy_obstacle_area(_right, _right, _obs_h, _org0, 0.2) == 0.0, \
        '통행 가능한 같은 경로끼리는 면적 0이어야 한다'
    assert homotopy_obstacle_area(_left, _right, _obs_h, _org0, 0.2) > 0.0, \
        '벽을 사이에 둔 두 경로인데 면적이 0이다'
    assert homotopy_obstacle_area(_left, _left, _obs_h, _org0, 0.2) > 0.0, \
        '장애물을 관통하는 경로는 자기 자신과 비교해도 면적이 잡혀야 한다'
    print('[1e/5] 논문 역검증 (바닥값 / 기준선ⓒ / 셀시퀀스 / ④비용갭 / ⑤반경 / ⑥knot / ⑦위상) OK')

    if not (_DATA_ROOT / _SCENE).is_dir() or not (_MESH_ROOT / _SCENE).is_dir():
        print(f'[2-5/5] SKIP (데이터 없음: {_DATA_ROOT / _SCENE})')
        print('[esdf_utils self-check] 통과 (로직만)')
        sys.exit(0)

    # ---- [2] 좌표 왕복 ----
    _mesh = load_scene_mesh(_MESH_ROOT, _SCENE)
    _occ, _origin = voxelize_surface(_mesh, align_to_m=ASTAR_CELL_M)
    # `np.mod(-11.6, 0.2)`는 부동소수 오차로 0이 아니라 0.2 쪽에 붙는다 — **가장 가까운** 격자선까지의
    # 거리로 봐야 한다. 0에 가까운지만 보면 정렬된 원점을 실패로 잡는다(실제로 겪음).
    _phase = np.mod(_origin, ASTAR_CELL_M)
    assert np.minimum(_phase, ASTAR_CELL_M - _phase).max() < 1e-9, \
        f'align_to_m을 줬는데 원점이 월드 격자에 안 붙었다 ({_origin}, 위상 {_phase})'
    _xy = np.array([[1.0, 2.0], [-5.0, 3.5]])
    _rt = cell_to_world(world_to_cell(_xy, _origin), _origin)
    assert np.abs(_rt - _xy).max() < VOXEL_SIZE_M, '좌표 왕복 오차가 셀 크기를 넘는다'
    print(f'[2/5] voxelize + 좌표 왕복 + 월드 정렬 OK (grid={_occ.shape}, 점유 {100 * _occ.mean():.1f}%)')

    # ---- [3] GT 궤적 clearance 회귀 게이트 — 이 모듈의 절대 기준선 ----
    _t = pq.read_table(_DATA_ROOT / _SCENE / 'data' / 'chunk-000' / 'episode_000000.parquet')
    _a = np.stack([np.asarray(x, dtype=np.float64).reshape(4, 4) for x in _t['action'].to_pylist()])
    _ex = np.asarray(_t['observation.camera_extrinsic'].to_pylist()[0], dtype=np.float64).reshape(4, 4)
    _h_b, _ = decompose_camera_extrinsic(_ex)
    _cam = np.stack([action_to_c2w(m, 'cam2world_gl')[:3, 3] for m in _a])
    _floor_z = float(np.median(_cam[:, 2]) - _h_b)

    _obstacle = derive_obstacle_2d(_occ, _origin, _floor_z, _h_b)
    _esdf_gt = compute_esdf_2d(_obstacle)
    _clear = sample_esdf_at(_esdf_gt, _cam[:, :2], _origin)
    _med = float(np.median(_clear))
    _lo, _hi = GT_CLEARANCE_MEDIAN_RANGE_M
    assert _lo < _med < _hi, (
        f'GT 궤적 clearance median={_med:.3f} m가 범위 ({_lo}, {_hi}) 밖이다. 맵 파이프라인 확인.')
    assert _clear.min() > GT_CLEARANCE_MIN_M, f'GT 궤적 최소 clearance={_clear.min():.3f} m가 너무 작다'
    if not (GT_CLEARANCE_TYPICAL_RANGE_M[0] <= _med <= GT_CLEARANCE_TYPICAL_RANGE_M[1]):
        print(f'      [경고] median {_med:.3f}이 통상 범위 {GT_CLEARANCE_TYPICAL_RANGE_M} 밖 (판정 아님)')
    print(f'[3/5] GT 궤적 clearance OK (h_b={_h_b:.3f} floor_z={_floor_z:+.3f} '
          f'median={_med:.3f} min={_clear.min():.3f} m)')

    # ---- [4] 바닥 탐지 fallback (Stage 3에서만 쓰이지만 회귀는 막는다) ----
    _floors = detect_floor_levels(_occ, _origin)
    assert len(_floors) > 0, '바닥 후보를 하나도 못 찾았다'
    assert abs(float(_floors[0]) - _floor_z) < 0.1, (
        f'바닥 탐지 1순위 {_floors[0]:.3f} m가 GT floor_z {_floor_z:.3f} m와 0.1 m 넘게 다르다 '
        f'(후보 {np.round(_floors[:5], 2)})')
    print(f'[4/5] 바닥 탐지 fallback OK (1순위 {_floors[0]:+.3f} vs GT {_floor_z:+.3f} m, '
          f'후보 {len(_floors)}개)')

    # ---- [5] USD vs obj — USD로 바꿔도 지오메트리를 잃지 않는가 ----
    #      표면 샘플링이 무작위라 셀 경계에서 XOR이 조금 생기는 건 정상이므로, "obj에만 있는 voxel이
    #      전부 USD 표면에 인접한가"로 판정한다 — 인접하면 경계 노이즈, 떨어져 있으면 진짜 누락이다.
    _usd_files = sorted((_USD_ROOT / _SCENE / 'matterport_mesh').glob('*/fixed.usd'))
    if _usd_files:
        from scipy.ndimage import binary_dilation

        _occ_usd, _origin_usd = voxelize_surface(load_scene_usd(_usd_files[0]),
                                                 align_to_m=ASTAR_CELL_M)
        assert np.abs(_origin_usd - _origin).max() < 1e-6, 'USD/obj origin 불일치'
        assert _occ_usd.shape == _occ.shape, f'shape 불일치 {_occ_usd.shape} vs {_occ.shape}'

        _obj_only = _occ & ~_occ_usd
        _usd_only = _occ_usd & ~_occ
        _orphan_obj = int((_obj_only & ~binary_dilation(_occ_usd)).sum())
        _orphan_usd = int((_usd_only & ~binary_dilation(_occ)).sum())
        _limit = max(20, int(1e-4 * _occ.sum()))
        assert _orphan_obj <= _limit, (
            f'obj에만 있고 USD 표면에서 떨어진 voxel {_orphan_obj}개 (허용 {_limit}) — '
            f'USD가 실제 지오메트리를 잃었을 수 있다. 02의 --geometry를 obj로 두고 조사할 것.')
        print(f'[5/5] USD vs obj OK (USD만 {int(_usd_only.sum())} / obj만 {int(_obj_only.sum())}; '
              f'고립 voxel obj-only {_orphan_obj}<={_limit} = 누락 없음, '
              f'usd-only {_orphan_usd} = collision Plane 추가분)')
    else:
        print('[5/5] USD vs obj SKIP (fixed.usd 없음)')

    print('[esdf_utils self-check] 전부 통과')
