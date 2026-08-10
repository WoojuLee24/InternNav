"""gs_vlnpe 파이프라인 공용 데이터셋 로더 — vln_n1 / vln_pe 두 포맷을 하나의 인터페이스로.

00~04 스크립트가 각자 인라인으로 들고 있던 vln_n1 전용 로딩 코드(`load_gt_episode` 4중복,
`load_rgb_frame`/`load_depth_frame_m`, `DEFAULT_DATA_ROOT` 8중복)를 이 모듈로 모으고,
`--dataset vln_pe`를 지원한다. 새 스크립트는 parquet을 직접 읽지 말고 반드시 여기를 거칠 것.

## 두 포맷의 차이 (전부 실제 데이터에서 실측 확인, 2026-08-07)

|              | vln_n1                                | vln_pe                                       |
|--------------|---------------------------------------|----------------------------------------------|
| GT pose      | `action` 컬럼 (4x4 OpenGL c2w)         | `camera_position`(3) + `camera_orientation`(wxyz 쿼터니언) |
| intrinsic    | `observation.camera_intrinsic` 컬럼    | 컬럼 없음 — USD 실측: 90° hfov 256x256 → fx=fy=cx=cy=128 |
| RGB          | 프레임별 .jpg                           | 에피소드당 .npy 스택 (T,256,256,3) uint8       |
| depth        | 프레임별 .png (raw/10000 = m)           | 에피소드당 .npy 스택 float32, m = value*10, 1.0=10m 클립 |
| world 좌표    | mp3d mesh 프레임 (Z-up)                | 동일 (house bbox 안에 궤적 들어감 실측 확인)     |

vln_pe 함정: `meta/info.json`이 vln_n1 스키마를 그대로 베낀 stale 메타데이터다(camera_intrinsic/
camera_extrinsic 컬럼이 있다고 주장하지만 실제 parquet엔 없음) — 절대 신뢰하지 말 것.

## vln_pe pose -> OpenCV 변환

`camera_orientation`은 unit-norm **(w,x,y,z)** 쿼터니언이고(2*atan2(z,w)가 `camera_yaw`를 1e-6
정확도로 재현함을 실측), R(q)의 컬럼이 world 프레임 기준 [forward, left, up]이다(Isaac
`Camera.get_world_pose(camera_axes="world")` 컨벤션). 15° 하향 마운트 + 토르소 피치가 포함된
완전한 3D 방향이므로 `camera_yaw`만으로 재구성하면 안 된다(피치 15~21°를 잃음). OpenCV
(x=right, y=down, z=forward)로는:
    R_cv 컬럼 = [-left, -up, forward]  즉  R_cv = R[:, [1, 2, 0]] * [-1, -1, +1]
이 부호는 00_inspect의 mesh-anchor 검증(GT depth를 이 pose로 world에 올려 mesh 표면거리 측정)
으로 실측 확정할 것 — 유도만 믿지 않는다(세션 반복 교훈).

실행(`python dataset_utils.py`)하면 자체 회귀 체크: vln_n1 로더가 기존 인라인 구현과 동일한
값을 내는지 + vln_pe 쿼터니언/마운트 오프셋 실측 관계가 성립하는지 확인한다.
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from geometry_utils import (
    action_to_c2w,
    decompose_camera_extrinsic,
)
from geometry_utils import load_depth_frame_m as _load_depth_frame_m_n1
from geometry_utils import load_rgb_frame as _load_rgb_frame_n1

# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

DATASETS = {
    'vln_n1': dict(
        data_root='data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i',
        render_wh=(480, 270),
        tag='',   # 출력 경로 접미사 — vln_n1은 기존 경로 그대로(하위호환)
    ),
    'vln_pe': dict(
        data_root='data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r',
        render_wh=(256, 256),
        tag='_vlnpe',
    ),
}

# vln_pe 카메라 intrinsic — parquet에 없음. h1_vln_pointcloud.usd의 h1_pano_camera_0에서 실측:
# focalLength=10.4775, horizontalAperture=20.955 → hfov 정확히 90°, 렌더 256x256 정사각.
# fx = 128/tan(45°) = 128.0.
VLNPE_K = np.array([[128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]])
VLNPE_IMG_WH = (256, 256)
# vln_pe depth npy는 0~1 정규화(내부 코드 internnav/evaluator/utils/common.py:193 norm_depth,
# max_depth=10) — m = value * 10, 1.0은 "10m 이상 클립"이므로 invalid로 취급.
VLNPE_DEPTH_SCALE_M = 10.0
# H1 로봇이 서 있을 때 torso_link의 바닥 기준 높이. 실측: 씬1(바닥 z≈0)에서 robot_position z가
# 0.941~0.957. floor_z/h_b 유도용 — 보행 흔들림 수준(±수 cm)의 근사치이며, 03/04의 실제 카메라
# 높이 합성은 ESDF의 floor_z를 쓰므로 이 상수의 오차는 2-B 경로의 카메라 높이에만 소폭 반영된다.
VLNPE_TORSO_STANDING_M = 0.95


def dataset_tag(dataset: str) -> str:
    """출력 경로(로그/paths/obs/esdf)에 붙일 접미사. vln_n1=''(기존 경로 유지), vln_pe='_vlnpe'."""
    return DATASETS[dataset]['tag']


def default_data_root(dataset: str) -> str:
    return DATASETS[dataset]['data_root']


def render_wh(dataset: str) -> tuple:
    return DATASETS[dataset]['render_wh']


# ---------------------------------------------------------------------------
# GT 에피소드 로더
# ---------------------------------------------------------------------------

def count_episodes(data_root: str, scene: str) -> int:
    """두 데이터셋 모두 LeRobot 레이아웃이 같아 dataset 구분 불필요."""
    return len(sorted((Path(data_root) / scene / 'data' / 'chunk-000').glob('episode_*.parquet')))


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) -> (3,3). scipy는 xyzw 순서를 받으므로 재배열."""
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def vlnpe_orientation_to_c2w_rotation(q_wxyz: np.ndarray) -> np.ndarray:
    """vln_pe `camera_orientation` -> OpenCV c2w 회전. 모듈 docstring의 변환식."""
    r = _quat_wxyz_to_matrix(np.asarray(q_wxyz, dtype=np.float64))
    return np.column_stack([-r[:, 1], -r[:, 2], r[:, 0]])


def load_gt_episode(data_root: str, scene: str, episode: int, dataset: str = 'vln_n1') -> dict:
    """에피소드 하나의 GT를 데이터셋 무관 공통 형태로 반환.

    Returns dict:
    - poses_c2w: (N,4,4) float64, OpenCV 컨벤션 camera-to-world (geometry_utils의
      `action_to_c2w(..., 'cam2world_gl')` 결과와 같은 컨벤션)
    - cam_xyz:   (N,3) — poses_c2w의 translation
    - body_xyz:  (N,3) — 로봇 몸통 궤적. **02(클리어런스)/03(경로 재현)은 이걸 쓸 것.**
      vln_pe의 카메라는 몸통보다 0.2m 앞에 돌출돼 있어(마운트 [0.2,0,0.72]) 제자리 회전만
      해도 카메라 xy가 반지름 0.2m 원을 그리며 가짜 경로 길이/회전량을 만든다(실측: 씬2
      회전 위주 에피소드에서 카메라 경로가 A* 대비 25배 길어짐). vln_n1은 로봇 위치 정보가
      없어 cam_xyz와 동일(기존 동작 유지 — 그 데이터는 카메라가 곧 몸이었다).
    - h_b:       카메라의 바닥 기준 높이 [m]
    - pitch_deg: 카메라 하향 피치 [deg] (아래를 볼수록 양수)
    - floor_z:   median(cam_z) - h_b — 02의 클리어런스 게이트용 근사 바닥 높이
    - k:         (3,3) 카메라 intrinsic
    """
    path = Path(data_root) / scene / 'data' / 'chunk-000' / f'episode_{episode:06d}.parquet'
    if dataset == 'vln_n1':
        table = pq.read_table(path, columns=['observation.camera_extrinsic',
                                             'observation.camera_intrinsic', 'action'])
        extrinsic = np.asarray(table['observation.camera_extrinsic'].to_pylist()[0],
                               dtype=np.float64).reshape(4, 4)
        k = np.asarray(table['observation.camera_intrinsic'].to_pylist()[0],
                       dtype=np.float64).reshape(3, 3)
        actions = np.stack([np.asarray(a, dtype=np.float64).reshape(4, 4)
                            for a in table['action'].to_pylist()])
        poses_c2w = np.stack([action_to_c2w(a, 'cam2world_gl') for a in actions])
        h_b, pitch_deg = decompose_camera_extrinsic(extrinsic)
        body_xyz = poses_c2w[:, :3, 3].copy()
    elif dataset == 'vln_pe':
        table = pq.read_table(path, columns=['observation.camera_position',
                                             'observation.camera_orientation',
                                             'observation.robot_position'])
        cam_pos = np.asarray(table['observation.camera_position'].to_pylist(), dtype=np.float64)
        cam_ori = np.asarray(table['observation.camera_orientation'].to_pylist(), dtype=np.float64)
        rob_pos = np.asarray(table['observation.robot_position'].to_pylist(), dtype=np.float64)
        n = len(cam_pos)
        poses_c2w = np.tile(np.eye(4), (n, 1, 1))
        for i in range(n):
            poses_c2w[i, :3, :3] = vlnpe_orientation_to_c2w_rotation(cam_ori[i])
            poses_c2w[i, :3, 3] = cam_pos[i]
        k = VLNPE_K.copy()
        # pitch: OpenCV forward(+z 컬럼)의 world-z 성분으로 유도. 토르소 피치 때문에 프레임마다
        # 15~21°로 흔들리므로 median을 대표값으로 쓴다(2-B 합성 경로용 — 2-A는 프레임별 실제
        # pose를 그대로 쓰므로 영향 없음).
        forward_z = poses_c2w[:, 2, 2]  # R_cv[:,2](forward)의 z성분 = poses[2,2]
        pitch_deg = float(np.degrees(np.arcsin(np.clip(-np.median(forward_z), -1.0, 1.0))))
        # h_b/floor_z: vln_pe엔 extrinsic 컬럼이 없어 로봇 기하로 유도 — torso 높이 상수의
        # 근거/한계는 VLNPE_TORSO_STANDING_M 주석 참고.
        floor_z_est = float(np.median(rob_pos[:, 2])) - VLNPE_TORSO_STANDING_M
        h_b = float(np.median(cam_pos[:, 2])) - floor_z_est
        body_xyz = rob_pos.copy()
    else:
        assert False, f'unreachable dataset={dataset!r}'

    cam_xyz = poses_c2w[:, :3, 3].copy()
    return {
        'poses_c2w': poses_c2w,
        'cam_xyz': cam_xyz,
        'body_xyz': body_xyz,
        'h_b': h_b,
        'pitch_deg': pitch_deg,
        'floor_z': float(np.median(cam_xyz[:, 2]) - h_b),
        'k': k,
    }


# ---------------------------------------------------------------------------
# 프레임 이미지 로더 — vln_pe는 에피소드당 npy 스택이라 lazy 캐시로 반복 로드를 피한다
# ---------------------------------------------------------------------------

_NPY_CACHE = {}
_NPY_CACHE_MAX = 4   # (rgb, depth) x 에피소드 2개 정도면 충분 — 메모리 폭주 방지


def _load_npy_stack(dir_path: Path, episode: int) -> np.ndarray:
    key = (str(dir_path), episode)
    if key not in _NPY_CACHE:
        if len(_NPY_CACHE) >= _NPY_CACHE_MAX:
            _NPY_CACHE.pop(next(iter(_NPY_CACHE)))
        _NPY_CACHE[key] = np.load(dir_path / f'episode_{episode:06d}.npy')
    return _NPY_CACHE[key]


def load_rgb_frame(rgb_dir: Path, episode: int, frame_idx: int, dataset: str = 'vln_n1') -> np.ndarray:
    """(H,W,3) uint8 RGB. `rgb_dir`는 두 데이터셋 모두
    `<scene>/videos/chunk-000/observation.images.rgb` (내용물만 .jpg vs .npy)."""
    if dataset == 'vln_n1':
        return _load_rgb_frame_n1(rgb_dir, episode, frame_idx)
    elif dataset == 'vln_pe':
        return _load_npy_stack(Path(rgb_dir), episode)[frame_idx]
    else:
        assert False, f'unreachable dataset={dataset!r}'


def load_depth_frame_m(depth_dir: Path, episode: int, frame_idx: int, max_depth_m: float,
                       dataset: str = 'vln_n1'):
    """(depth_m with NaN at invalid, valid_mask). vln_pe: m = raw*10, raw>=1.0(10m 클립)은 invalid."""
    if dataset == 'vln_n1':
        return _load_depth_frame_m_n1(depth_dir, episode, frame_idx, max_depth_m)
    elif dataset == 'vln_pe':
        raw = _load_npy_stack(Path(depth_dir), episode)[frame_idx].astype(np.float32)
        depth_m = raw * VLNPE_DEPTH_SCALE_M
        valid = (raw > 0.0) & (raw < 1.0) & (depth_m <= max_depth_m)
        return np.where(valid, depth_m, np.nan), valid
    else:
        assert False, f'unreachable dataset={dataset!r}'


# ---------------------------------------------------------------------------
# 자체 회귀 체크
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    repo = Path(__file__).resolve().parents[3]
    scene = '17DRP5sb8fy'

    # --- vln_n1: 기존 인라인 구현(02/03의 load_gt_episode와 동일 코드)과 값이 같은지 ---
    n1_root = repo / DATASETS['vln_n1']['data_root']
    got = load_gt_episode(str(n1_root), scene, 0, 'vln_n1')
    t = pq.read_table(n1_root / scene / 'data/chunk-000/episode_000000.parquet',
                      columns=['observation.camera_extrinsic', 'action'])
    ext = np.asarray(t['observation.camera_extrinsic'].to_pylist()[0], dtype=np.float64).reshape(4, 4)
    acts = np.stack([np.asarray(a, dtype=np.float64).reshape(4, 4) for a in t['action'].to_pylist()])
    exp_h_b, exp_pitch = decompose_camera_extrinsic(ext)
    exp_xyz = np.stack([action_to_c2w(a, 'cam2world_gl')[:3, 3] for a in acts])
    assert np.array_equal(got['cam_xyz'], exp_xyz), 'vln_n1 cam_xyz가 기존 구현과 다르다'
    assert got['h_b'] == exp_h_b and got['pitch_deg'] == exp_pitch, 'vln_n1 h_b/pitch 불일치'
    assert np.allclose(got['floor_z'], np.median(exp_xyz[:, 2]) - exp_h_b), 'vln_n1 floor_z 불일치'
    rgb = load_rgb_frame(n1_root / scene / 'videos/chunk-000/observation.images.rgb', 0, 0, 'vln_n1')
    assert rgb.shape == (270, 480, 3) and rgb.dtype == np.uint8
    print(f'[OK] vln_n1: 기존 구현과 동일 (frames={len(exp_xyz)}, h_b={exp_h_b:.3f}, '
          f'pitch={exp_pitch:.2f}deg)')

    # --- vln_pe: 실측으로 확인한 데이터 내부 관계들이 성립하는지 ---
    pe_root = repo / DATASETS['vln_pe']['data_root']
    t = pq.read_table(pe_root / scene / 'data/chunk-000/episode_000000.parquet')
    cam_yaw = np.asarray(t['observation.camera_yaw'].to_pylist(), dtype=np.float64)
    cam_ori = np.asarray(t['observation.camera_orientation'].to_pylist(), dtype=np.float64)
    cam_pos = np.asarray(t['observation.camera_position'].to_pylist(), dtype=np.float64)
    rob_pos = np.asarray(t['observation.robot_position'].to_pylist(), dtype=np.float64)
    rob_ori = np.asarray(t['observation.robot_orientation'].to_pylist(), dtype=np.float64)
    # 1) 쿼터니언 해석(wxyz) 검증: R의 forward(0번 컬럼) yaw == camera_yaw
    for i in range(len(cam_yaw)):
        r = _quat_wxyz_to_matrix(cam_ori[i])
        yaw = np.arctan2(r[1, 0], r[0, 0])
        assert abs(np.angle(np.exp(1j * (yaw - cam_yaw[i])))) < 1e-5, f'frame {i} yaw 불일치'
    # 2) 카메라 마운트 관계: cam_pos == rob_pos + R(rob_ori) @ [0.2, 0, 0.72]
    mount = np.array([0.2, 0.0, 0.72])
    err = max(np.linalg.norm(rob_pos[i] + _quat_wxyz_to_matrix(rob_ori[i]) @ mount - cam_pos[i])
              for i in range(len(cam_pos)))
    # 실측: 전 프레임 최대 잔차 8.2mm(보행 중 토르소 유연 관절 때문) — 해석이 맞는지 확인하는
    # 것이 목적이므로 2cm면 충분히 판별된다(해석이 틀리면 수십 cm 오차가 남).
    assert err < 2e-2, f'마운트 오프셋 잔차 {err:.4f}m — cam=rob+R@[0.2,0,0.72] 관계가 깨졌다'
    # 3) 로더 출력 sanity
    got = load_gt_episode(str(pe_root), scene, 0, 'vln_pe')
    assert got['poses_c2w'].shape == (len(cam_pos), 4, 4)
    assert np.allclose(got['cam_xyz'], cam_pos)
    assert 10.0 < got['pitch_deg'] < 25.0, f'pitch {got["pitch_deg"]:.1f}deg — 15도 마운트 기대 범위 밖'
    assert 1.4 < got['h_b'] < 1.9, f'h_b {got["h_b"]:.3f}m — H1 카메라 높이(~1.65) 기대 범위 밖'
    rgb = load_rgb_frame(pe_root / scene / 'videos/chunk-000/observation.images.rgb', 0, 0, 'vln_pe')
    depth_m, valid = load_depth_frame_m(pe_root / scene / 'videos/chunk-000/observation.images.depth',
                                        0, 0, 10.0, 'vln_pe')
    assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
    assert depth_m.shape == (256, 256) and valid.any()
    assert np.nanmax(depth_m) < 10.0 and np.nanmin(depth_m) > 0.0
    print(f'[OK] vln_pe: yaw/마운트/스택 검증 통과 (frames={len(cam_pos)}, '
          f'h_b={got["h_b"]:.3f}, pitch={got["pitch_deg"]:.2f}deg, '
          f'depth range {np.nanmin(depth_m):.2f}~{np.nanmax(depth_m):.2f}m)')
    print('ALL_CHECKS_PASSED')
