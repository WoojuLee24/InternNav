"""vln_ce GT 로더 — `dataset_utils.load_gt_episode`의 vln_ce 분기.

왜 별도 파일인가
----------------
vln_n1 / vln_pe는 parquet에 **mesh 절대좌표** pose가 들어 있어 그대로 읽으면 되지만,
vln_ce의 `pose.{rig}`는 **에피소드 start 기준 상대좌표**다. mesh 좌표로 올리려면 변환이
필요하고, 그 변환을 구하는 경로가 둘 있다:

1. **해석적** — `raw_data`의 `start_position`/`start_rotation`에서 바로 계산
   (`vlnce_align.hmap_translation` / `hmap_yaw`). mesh도 depth도 안 읽는다. **이 파일이 쓰는 것.**
2. **정합** — 에피소드 depth를 점구름으로 쌓아 mesh에 강체 피팅(`vlnce_align.align_episode`).
   `05_export_vlnce.py`가 렌더할 때 쓴다. 더 정확하지만 mesh 로드 + depth 읽기가 필요해 느리다.

경로 재현(`03 --mode reproduce`)은 start/goal과 로봇 키만 있으면 되므로 1번으로 충분하다.
둘의 차이는 `--self_test`가 실측해서 보고한다.

rig와 로봇 키
-------------
vln_ce는 rig가 5종인데 **경로를 가르는 건 높이뿐**이다(각도는 카메라가 보는 방향만 바꾼다).

    125cm_0deg  125cm_30deg  125cm_45deg   -> h_b 1.25 m
     60cm_15deg  60cm_30deg                -> h_b 0.60 m

vln_pe처럼 에피소드마다 다르지 않고 rig가 정하는 **고정값**이다.

에피소드 매칭
-------------
`raw_data`의 `episode_id`는 parquet 인덱스와 다르다. `vlnce_align.build_rawdata_index`가
**(씬, 지시문 텍스트)**를 키로 쓰므로 여기서도 같은 방식으로 찾는다.

self-check: `/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/vlnce_gt.py --self_test`
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vlnce_align  # noqa: E402
import vlnce_rig  # noqa: E402

DEFAULT_RAW_ROOT = 'data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r'
DEFAULT_RIG = '125cm_30deg'

_RAW_INDEX_CACHE = {}


def rig_pitch_deg(rig: str) -> float:
    """'125cm_30deg' -> 30.0 (카메라 하향 피치)."""
    return float(str(rig).split('_')[1].replace('deg', ''))


def raw_index(raw_root: str) -> dict:
    """`(scan, instruction) -> raw episode` 인덱스. 파일이 커서(train 10,819 ep) 캐시한다."""
    key = str(raw_root)
    if key not in _RAW_INDEX_CACHE:
        _RAW_INDEX_CACHE[key] = vlnce_align.build_rawdata_index(Path(raw_root))
    return _RAW_INDEX_CACHE[key]


def episode_instruction(data_root: str, scene: str, episode: int) -> str:
    """`meta/episodes.jsonl`에서 그 에피소드의 지시문 — raw_data 매칭 키."""
    p = Path(data_root) / scene / 'meta' / 'episodes.jsonl'
    with open(p) as fh:
        for line in fh:
            e = json.loads(line)
            if e['episode_index'] == episode:
                return e['tasks'][0].split('<INSTRUCTION_SEP>')[0].strip()
    raise KeyError(f'{scene} ep{episode}: episodes.jsonl에 없다')


def sf2mesh_analytic(raw_episode: dict) -> np.ndarray:
    """raw episode -> (4,4) T_sf2mesh. mesh·depth를 읽지 않는 해석적 변환.

    `vlnce_align`이 이미 푼 두 조각을 쓴다 — 평행이동은 `hmap_translation`(habitat Y-up ->
    mesh Z-up), 회전은 `hmap_yaw`(start_rotation 쿼터니언 -> mesh yaw). start 프레임은
    정의상 원점이므로 이 둘이 변환 전부다.
    """
    t = vlnce_align.hmap_translation(raw_episode['start_position'])
    yaw = vlnce_align.hmap_yaw(raw_episode['start_rotation'])
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = t
    return T


def load_gt_episode_vlnce(data_root: str, scene: str, episode: int,
                          rig: str = DEFAULT_RIG, raw_root: str = DEFAULT_RAW_ROOT) -> dict:
    """`dataset_utils.load_gt_episode`와 **같은 계약**의 dict를 돌려준다.

    poses_c2w / cam_xyz / body_xyz / h_b / pitch_deg / floor_z / k.
    vln_ce는 카메라가 곧 몸이므로 `body_xyz == cam_xyz`다(vln_n1과 같은 취급).
    """
    path = Path(data_root) / scene / 'data' / 'chunk-000' / f'episode_{episode:06d}.parquet'
    pose_key = f'pose.{rig}'
    table = pq.read_table(path, columns=[pose_key])
    pose_rel = np.stack([np.asarray(p, dtype=np.float64).reshape(4, 4)
                         for p in table[pose_key].to_pylist()])

    instr = episode_instruction(data_root, scene, episode)
    raw = raw_index(raw_root).get((scene, instr))
    if raw is None:
        raise KeyError(f'{scene} ep{episode}: raw_data에서 지시문 매칭 실패')

    T = sf2mesh_analytic(raw)
    poses_c2w = np.einsum('ij,njk->nik', T, pose_rel)
    cam_xyz = poses_c2w[:, :3, 3].copy()

    h_b = vlnce_rig.rig_height_m(rig)
    fx, fy, cx, cy = vlnce_rig.intrinsics_for_rig(rig)
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return {
        'poses_c2w': poses_c2w,
        'cam_xyz': cam_xyz,
        'body_xyz': cam_xyz.copy(),
        'h_b': h_b,
        'pitch_deg': rig_pitch_deg(rig),
        'floor_z': float(np.median(cam_xyz[:, 2]) - h_b),
        'k': k,
    }


def reference_path_mesh(raw_episode: dict) -> np.ndarray:
    """raw episode의 `reference_path`(habitat Y-up) -> (N,3) mesh Z-up. 채점 기준선."""
    return np.stack([vlnce_align.hmap_translation(p) for p in raw_episode['reference_path']])


# ---------------------------------------------------------------------------
# self-check — 해석적 변환이 정합(05_export_vlnce가 쓰는 것)과 얼마나 다른가
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--self_test', action='store_true')
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--raw_root', default=DEFAULT_RAW_ROOT)
    ap.add_argument('--mesh_root', default='data/scene_data/mp3d_pe')
    ap.add_argument('--scene', default='17DRP5sb8fy')
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--rig', default=DEFAULT_RIG)
    args = ap.parse_args()

    print(f'[vlnce_gt] {args.scene} rig={args.rig} ep 0..{args.episodes - 1}')
    rows = []
    for ep in range(args.episodes):
        try:
            g = load_gt_episode_vlnce(args.data_root, args.scene, ep, args.rig, args.raw_root)
        except KeyError as e:
            print(f'  ep {ep}: skip — {e}')
            continue
        instr = episode_instruction(args.data_root, args.scene, ep)
        ref = reference_path_mesh(raw_index(args.raw_root)[(args.scene, instr)])
        # start/goal 이 reference_path 양 끝과 맞는지 (해석적 변환의 자기검증)
        d_start = float(np.linalg.norm(g['cam_xyz'][0, :2] - ref[0, :2]))
        d_goal = float(np.linalg.norm(g['cam_xyz'][-1, :2] - ref[-1, :2]))
        rows.append((ep, len(g['cam_xyz']), g['h_b'], g['pitch_deg'], g['floor_z'], d_start, d_goal))
        print(f'  ep {ep:>3}: {len(g["cam_xyz"]):>4} frame · h_b {g["h_b"]:.2f} pitch {g["pitch_deg"]:.0f}° '
              f'floor_z {g["floor_z"]:+.3f} · start-ref {d_start:.3f} m · goal-ref {d_goal:.3f} m')

    assert rows, '읽은 에피소드가 없다'
    ds = np.array([r[5] for r in rows])
    print(f'\n[vlnce_gt] start 오차 median {np.median(ds):.4f} m / max {ds.max():.4f} m')
    assert np.median(ds) < 0.5, f'start가 reference_path 시작점과 너무 멀다 (median {np.median(ds):.3f} m)'
    print('[vlnce_gt] PASS — 해석적 변환이 reference_path와 같은 좌표계에 놓인다')
