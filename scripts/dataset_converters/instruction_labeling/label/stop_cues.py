"""정지 단서 후보 계산 (route.stop_cue 옵션). GT 지시문은 쓰지 않는다 — MP3D `.house` annotation 과 마지막 프레임 pose·depth 만.

A (물체): 정지 지점 수평 max_dist_m 안, 높이가 그럴듯하고, 마지막 프레임(정면·30° 아래 중 하나)에 **보이는** 물체.
          보임 = 물체 중심·축 끝점 중 하나라도 화면 안에 투영되고 depth 로 가려지지 않음 (depth >= 카메라 거리 - occl_tol_m).
좌표: .house 는 mesh Z-up. habitat (x, y_up, z) <-> mesh (x, -z, y).
      pose(릴리스·렌더 공통) 는 시작 기준 로컬 OpenCV c2w (x 전방 / y 좌 / z 위, 시작 바닥 = 0).

.house 줄: `C idx mapidx raw_name mpcat40_idx mpcat40_name ...`, `O idx region cat px py pz a0x a0y a0z a1x a1y a1z r0 r1 r2 ...`
"""

from functools import lru_cache

import numpy as np

FX = 388.19  # 125cm rig (vlnce_rig.py)
W, H = 640, 480
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0


@lru_cache(maxsize=64)
def load_objects(house_path: str) -> list:
    cats, objs = {}, []
    for line in open(house_path):
        t = line.split()
        if not t:
            continue
        if t[0] == 'C':
            cats[int(t[1])] = t[5] if len(t) > 5 else 'misc'
        elif t[0] == 'O':
            c = np.array(list(map(float, t[4:7])))
            a0, a1 = np.array(list(map(float, t[7:10]))), np.array(list(map(float, t[10:13])))
            a2 = np.cross(a0, a1)
            r = np.array(list(map(float, t[13:16])))
            objs.append({'idx': int(t[1]), 'region': int(t[2]), 'cat': int(t[3]), 'center': c,
                         'axes': np.stack([a0, a1, a2]), 'radii': r})
    for o in objs:
        o['mpcat40'] = cats.get(o['cat'], 'misc')
    return objs


def start_frame(ep):
    """habitat world -> 시작 기준 로컬 (x 전방, y 좌, z 위) 변환."""
    x, y, z, w = ep['start_rotation']
    yaw = 2.0 * np.arctan2(y, w)
    fwd = np.array([-np.sin(yaw), 0.0, -np.cos(yaw)])
    left = np.array([-np.cos(yaw), 0.0, np.sin(yaw)])
    up = np.array([0.0, 1.0, 0.0])
    return np.asarray(ep['start_position'], float), np.stack([fwd, left, up])


def mesh_to_local(p_mesh, p0, B):
    p_hab = np.array([p_mesh[..., 0], p_mesh[..., 2], -p_mesh[..., 1]]).T if p_mesh.ndim > 1 else \
        np.array([p_mesh[0], p_mesh[2], -p_mesh[1]])
    return (p_hab - p0) @ B.T


def project(p_local, c2w):
    R, t = c2w[:3, :3], c2w[:3, 3]
    pc = (p_local - t) @ R  # = R^T (p - t)
    Z = pc[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = CX + FX * pc[..., 0] / Z
        v = CY + FX * pc[..., 1] / Z
    return u, v, Z


def side_of(p_local, c2w, ahead_deg):
    pc = (p_local - c2w[:3, 3]) @ c2w[:3, :3]
    ang = np.degrees(np.arctan2(pc[0], pc[2]))  # + = 오른쪽 (OpenCV x 오른쪽)
    if abs(ang) <= ahead_deg:
        return 'ahead'
    return 'right' if ang > 0 else 'left'


def object_candidates(ep, house_path, final_poses: dict, final_depths: dict, cfg: dict) -> list:
    """final_poses/final_depths: {rig: 4x4 c2w (로컬)} / {rig: HxW depth[m]} — 마지막 프레임.
    반환: 가까운 순 [{'name', 'mpcat40', 'dist_m', 'side', 'visible_in': [rig], 'center_local'}]"""
    objs = load_objects(str(house_path))
    p0, B = start_frame(ep)
    main = final_poses[cfg['side_rig']]
    stop_xy = main[:2, 3]
    floor_z = main[2, 3] - cfg['camera_height_m']
    out = []
    for o in objs:
        name = cfg['names'].get(o['mpcat40'])
        if not name:  # 제외 범주 (wall, floor, ceiling, misc, objects, void ...)
            continue
        c = mesh_to_local(o['center'], p0, B)
        dz = c[2] - floor_z
        if not (cfg['min_height_m'] <= dz <= cfg['max_height_m']):
            continue
        hr = float(np.max(o['radii'][:2]))  # 수평 반경 근사
        dist = max(0.0, float(np.linalg.norm(c[:2] - stop_xy)) - hr)
        if dist > cfg['max_dist_m']:
            continue
        pts_mesh = np.vstack([o['center']] + [o['center'] + s * o['radii'][k] * o['axes'][k]
                                              for k in range(3) for s in (-0.7, 0.7)])
        if cfg.get('center_visible'):  # v2: 중심이 보여야 한다 (가장자리만 걸친 물체·가려진 낮은 물체 제외)
            pts_mesh = pts_mesh[:1]
        pts = mesh_to_local(pts_mesh, p0, B)
        vis = []
        for rig, c2w in final_poses.items():
            u, v, Z = project(pts, c2w)
            d = final_depths[rig]
            ok = False
            for ui, vi, zi in zip(u, v, Z):
                if zi > cfg['min_cam_dist_m'] and 0 <= ui < W and 0 <= vi < H:
                    dep = float(d[int(vi), int(ui)])
                    if dep <= 0 or dep >= zi - cfg['occl_tol_m']:
                        ok = True
                        break
            if ok:
                vis.append(rig)
        if not vis:
            continue
        side = side_of(c, main, cfg['ahead_deg'])
        if name in (cfg.get('ahead_only') or []) and side != 'ahead':  # v2: 문은 정면에 있을 때만 (옆·뒤 = 방금 지나온 문)
            continue
        size = float(np.prod(np.maximum(o['radii'], 0.05)))
        out.append({'name': name, 'mpcat40': o['mpcat40'], 'dist_m': round(dist, 2), 'size': size,
                    'side': side, 'visible_in': vis, 'center_local': c.round(3).tolist()})
    w = cfg.get('size_weight', 0.0)  # v2: 거리 - w*log(부피) 로 정렬 (같은 거리면 큰 가구 우선)
    out.sort(key=lambda x: x['dist_m'] - w * np.log(x['size']))
    # 같은 이름은 가장 가까운 것 하나만
    seen, uniq = set(), []
    for x in out:
        if x['name'] not in seen:
            seen.add(x['name'])
            uniq.append(x)
    return uniq
