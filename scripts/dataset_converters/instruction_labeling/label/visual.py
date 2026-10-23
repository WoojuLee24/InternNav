"""작성자 B·C: 이동 정보를 번호 목록(route words) 대신 그림으로 준다 (e2e.build_request 가 위임).

  B frames.overlay=arrow-v1    : 궤적 사진 위에 "여기서 경로가 가는 쪽" 화살표(아래 흰 원) + 바닥 초록선(남은 경로).
                                 프레임 = 출발 + waypoint 에 도착한 순간(돌기 전) + 도착. 사진 설명에 회전 글은 없다.
  C frames.select=topdown-v1   : MP3D .house 방 바닥 다각형으로 그린 평면도 + 경로 1장, 출발·도착 사진.
                                 평면도는 출발 때 바라보는 방향이 위쪽 (사진의 좌·우와 같게).
  route.motion=none 이면 프롬프트의 번호 목록을 빼고 방 순서 줄만 남긴다.
GT 지시문은 쓰지 않는다 — reference_path, start_rotation, 궤적 pose·depth, scene annotation 만 쓴다.
"""

import base64
import io
import math
import re
import threading

import cv2
import numpy as np

from frames import subsample, waypoints_local
from route_words import diff, heading, route_words
from sources import house_path
from style import episode_style

FX = 320.0 / math.tan(math.radians(79.0 / 2))  # render.yaml: 640x480, hfov 79
CX, CY = 320.0, 240.0
_MPL_LOCK = threading.Lock()


def path_local3(ep: dict, step_m: float = 0.25) -> np.ndarray:
    """reference_path -> 시작 기준 로컬 (x 전방, y 좌, z 바닥 높이 = 시작 바닥 0) 를 step_m 간격으로 (N,3)."""
    xy = waypoints_local(ep)
    z = np.asarray(ep['reference_path'])[:, 1] - ep['start_position'][1]
    P = np.c_[xy, z]
    out = []
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b[:2] - a[:2]) / step_m)))
        out += [a + (b - a) * k / n for k in range(n)]
    out.append(P[-1])
    return np.asarray(out)


def _arc(P: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]


def overlay_frames(ep: dict, track: dict, turn_deg: float) -> list:
    """출발(0) + waypoint i 에 가장 가까이 온 프레임(돌기 전) + 도착. waypoint 순서대로 앞으로만 찾는다."""
    path = ep['reference_path']
    xy = track['pose'][:, :2, 3]
    wl = waypoints_local(ep)
    out, cursor, prev_h = [{'frame': 0, 'label': 'start', 'waypoint_idx': 0}], 0, None
    for i in range(len(path) - 1):
        h = heading(path[i], path[i + 1])
        if prev_h is not None and abs(diff(prev_h, h)) >= turn_deg:
            arrive = cursor + int(np.argmin(np.linalg.norm(xy[cursor:] - wl[i], axis=1)))
            out.append({'frame': arrive, 'label': f'wp_{i}', 'waypoint_idx': i})
            cursor = arrive
        prev_h = h
    out.append({'frame': len(xy) - 1, 'label': 'goal', 'waypoint_idx': len(path) - 1})
    seen, uniq = set(), []
    for f in sorted(out, key=lambda x: x['frame']):
        if f['frame'] not in seen:
            seen.add(f['frame'])
            uniq.append(f)
    return uniq


def draw_overlay(img: np.ndarray, depth_m: np.ndarray, pose: np.ndarray, P: np.ndarray, k0: int, ov: dict) -> tuple:
    """img(BGR) 위에 남은 경로 P[k0:] 의 바닥선과 진행 방향 화살표를 그린다. pose = 시작 기준 로컬 OpenCV c2w."""
    out = img.copy()
    R, t = pose[:3, :3], pose[:3, 3]
    arc = _arc(P)
    seg = P[k0:][arc[k0:] - arc[k0] <= ov['line_m']]
    C = (seg - t) @ R  # 카메라 좌표 (x 오른쪽, y 아래, z 앞)
    pts = []
    for c in C:
        if c[2] < 0.3:
            pts.append(None)
            continue
        u, v = FX * c[0] / c[2] + CX, FX * c[1] / c[2] + CY
        ui, vi = int(round(u)), int(round(v))
        vis = 0 <= ui < img.shape[1] and 0 <= vi < img.shape[0]
        if vis:
            d = depth_m[vi, ui]
            vis = d <= 0 or d >= 9.9 or c[2] <= d + ov['occlusion_m']  # 벽 뒤 점은 그리지 않는다
        pts.append((ui, vi) if vis else None)
    layer = out.copy()
    for a, b in zip(pts[:-1], pts[1:]):
        if a and b:
            cv2.line(layer, a, b, (40, 200, 40), 7, cv2.LINE_AA)
    out = cv2.addWeighted(layer, 0.7, out, 0.3, 0)
    # 진행 방향: 남은 경로를 따라 lookahead_m 앞 점의 방위 (카메라 정면 기준, + = 왼쪽)
    j = k0 + int(np.searchsorted(arc[k0:] - arc[k0], ov['lookahead_m']))
    j = min(j, len(P) - 1)
    fwd = R[:2, 2]
    vec = P[j, :2] - t[:2]
    rel = math.degrees(math.atan2(fwd[0] * vec[1] - fwd[1] * vec[0], fwd @ vec)) if np.linalg.norm(vec) > 1e-3 else 0.0
    cx, cy, r = 320, 412, 52
    disk = out.copy()
    cv2.circle(disk, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)
    out = cv2.addWeighted(disk, 0.8, out, 0.2, 0)
    cv2.circle(out, (cx, cy), r, (30, 30, 30), 2, cv2.LINE_AA)
    a = math.radians(rel)
    tip = (int(round(cx - math.sin(a) * (r - 8))), int(round(cy - math.cos(a) * (r - 8))))
    tail = (int(round(cx + math.sin(a) * (r - 22))), int(round(cy + math.cos(a) * (r - 22))))
    cv2.arrowedLine(out, tail, tip, (0, 0, 220), 6, cv2.LINE_AA, tipLength=0.45)
    return out, rel


def topdown_image(ep: dict, P: np.ndarray, td: dict, names: dict) -> np.ndarray:
    """평면도 (BGR). 화면 위 = 출발 때 바라보는 방향 (로컬 +x), 화면 왼쪽 = 로컬 +y."""
    with _MPL_LOCK:  # annotate.py 는 스레드 64 개로 돈다. matplotlib 은 스레드 안전하지 않아 그림을 하나씩 그린다
        return _topdown_image(ep, P, td, names)


def _topdown_image(ep: dict, P: np.ndarray, td: dict, names: dict) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Polygon
    from rooms import load_regions

    x, y, z, w = ep['start_rotation']
    yaw = 2.0 * np.arctan2(y, w)
    fwd = np.array([-np.sin(yaw), -np.cos(yaw)])
    left = np.array([-np.cos(yaw), np.sin(yaw)])
    p0 = np.asarray(ep['start_position'])[[0, 2]]
    sy = ep['start_position'][1]
    hs = P[:, 2] + sy  # 경로 바닥 높이 (habitat y = mesh z)

    def to_map(xz):  # habitat (x, z) -> 그림 좌표 (오른쪽 = -로컬 y, 위 = 로컬 x)
        q = np.asarray(xz) - p0
        return np.c_[-(q @ left), q @ fwd]

    S = td['size_px'] / 100
    fig = Figure(figsize=(S, S), dpi=100)  # pyplot 전역 상태를 쓰지 않는다
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    m = np.c_[-P[:, 1], P[:, 0]]
    lo, hi = m.min(0) - td['margin_m'], m.max(0) + td['margin_m']
    half = max(hi - lo) / 2
    mid = (lo + hi) / 2
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    for r in load_regions(str(house_path(ep))):
        if np.min(np.abs(hs - r['floor_z'])) > td['max_dz_m']:
            continue
        v = r['poly'].vertices  # mesh (x, y) -> habitat (x, z = -y)
        mv = to_map(np.c_[v[:, 0], -v[:, 1]])
        ax.add_patch(Polygon(mv, closed=True, facecolor='#ececec', edgecolor='#555555', lw=1.2))
        nm = names.get(r['label'])
        c = mv.mean(0)
        if nm and abs(c[0] - mid[0]) < half and abs(c[1] - mid[1]) < half:
            ax.text(c[0], c[1], nm, ha='center', va='center', fontsize=11, color='#222222')
    ax.plot(m[:, 0], m[:, 1], '-', color='#20b020', lw=4, solid_capstyle='round')
    ax.plot(m[0, 0], m[0, 1], 'o', color='#1f5fd0', ms=13)
    ax.annotate('', xy=(m[0, 0], m[0, 1] + 1.2), xytext=(m[0, 0], m[0, 1]),
                arrowprops=dict(arrowstyle='-|>', color='#1f5fd0', lw=3, mutation_scale=22))
    ax.plot(m[-1, 0], m[-1, 1], '*', color='#e02020', ms=22, mec='black')
    ax.set_aspect('equal')
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor='white')
    return cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)


def _jpeg(img: np.ndarray) -> bytes:
    return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def build_images(cfg: dict, ep: dict, track: dict) -> tuple:
    """[(caption, jpeg bytes, label)], intro, 기록용 정보."""
    fr = cfg['frames']
    rig = fr['rig']
    P = path_local3(ep)
    ov, td = fr['overlay'], fr['select_def'].get('topdown')
    if ov:
        sel = subsample(overlay_frames(ep, track, float(fr['turn_deg'])), int(fr['max_images']))
        xy, arc_k, items, rels = track['pose'][:, :2, 3], 0, [], []
        for f in sel:
            img = cv2.imread(str(track['img'](f['frame'], rig)))
            dep = cv2.imread(str(track['depth'](f['frame'], rig)), -1).astype(np.float32) / 1000.0
            k0 = arc_k + int(np.argmin(np.linalg.norm(P[arc_k:, :2] - xy[f['frame']], axis=1)))  # 경로 진행은 앞으로만
            arc_k = k0
            lab = f['label']
            kind = 'start' if lab == 'start' else 'goal' if lab == 'goal' else 'waypoint'
            if kind == 'goal':
                out, rel = img, None
            else:
                out, rel = draw_overlay(img, dep, track['pose'][f['frame']], P, k0, ov)
            rels.append(None if rel is None else round(rel, 1))
            items.append((ov['captions'][kind], _jpeg(out), lab))
        return items, ov['intro'], {'frames': [f['frame'] for f in sel], 'frame_labels': [f['label'] for f in sel], 'arrow_deg': rels}
    elif td:
        names = (cfg['route'].get('rooms') or {}).get('names') or {}
        T = len(track['action'])
        items = [(td['caption'], _jpeg(topdown_image(ep, P, td, names)), 'topdown')]
        for f, cap, lab in [(0, 'start position, the direction you face before moving', 'start'),
                            (T - 1, 'destination, where you stop (facing the way you arrived)', 'goal')]:
            items.append((cap, track['img'](f, rig).read_bytes(), lab))
        return items, td['intro'], {'frames': ['topdown', 0, T - 1], 'frame_labels': [x[2] for x in items]}
    else:
        assert False, 'unreachable: visual 요청인데 overlay·topdown 이 없다'


def build_request_visual(cfg: dict, ep: dict, track: dict) -> tuple:
    """e2e.build_request 와 같은 반환 (content, inputs, es)."""
    assert 'instruction' not in ep, 'GT 누출: 생성 입력에 instruction 이 있다'
    rc = cfg['route']
    es = episode_style(cfg, ep, track['item'].get('sib_idx', 0))
    route, g = route_words(ep, rc, cfg['seed'])
    tmpl = cfg['style']['template']
    if rc['motion']:  # 번호 목록을 빼고 방 순서 줄만
        route = '\n'.join(l for l in route.split('\n') if not re.match(r'^\d+\. ', l))
        assert 'PATH MOTION SEQUENCE:' in tmpl
        tmpl = tmpl.replace('PATH MOTION SEQUENCE:', rc['motion']['header'])
    items, intro, rec = build_images(cfg, ep, track)
    content = []
    for k, (cap, jpg, _) in enumerate(items, 1):
        content.append({'type': 'text', 'text': f'Image {k}: {cap}'})
        content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(jpg).decode()}})
    prompt = tmpl.format(route=route, rules=es['rules'], extra=es['extra'],
                         scene_intro=intro.format(n_images=len(items)), scene_context=cfg['delivery']['context'])
    content.append({'type': 'text', 'text': prompt})
    inputs = dict(rec, rigs=[cfg['frames']['rig']], route_words=route, rooms=g.get('rooms'),
                  initial_turn_deg=round(g['initial'], 1), start_turn_said=g['start_turn_said'], stop_style=es['stop'], stop_cue=None)
    return content, inputs, es

