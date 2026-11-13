"""노은역 "stop by A" 라벨 품질 측정 — A 를 검출하고 목적지까지 거리를 잰다.

왜: 성공 반경이 3 m 인데 문장이 가리키는 A 가 목적지에서 멀면 SR@3m 가 성립하지 않는다.
    노은역 문장은 Gemma 3 1B 가 목적지 프레임을 보고 지은 것이고 (`stop_cue: null`),
    MP3D 쪽 `instruction_labeling/label/stop_cues.py` 가 하는 거리 제한 · depth 가림 검사가 없다.

재는 법: 문장 끝의 "stop (by|near|at) A" 에서 A 를 뽑고 -> 목적지 프레임에서 GroundingDINO 로 A 를 검출 ->
        검출 상자 안 depth 중앙값 + c2w 로 A 의 world 위치 -> 목적지까지 수평 거리.

    python3 scripts/apptainer/noeun_landmark_check.py --snap logs/goal_snap_d --out logs/landmark
"""
import argparse, gzip, json, math, re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

W, H = 640, 480
HFOV_DEG = 68.7
STOP_RE = re.compile(r'\bstop(?:ping)?\s+(?:by|near|at|on|in|once you reach|when you reach)\s+(.+?)[.;]?$', re.I)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--snap', required=True, help='noeun_vlnpe_snap.py --save_depth 결과 폴더')
    ap.add_argument('--val', default='data/data/noeun_eval/labels/vln_pe_noeun_nav3d_rbq_v6_b1/raw_data/r2r/val_seen/val_seen.json.gz')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='IDEA-Research/grounding-dino-tiny')
    ap.add_argument('--box_thr', type=float, default=0.25)
    ap.add_argument('--mesh', default='data/data/noeun_eval/navmesh/noeun_station/noeun_station.glb',
                    help='씬 메시 (ours Z-up). depth 대신 광선을 쏴서 3D 위치를 구한다')
    ap.add_argument('--use_mesh', action='store_true', help='depth 대신 메시 광선 사용')
    ap.add_argument('--text_thr', type=float, default=0.25)
    return ap.parse_args()


def landmark(text):
    """문장 끝의 'stop ... A' 에서 A. 못 뽑으면 None."""
    for p in reversed(re.split(r'(?<=[.!?])\s+', text.strip())):
        m = STOP_RE.search(p.strip())
        if m:
            a = m.group(1).strip()
            a = re.sub(r'^(the|a|an)\s+', '', a, flags=re.I)
            return a
    return None


def ray_dir(u, v, c2w):
    """화면 (u,v) 를 지나는 world 광선 방향 (단위벡터). c2w 는 noeun_vlnpe_snap.c2w_cv 규약."""
    fx = (W / 2) / math.tan(math.radians(HFOV_DEG) / 2)
    d_cam = np.array([(u - (W - 1) / 2) / fx, (v - (H - 1) / 2) / fx, 1.0])
    d = np.asarray(c2w)[:3, :3] @ d_cam
    return d / np.linalg.norm(d)


def pixel_to_world(u, v, depth_m, c2w):
    """OpenCV 화면 좌표 + depth[m] -> world 점. c2w 는 noeun_vlnpe_snap.c2w_cv 규약."""
    fx = (W / 2) / math.tan(math.radians(HFOV_DEG) / 2)
    fy = fx
    x = (u - (W - 1) / 2) * depth_m / fx
    y = (v - (H - 1) / 2) * depth_m / fy
    p_cam = np.array([x, y, depth_m, 1.0])
    return (np.asarray(c2w) @ p_cam)[:3]


def main():
    a = parse()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    snap = Path(a.snap)
    h2o = lambda q: np.array([q[0], -q[2], q[1]])

    eps = {str(e['episode_id']): e for e in json.load(gzip.open(a.val, 'rt'))['episodes']}
    recs = [json.loads(l) for l in (snap / 'snaps.jsonl').open()]
    by = {}
    for r in recs:
        by.setdefault(str(r['ep']), []).append(r)

    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(a.model).to('cuda').eval()
    print(f'[landmark] {a.model} 로드', flush=True)

    rows = []
    for ep in sorted(by, key=int):
        e = eps[ep]
        txt = e['instruction']['instruction_text'] if isinstance(e['instruction'], dict) else e['instruction']
        A = landmark(txt)
        gp = h2o(e['goals'][0]['position'])
        # 목적지에 가장 가까운 스냅
        best = min(by[ep], key=lambda r: math.hypot(r['robot_pos'][0] - gp[0], r['robot_pos'][1] - gp[1]))
        cam_d = math.hypot(best['robot_pos'][0] - gp[0], best['robot_pos'][1] - gp[1])
        row = {'ep': ep, 'instruction': txt, 'landmark': A, 'frame': best['rgb'],
               'cam_to_goal_m': round(cam_d, 2), 'step': best['step']}
        if A is None or 'depth' not in best:
            row['note'] = 'A 추출 실패' if A is None else 'depth 없음'
            rows.append(row); continue

        img = Image.open(snap / best['rgb']).convert('RGB')
        dep = np.load(snap / best['depth'])['depth'].astype(np.float32)
        if dep.ndim == 3:
            dep = dep[..., 0]
        inp = proc(images=img, text=A.rstrip('.') + '.', return_tensors='pt').to('cuda')
        with torch.no_grad():
            o = model(**inp)
        res = proc.post_process_grounded_object_detection(
            o, inp.input_ids, threshold=a.box_thr, text_threshold=a.text_thr,
            target_sizes=[img.size[::-1]])[0]
        if len(res['scores']) == 0:
            row.update(detected=False, note='검출 없음 (환각 의심)')
            rows.append(row); continue

        i = int(torch.argmax(res['scores']))
        x0, y0, x1, y1 = [float(v) for v in res['boxes'][i]]
        u, v = (x0 + x1) / 2, (y0 + y1) / 2
        x0i, y0i = max(0, int(x0)), max(0, int(y0))
        x1i, y1i = min(W, int(x1)), min(H, int(y1))
        patch = dep[y0i:y1i, x0i:x1i]
        patch = patch[np.isfinite(patch) & (patch > 0.05)]
        if patch.size == 0:
            row.update(detected=True, note='상자 안 depth 없음')
            rows.append(row); continue
        d_med = float(np.median(patch))
        wpt = pixel_to_world(u, v, d_med, best['c2w'])
        row.update(
            detected=True,
            score=round(float(res['scores'][i]), 3),
            box=[round(x0), round(y0), round(x1), round(y1)],
            box_area_pct=round((x1 - x0) * (y1 - y0) / (W * H) * 100, 2),
            u_center=round(u), off_center_deg=round((u - (W - 1) / 2) / ((W / 2) / math.tan(math.radians(HFOV_DEG) / 2)) * 180 / math.pi, 1),
            depth_m=round(d_med, 2),
            landmark_xyz=[round(float(t), 2) for t in wpt],
            dist_to_goal_m=round(float(math.hypot(wpt[0] - gp[0], wpt[1] - gp[1])), 2),
        )
        rows.append(row)
        print(f"  ep{ep:>3s} A='{A[:36]}' score {row.get('score')} depth {row.get('depth_m')} m -> 목적지까지 {row.get('dist_to_goal_m')} m", flush=True)

    json.dump(rows, open(out / 'landmarks.json', 'w'), ensure_ascii=False, indent=1)
    print(f'[landmark] 저장 {out}/landmarks.json  ({len(rows)} ep)')


if __name__ == '__main__':
    main()
