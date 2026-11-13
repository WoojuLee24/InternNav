"""노은역 걸어 보기 — 지금 로봇 카메라 화면을 SAM 으로 분할해 칠한 그림을 준다 (GPU, isaac 컨테이너).

두 가지:
  /seg/house.jpg   (?id=N 이면 그 물체 하나만 3D 상자 선 + SAM — 페이지에서 고른 물체 보기)
                   .house 상자로 바로 분할 (빠름, 약 4~5 장/s). 페이지 서버의 랜드마크(.house 중심·축·반지름)를
                   카메라에 투영한 2D 상자를 SAM 에 준다. 가려진 것은 페이지 서버 /visible (메쉬 광선)으로 뺀다.
                   -> ".house 가 여기 있다고 한 물체가 실제로 그 모양인가" 를 본다. .house 에 빠진 물체는 안 나온다.
  /seg/detect.jpg  69 와 같은 검출 (GroundingDINO 어휘 전체 -> SAM). 느리다 (약 0.6~0.9 s). 빠진 물체 찾기용.
  /seg/pick/*      물체 추가: 화면을 멈추고(start) 사람이 누른 점·상자로 분할(mask, SAM2.1 또는 SAM)하고
                   이름을 추천(suggest, GroundingDINO 또는 Gemma)한다. 3D 위치·저장은 페이지 서버(/add_preview · /add).
                   Gemma 는 이 서버가 띄우지 않는다 — --vlm_url (OpenAI 형식, 69 와 같음) 이 켜져 있을 때만 쓴다.

카메라 자세 = teleop_view /state 의 x, y, z, R(몸 회전 전체, 없으면 yaw) + trunk_to_cam_cv — 렌더와 같은 자세.
/state 와 /frame.jpg 를 차례로 받으므로 로봇이 움직이는 중이면 한 프레임쯤 어긋날 수 있다.

    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/noeun/tools/noeun_seg_server.py
"""
import argparse
import base64
import colorsys
import io
import json
import math
import re
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'utils'))
from render_poses import trunk_to_cam_cv  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8768)
    ap.add_argument('--teleop', default='http://127.0.0.1:8765')
    ap.add_argument('--page', default='http://127.0.0.1:8766', help='noeun_landmark_walk.py (/objects · /visible)')
    ap.add_argument('--vocab', default='logs/scene_landmarks/vocab.json', help='검출 모드 어휘 (69 와 같은 것)')
    ap.add_argument('--det', default='IDEA-Research/grounding-dino-base')
    ap.add_argument('--sam', default='facebook/sam-vit-base')
    ap.add_argument('--box_thr', type=float, default=0.25, help='69 와 같음')
    ap.add_argument('--max_obj', type=int, default=16, help='.house 모드에서 한 번에 칠할 물체 수 (가까운 순)')
    ap.add_argument('--max_m', type=float, default=15.0, help='이보다 먼 물체는 안 칠한다')
    ap.add_argument('--sam2_ckpt', default='checkpoints/sam2.1-hiera-base-plus/sam2.1_hiera_base_plus.pt')
    ap.add_argument('--sam2_cfg', default='configs/sam2.1/sam2.1_hiera_b+.yaml', help='sam2 패키지 안 설정')
    ap.add_argument('--vlm_url', default='http://127.0.0.1:8200/v1/chat/completions', help='Gemma 서버 (69 와 같음)')
    ap.add_argument('--vlm_model', default='gemma-4-31B-it')
    ap.add_argument('--font', default='.venv_label/fonts/NanumGothic-Regular.ttf')
    return ap.parse_args()


def get(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def color(i):
    r, g, b = colorsys.hsv_to_rgb((i * 0.618) % 1, 0.75, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def main():
    a = parse()
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
    spn = SamProcessor.from_pretrained(a.sam)
    sam = SamModel.from_pretrained(a.sam).cuda().eval()
    det = {}                                   # 검출기는 처음 부를 때 올린다 (GPU 메모리 아끼기)
    vocab = json.load(open(a.vocab))['vocab']
    chunks, cur, n = [], [], 0                 # 69 와 같이 256 토큰 한도로 쪼갠다
    for w in vocab:
        if cur and n + len(w) + 2 > 700:
            chunks.append(cur); cur, n = [], 0
        cur.append(w); n += len(w) + 2
    chunks = ['. '.join(c) + '.' for c in chunks + ([cur] if cur else [])]
    M_cam = trunk_to_cam_cv()
    meta = json.loads(get(a.teleop + '/meta'))
    font = ImageFont.truetype(a.font, 15) if Path(a.font).exists() else ImageFont.load_default()
    gpu = threading.Lock()                     # GPU 는 한 번에 하나씩
    objs = {'t': 0.0, 'v': []}

    def objects():
        if time.time() - objs['t'] > 5:        # 페이지에서 고친 값이 몇 초 안에 반영된다
            objs['v'] = [o for o in json.loads(get(a.page + '/objects', 10)) if o['kept']]   # in_house 는 쓰는 쪽에서 거른다
            objs['t'] = time.time()
        return objs['v']

    def snapshot():
        s = json.loads(get(a.teleop + '/state'))['pose']
        img = Image.open(io.BytesIO(get(a.teleop + '/frame.jpg'))).convert('RGB')
        return s, img

    def camera(s, W, H):
        c, sn = math.cos(s['yaw']), math.sin(s['yaw'])
        T = np.array([[c, -sn, 0, s['x']], [sn, c, 0, s['y']], [0, 0, 1, s['z']], [0, 0, 0, 1]])
        if s.get('R'):                         # 몸 기울기까지 (렌더 카메라와 같다). 옛 teleop 은 yaw 만
            T[:3, :3] = np.asarray(s['R'], float)
        C = np.asarray(s['C'], float) if s.get('C') else T @ M_cam   # C = teleop 이 실제로 찍은 카메라 (시점 옮김 포함)
        f = W / 2 / math.tan(math.radians(meta['hfov_deg']) / 2)
        return C[:3, :3], C[:3, 3], f

    def masks(img, boxes):
        si = spn(img, input_boxes=[boxes], return_tensors='pt').to('cuda')
        with torch.no_grad():
            so = sam(**si, multimask_output=False)
        return spn.image_processor.post_process_masks(
            so.pred_masks.cpu(), si['original_sizes'].cpu(), si['reshaped_input_sizes'].cpu())[0].numpy()[:, 0]

    EDGES = [(i, i ^ k) for i in range(8) for k in (1, 2, 4) if i < i ^ k]   # 꼭짓점 번호 = 4·sx + 2·sy + sz

    def box3d(o, R, t, f, W, H):
        """.house 상자 꼭짓점 8 개 -> 화면 (u, v), 깊이 z, 중심 깊이, 2D 상자 (화면에 안 들면 None)"""
        h = o['house']
        c, A, r = np.asarray(h['center']), np.asarray(h['axes']), np.asarray(h['radii'])
        X = np.array([c + sx * r[0] * A[0] + sy * r[1] * A[1] + sz * r[2] * A[2]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        P = (X - t) @ R                          # 카메라 좌표 (OpenCV: x 오른쪽, y 아래, z 앞)
        zc = float((c - t) @ R[:, 2])
        z = np.maximum(P[:, 2], 1e-6)
        uv = np.c_[f * P[:, 0] / z + W / 2, f * P[:, 1] / z + H / 2]
        b = None
        if (P[:, 2] >= 0.1).all():
            b = [max(0, uv[:, 0].min()), max(0, uv[:, 1].min()), min(W - 1, uv[:, 0].max()), min(H - 1, uv[:, 1].max())]
            if b[2] - b[0] < 6 or b[3] - b[1] < 6:
                b = None
        return uv, P[:, 2], zc, b

    def wire(d, uv, z, col, width=3):
        """3D 상자 선 12 개 (카메라 뒤로 넘어가는 선은 뺀다). 검은 바깥선으로 어떤 배경에서도 보이게."""
        for w, cl in ((width + 3, (0, 0, 0)), (width, col)):
            for i, j in EDGES:
                if z[i] > 0.1 and z[j] > 0.1:
                    d.line([tuple(uv[i]), tuple(uv[j])], fill=cl, width=w)

    def paint(img, items, title, wires=()):
        """items = [(mask, 상자, 이름, 색)] · wires = [(uv, z, 색)] -> 칠한 jpg"""
        base = np.asarray(img).astype(np.float32)
        for m, _, _, col in items:
            base[m] = base[m] * 0.5 + np.asarray(col, np.float32) * 0.5
        out = Image.fromarray(base.astype(np.uint8))
        d = ImageDraw.Draw(out)
        for uv, z, col in wires:
            wire(d, uv, z, col)
        for m, b, name, col in items:
            if not wires:
                d.rectangle(b, outline=col, width=2)
            tw = d.textlength(name, font=font)
            ty = b[1] - 19 if b[1] >= 19 else b[1]     # 상자가 화면 위에 붙으면 이름을 상자 안에 (밖이면 안 보인다)
            d.rectangle([b[0], ty, b[0] + tw + 8, ty + 19], fill=(0, 0, 0))
            d.text((b[0] + 4, ty + 1), name, fill=col, font=font)
        d.rectangle([0, 0, d.textlength(title, font=font) + 12, 22], fill=(0, 0, 0))
        d.text((6, 3), title, fill=(255, 255, 255), font=font)
        buf = io.BytesIO()
        out.save(buf, 'JPEG', quality=85)
        return buf.getvalue()

    def seg_house(only=None):
        """only = 랜드마크 id 하나 -> 그 물체의 .house 3D 상자 + 그 상자로 SAM (페이지에서 고른 물체 보기)."""
        t0 = time.time()
        s, img = snapshot()
        W, H = img.size
        R, t, f = camera(s, W, H)
        cq = f"&cx={t[0]}&cy={t[1]}&cz={t[2]}"                     # 카메라 자리 (옮겼으면 그 자리)
        vis = set(json.loads(get(a.page + f"/visible?x={s['x']}&y={s['y']}&z={s['z']}&yaw={s['yaw']}" + cq)))
        if only is not None:
            o = next((o for o in objects() if o['id'] == only), None)
            if o is None or not o.get('house'):
                return paint(img, [], f'#{only}: 남긴 랜드마크가 아님 (뺀 것) — .house 상자 없음')
            uv, z, zc, b = box3d(o, R, t, f, W, H)
            h = o['house']
            name = f"#{o['id']} {h['raw_name']} {zc:.1f}m" + ('' if h['in_house'] else ' (높이로 .house 밖)')
            if zc < 0.3:
                return paint(img, [], f'{name} — 카메라 뒤 (안 보임)')
            if b is None:
                return paint(img, [], f'{name} — 화면 밖', [(uv, z, (255, 0, 200))])
            if o['id'] not in vis:          # 벽 뒤 — 상자만 (SAM 은 앞의 다른 물체를 칠하므로 안 돌린다)
                return paint(img, [], f'{name} — 가려짐 (상자만)', [(uv, z, (255, 0, 200))])
            with gpu:
                m = masks(img, [[float(x) for x in b]])[0]
            return paint(img, [(m, b, name, (255, 0, 200))],
                         f'고른 물체 .house 상자 + SAM · {(time.time() - t0) * 1000:.0f} ms', [(uv, z, (255, 0, 200))])
        cand = []
        for o in objects():
            if o['id'] not in vis or not o['house']['in_house']:
                continue
            uv, z, zc, b = box3d(o, R, t, f, W, H)
            if zc < 0.3 or zc > a.max_m or b is None:
                continue
            cand.append((zc, o, b))
        cand = sorted(cand, key=lambda x: x[0])[:a.max_obj]
        items = []
        if cand:
            with gpu:
                ms = masks(img, [[float(x) for x in b] for _, _, b in cand])
            items = [(m, b, f"#{o['id']} {o['house']['raw_name']} {z:.1f}m", color(o['id'])) for m, (z, o, b) in zip(ms, cand)]
        return paint(img, items, f'.house 상자 -> SAM · {len(items)}개 · {(time.time() - t0) * 1000:.0f} ms')

    def seg_detect():
        t0 = time.time()
        _, img = snapshot()
        with gpu:
            if not det:
                det['p'] = AutoProcessor.from_pretrained(a.det)
                det['m'] = AutoModelForZeroShotObjectDetection.from_pretrained(a.det).cuda().eval()
            boxes, labs = [], []
            for ch in chunks:
                inp = det['p'](images=img, text=ch, return_tensors='pt').to('cuda')
                with torch.no_grad():
                    o = det['m'](**inp)
                r = det['p'].post_process_grounded_object_detection(
                    o, inp.input_ids, threshold=a.box_thr, text_threshold=a.box_thr, target_sizes=[img.size[::-1]])[0]
                boxes += [[float(v) for v in b] for b in r['boxes']]
                labs += [str(x) for x in r.get('text_labels', r.get('labels'))]
            ms = masks(img, boxes) if boxes else []
        items = [(m, b, lab, color(hash(lab) % 997)) for m, b, lab in zip(ms, boxes, labs)]
        return paint(img, items, f'검출기(GroundingDINO) -> SAM · {len(items)}개 · {(time.time() - t0) * 1000:.0f} ms')

    # ---------------- 물체 추가 (pick) ----------------
    PICK = {}                                  # fid -> {pose, img, mask, sam2_set}
    sam2 = {}

    def sam2_pred():
        if not sam2:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2['p'] = SAM2ImagePredictor(build_sam2(a.sam2_cfg, a.sam2_ckpt, device='cuda'))
            sam2['fid'] = None
        return sam2['p']

    def pick_start():
        s, img = snapshot()
        fid = str(int(time.time() * 1000))
        PICK.clear()                           # 한 사람이 쓰는 도구 — 마지막 한 장만 둔다
        PICK[fid] = {'pose': s, 'img': img, 'mask': None}
        return {'fid': fid, 'pose': s, 'W': img.size[0], 'H': img.size[1]}

    def jpg(im):
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=88)
        return buf.getvalue()

    def pick_mask(b):
        P = PICK[b['fid']]
        img, pts, box = P['img'], b.get('points') or [], b.get('box')

        def cur_box(im):             # 다시 맞추는 물체의 지금 .house 상자 (청록 선) — 새 모양과 비교용
            o = next((o for o in objects() if o['id'] == b.get('id')), None) if b.get('id') is not None else None
            if o and o.get('house'):
                uv, z, zc, _ = box3d(o, *camera(P['pose'], *im.size), *im.size)
                if zc > 0.3:
                    wire(ImageDraw.Draw(im), uv, z, (0, 230, 255), 2)
            return im
        if not pts and not box:
            P['mask'] = None
            return {'jpg': base64.b64encode(jpg(cur_box(img.copy()))).decode(), 'n_px': 0, 'pixels': []}
        model = b.get('model', 'sam2')
        t0 = time.time()
        with gpu:
            if model == 'sam2':
                pr = sam2_pred()
                if sam2['fid'] != b['fid']:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        pr.set_image(np.asarray(img))
                    sam2['fid'] = b['fid']
                multi = len(pts) == 1 and not box     # 점 하나는 애매하다 -> 후보 3 개 중 점수 높은 것
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    ms, sc, _ = pr.predict(point_coords=np.array([p[:2] for p in pts], float) if pts else None,
                                           point_labels=np.array([p[2] for p in pts]) if pts else None,
                                           box=np.array(box, float) if box else None, multimask_output=multi)
                m = ms[int(np.argmax(sc))] > 0
            elif model == 'sam1':
                kw = {}
                if pts:
                    kw['input_points'] = [[[list(map(float, p[:2])) for p in pts]]]
                    kw['input_labels'] = [[[int(p[2]) for p in pts]]]
                if box:
                    kw['input_boxes'] = [[list(map(float, box))]]
                si = spn(img, return_tensors='pt', **kw).to('cuda')
                with torch.no_grad():
                    so = sam(**si, multimask_output=len(pts) == 1 and not box)
                ms = spn.image_processor.post_process_masks(
                    so.pred_masks.cpu(), si['original_sizes'].cpu(), si['reshaped_input_sizes'].cpu())[0].numpy()[0]
                m = ms[int(so.iou_scores[0, 0].argmax())] if len(ms) > 1 else ms[0]
            else:
                assert False, f'unreachable model={model!r}'
        m = np.asarray(m, bool)
        # 다듬기: SAM2 가 그린 상자 밖에 점 몇 개를 흘리기도 한다 (이젤 실측: 상자 끝 486 인데 625 에 점) -> 3D 상자가 늘어난다
        from scipy.ndimage import label
        if box:
            keep = np.zeros_like(m)
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            keep[max(0, y0 - 3):y1 + 4, max(0, x0 - 3):x1 + 4] = True
            m &= keep
        lab, n = label(m)
        if n > 1:                              # 전체의 2 % 보다 작은 떨어진 조각 = 부스러기 (다리처럼 큰 조각은 남는다)
            area = np.bincount(lab.ravel())[1:]
            m = np.isin(lab, 1 + np.nonzero(area >= max(30, 0.02 * area.sum()))[0])
        P['mask'] = m
        # 3D 로 쏠 점 최대 400 개 — **줄마다 같은 수**로 뽑는다. 넓이 비례로 뽑으면 이젤 다리처럼 가는 부분이
        # 몇 점밖에 안 남고 상자 백분위에서 잘린다. 가장자리에서 뒤 벽에 맞는 광선은 페이지 서버가 물체 거리로 옮긴다.
        ys_all, xs_all = np.nonzero(m)
        xs, ys = [], []
        if len(xs_all):
            rows = np.unique(ys_all)
            per = max(1, -(-400 // len(rows)))
            for y in rows:
                xr = xs_all[ys_all == y]
                kk = np.linspace(0, len(xr) - 1, min(per, len(xr))).astype(int)
                xs += xr[kk].tolist(); ys += [int(y)] * len(kk)
        xs, ys = np.array(xs, int), np.array(ys, int)
        k = np.linspace(0, len(xs) - 1, min(400, len(xs))).astype(int) if len(xs) else []
        # 안쪽 = 가장자리에서 3 px 넘게 들어간 픽셀 (판 · 몸통). 메쉬 거리가 믿을 만한 곳 -> 페이지 서버가 물체 거리를 여기서 잰다
        from scipy.ndimage import distance_transform_edt
        inner = distance_transform_edt(m) > 3
        # 잘 보이게: 마스크 밖은 어둡게, 안은 자홍색, 둘레는 흰 선 (노란 물체 위 노란 칠은 안 보였다)
        base = np.asarray(img).astype(np.float32)
        base[~m] *= 0.45
        base[m] = base[m] * 0.5 + np.array([255, 0, 200], np.float32) * 0.5
        rim = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        rim = rim | np.roll(rim, 1, 0) | np.roll(rim, 1, 1)
        base[rim] = 255
        out = cur_box(Image.fromarray(base.astype(np.uint8)))
        d = ImageDraw.Draw(out)
        t = f'SAM {model} · 분할 {int(m.sum())} px'
        d.rectangle([0, 0, d.textlength(t, font=font) + 12, 22], fill=(0, 0, 0))
        d.text((6, 3), t, fill=(255, 255, 255), font=font)
        for x, y, lab in pts:
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(40, 200, 80) if lab else (230, 50, 50), outline=(255, 255, 255), width=2)
        if box:
            d.rectangle(box, outline=(80, 170, 255), width=2)
        bb = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
        return {'jpg': base64.b64encode(jpg(out)).decode(), 'n_px': int(m.sum()), 'bbox': bb,
                'pixels': [[int(xs[i]), int(ys[i]), int(inner[ys[i], xs[i]])] for i in k], 'ms': round((time.time() - t0) * 1000)}

    words = sorted({w.strip().lower() for w in vocab if not w.lower().startswith('no ')}, key=len, reverse=True)

    def to_vocab(lab):
        """DINO 는 붙어 있는 어휘 여러 개를 한 이름으로 붙여 낸다 ('store front storefronts subway entrance')
        -> 들어 있는 어휘 단어로 다시 쪼갠다 (긴 것 먼저). 'no objects visible' 같은 69 어휘 찌꺼기는 뺀다."""
        rest, out = ' ' + lab.strip().lower() + ' ', []
        for w in words:
            if f' {w} ' in rest:
                out.append(w)
                rest = rest.replace(f' {w} ', ' | ')
        return out

    def suggest_dino(P):
        m = P['mask']
        with gpu:
            if not det:
                det['p'] = AutoProcessor.from_pretrained(a.det)
                det['m'] = AutoModelForZeroShotObjectDetection.from_pretrained(a.det).cuda().eval()
            best = {}
            for ch in chunks:
                inp = det['p'](images=P['img'], text=ch, return_tensors='pt').to('cuda')
                with torch.no_grad():
                    o = det['m'](**inp)
                r = det['p'].post_process_grounded_object_detection(
                    o, inp.input_ids, threshold=0.15, text_threshold=0.25, target_sizes=[P['img'].size[::-1]])[0]
                for bx, sc, lab in zip(r['boxes'], r['scores'], r.get('text_labels', r.get('labels'))):
                    x0, y0, x1, y1 = [int(round(float(v))) for v in bx]
                    inter = int(m[max(y0, 0):y1, max(x0, 0):x1].sum())
                    iou = inter / max(1, m.sum() + (x1 - x0) * (y1 - y0) - inter)   # 마스크 ↔ 상자 겹침
                    v = float(sc) * iou
                    for w in to_vocab(str(lab)):
                        if iou > 0.1 and v > best.get(w, (0,))[0]:
                            best[w] = (v, float(sc), iou)
        return [{'name': k, 'score': round(v[0], 3), 'det': round(v[1], 2), 'iou': round(v[2], 2)}
                for k, v in sorted(best.items(), key=lambda kv: -kv[1][0])[:8]]

    Q_NAME = ('The object highlighted in yellow (inside the box) is in a Korean subway station. '
              'Answer ONLY a JSON object: {"name": short english noun phrase a person would use, lowercase, '
              '"aliases": [up to 3 other english names], "common": one broad english class word}.')

    def suggest_gemma(P):
        m = P['mask']
        ys, xs = np.nonzero(m)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        pw, ph = max(40, (x1 - x0) * 0.6), max(40, (y1 - y0) * 0.6)
        W, H = P['img'].size
        crop = [int(max(0, x0 - pw)), int(max(0, y0 - ph)), int(min(W, x1 + pw)), int(min(H, y1 + ph))]
        base = np.asarray(P['img']).astype(np.float32)
        base[m] = base[m] * 0.6 + np.array([255, 210, 74], np.float32) * 0.4
        im = Image.fromarray(base.astype(np.uint8))
        ImageDraw.Draw(im).rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], outline=(255, 210, 74), width=3)
        im = im.crop(crop)
        body = {'model': a.vlm_model, 'max_tokens': 120, 'temperature': 0,
                'messages': [{'role': 'user', 'content': [
                    {'type': 'text', 'text': Q_NAME},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(jpg(im)).decode()}}]}]}
        req = urllib.request.Request(a.vlm_url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                txt = json.loads(r.read())['choices'][0]['message']['content']
        except OSError as e:
            raise RuntimeError(f'Gemma 서버 꺼짐 ({a.vlm_url}) — 서버를 띄운 뒤 다시 누르세요: {e}')
        js = json.loads(re.search(r'\{.*\}', txt, re.S).group(0))
        out = [{'name': str(js['name']).strip().lower(), 'aliases': [str(x).strip().lower() for x in js.get('aliases') or []],
                'common': str(js.get('common') or '').strip().lower()}]
        return out + [{'name': x, 'aliases': [], 'common': out[0]['common']} for x in out[0]['aliases']]

    def pick_suggest(b):
        P = PICK[b['fid']]
        if P['mask'] is None or not P['mask'].any():
            raise RuntimeError('먼저 물체를 눌러 분할하세요')
        t0 = time.time()
        model = b.get('model', 'dino')
        if model == 'dino':
            r = suggest_dino(P)
        elif model == 'gemma':
            r = suggest_gemma(P)
        else:
            assert False, f'unreachable model={model!r}'
        return {'model': model, 'names': r, 'ms': round((time.time() - t0) * 1000)}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def do_GET(self):
            p = self.path.split('?')[0]
            try:
                if p == '/seg/house.jpg':
                    q = dict(kv.split('=', 1) for kv in self.path.split('?', 1)[1].split('&')) if '?' in self.path else {}
                    body = seg_house(int(q['id']) if q.get('id') else None)
                elif p == '/seg/detect.jpg':
                    body = seg_detect()
                elif p == '/seg/pick/frame.jpg':
                    fid = self.path.split('fid=')[-1]
                    body = jpg(PICK[fid]['img'])
                else:
                    return self._send(404, b'not found', 'text/plain')
                self._send(200, body, 'image/jpeg')
            except Exception as e:      # teleop/페이지가 아직 안 떴거나 끊겼다
                self._send(502, f'{type(e).__name__}: {e}'.encode(), 'text/plain; charset=utf-8')

        def do_POST(self):
            p = self.path.split('?')[0]
            try:
                b = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
                if p == '/seg/pick/start':
                    r = pick_start()
                elif p == '/seg/pick/mask':
                    r = pick_mask(b)
                elif p == '/seg/pick/suggest':
                    r = pick_suggest(b)
                else:
                    return self._send(404, b'not found', 'text/plain')
                self._send(200, json.dumps(r, ensure_ascii=False).encode(), 'application/json; charset=utf-8')
            except KeyError as e:
                self._send(400, json.dumps({'error': f'멈춘 화면이 없습니다 (다시 "물체 추가"): {e}'}, ensure_ascii=False).encode(), 'application/json')
            except Exception as e:
                self._send(500, json.dumps({'error': f'{type(e).__name__}: {e}'}, ensure_ascii=False).encode(), 'application/json')

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(f'[seg] http://localhost:{a.port}  (teleop {a.teleop} · page {a.page})', flush=True)
    ThreadingHTTPServer(('0.0.0.0', a.port), H).serve_forever()


if __name__ == '__main__':
    main()
