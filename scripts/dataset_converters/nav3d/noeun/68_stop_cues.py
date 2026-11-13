"""noeun 68 · 정지 단서(stop_cue) 생성 — MP3D `.house` 대신 **GroundingDINO + SAM + 씬 메시 광선**으로.

왜: `instruction_labeling/label/stop_cues.py` 는 MP3D `.house` 물체 annotation 을 읽는다. 노은역은 GS 스캔이라
    `.house` 가 없어 `stop_cue: null` 로 생성됐고, Gemma 3 1B 가 목적지 프레임만 보고 멈출 대상을 **지어냈다**.
    결과: val_seen 40 중 21 개가 "A 가 목적지 2 m 밖" 또는 "서술 범위 3 m 초과" (ep23 'stairs' 16.7 m 등).
    성공 기준이 목적지 3 m 안이므로 **문장대로 해도 실패**하는 문제였다.

출력은 `stop_cues.object_candidates` 와 같은 모양이라 `build_stop_cues.py` · `config.py` 의 A-grounded 경로에 그대로 꽂힌다:
    [{'name', 'dist_m', 'side', 'visible_in', 'center_local', ...}]

고르는 기준 (사용자 결정, 2026-11-12):
  a 가깝다 : 3D 중앙점 ↔ 목적지 수평거리 <= max_dist_m (2.0, route.yaml A-grounded-v1 과 같은 값)
             근거 = 로봇이 A 옆(약 1 m)에 서면 목적지에서 최대 3 m -> 성공 반경 안
  b 크다   : 마스크 면적 >= area_min (0.5 % = 40x40 px) — 더 작으면 사람도 못 알아본다
  c 좁다   : 서술 범위(3D 점들의 p90) <= spread_max (3.0) — "the street" 같은 **구역**을 거른다
  d 계단 중간 아님 : 목적지가 verify.json 의 계단 연결부 bbox 안이면 단서를 만들지 않는다

허용 서술 유형: 단일 물체 · 두 물체 사이 · 물체의 특정 끝 · 면의 중심.
  "경계를 막 지난 곳" · "~의 앞" 은 **거리 표현이 모호**해서 쓰지 않는다 (퍼짐으로는 안 걸러진다).

3D 위치는 씬 메시 광선으로 구한다. Isaac depth 와 대조: 80 점 중 96 % 가 0.3 m 안, 상관 0.90,
어긋난 2 점은 Isaac 의 max_depth=10 m 포화 — 메시 쪽이 더 멀리까지 잰다.

    python3 scripts/dataset_converters/nav3d/noeun/68_stop_cues.py --snaps logs/goal_snap,logs/goal_snap2 --out logs/noeun_stop_cues
"""
import argparse, gzip, json, math, re
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

W, H, HFOV_DEG = 640, 480, 68.7
FX = (W / 2) / math.tan(math.radians(HFOV_DEG) / 2)

# 노은역에 실제로 있는 것들 (12_build_navmesh · 61_build_zones 의 구역 이름과 사진 확인 기준).
# 열린 어휘 검출기에 주는 후보군 — `.house` 의 mpcat40 목록 자리를 대신한다.
VOCAB = [
    'pillar', 'column', 'sign', 'signboard', 'vending machine', 'bench', 'seat',
    'stairs', 'escalator', 'ticket gate', 'turnstile', 'door', 'glass door',
    'station map', 'information board', 'trash can', 'elevator', 'railing',
    'kiosk', 'ticket machine', 'fire extinguisher', 'planter', 'advertisement screen',
]


def canon(tok: str, vocab=None) -> str:
    """검출기가 돌려준 토막을 어휘의 원래 이름으로 되돌린다.

    GroundingDINO 는 "a. b. c." 프롬프트의 **토큰별 점수**로 연속 구간을 뽑아 이름으로 준다.
    그래서 어휘 항목이 잘리거나('trash can' -> 'trash') 경계를 넘어 붙는다('pillar. column.' -> 'pillar column').
    -> 토막을 포함하거나 토막에 포함되는 어휘 중 **가장 많이 겹치는 것**으로 되돌린다.
    """
    vocab = vocab or VOCAB
    # WordPiece 조각('##osk' 는 kiosk 의 꼬리) 을 붙여 되돌린다
    t = tok.lower().strip(' .').replace(' ##', '').replace('##', '')
    if not t:
        return None
    best, best_score = None, 0
    tw = set(t.split())
    for v in vocab:
        vw = set(v.lower().split())
        inter = len(tw & vw)
        if not inter:
            continue
        # 겹치는 단어 수가 많고, 어휘가 짧을수록(= 토막이 그 어휘를 거의 다 덮을수록) 좋다
        score = inter * 100 - abs(len(vw) - inter) * 10 - len(vw)
        if score > best_score:
            best, best_score = v, score
    if best is None:
        # 어휘에 못 붙는 토막은 쓰지 않는다 (예전엔 '##osk' 가 그대로 이름이 됐다)
        for v in vocab:
            if t and (t in v.lower().replace(' ', '') or v.lower().replace(' ', '') in t):
                return v
    return best


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--snaps', required=True, help='목적지 프레임 폴더 (쉼표로 여러 개)')
    ap.add_argument('--val', required=True, help='episode json.gz (habitat 좌표)')
    ap.add_argument('--mesh', default='data/data/noeun_eval/navmesh/noeun_station/noeun_station.glb')
    ap.add_argument('--verify', default='data/data/noeun_eval/navmesh/noeun_station/verify.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--det', default='IDEA-Research/grounding-dino-base')
    ap.add_argument('--sam', default='facebook/sam-vit-base')
    ap.add_argument('--box_thr', type=float, default=0.30)
    ap.add_argument('--max_dist_m', type=float, default=2.0)
    ap.add_argument('--area_min', type=float, default=0.5)
    ap.add_argument('--spread_max', type=float, default=3.0)
    ap.add_argument('--ahead_deg', type=float, default=20.0, help='route.yaml 과 같음 — 이 안이면 "right in front of you"')
    ap.add_argument('--inst_r', type=float, default=10.0)
    ap.add_argument('--max_n', type=int, default=1)
    ap.add_argument('--vlm_url', default='http://127.0.0.1:8200/v1/chat/completions',
                    help='Gemma 서버 (vlm_server.py). 비우면 어휘 검출만 쓴다 (예전 방식)')
    ap.add_argument('--vlm_model', default='gemma-4-31B-it')
    ap.add_argument('--cross_m', type=float, default=1.0,
                    help='Gemma 점과 DINO+SAM 중앙점의 3D 거리가 이보다 멀면 버린다 (교차 검증)')
    ap.add_argument('--stair_pad_z', type=float, default=0.5)
    return ap.parse_args()


# Gemma 는 **1000 기준**으로 좌표를 답한다 (직접 물으면 "range: 1000x750" 이라고 말한다).
# 640x480 이라고 알려줘도 1000 기준으로 답해서 화면 밖 좌표가 나온다 -> **비율(0~1)로 받는다**.
# "눈에 띄는 것" 을 물으면 Gemma 는 멀리 있는 큰 것을 고른다 (ep27 자판기 4.1 m · ep37 15.1 m).
# 우리가 쓸 수 있는 것은 **목적지 2 m 안** 뿐이므로 "팔 뻗으면 닿을 거리" 를 명시한다.
Q_NAME = ("You are a robot standing exactly at the destination of a navigation route in a subway station, "
          "looking straight ahead. Name ONE object that is VERY CLOSE to you — within about two meters, "
          "near enough to touch or stand beside — that a person would use to say 'stop here'. "
          "Prefer a single compact object (a pillar, a bench, a sign stand, a vending machine, a railing) "
          "over a large area or the floor. Answer with ONLY a short noun phrase. No explanation.")
Q_POINT = ("Point at the center of {x}. Answer with ONLY two decimals between 0 and 1 "
           "(fraction of image width and height, origin top-left): x y")


def vlm_ask(url, model, img_path, prompt, max_tokens=48, timeout=300):
    import base64, urllib.request
    img = base64.b64encode(open(img_path, 'rb').read()).decode()
    body = {'model': model, 'max_tokens': max_tokens, 'temperature': 0.0,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img}'}}]}]}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    return r['choices'][0]['message']['content'].strip()


def main():
    a = parse()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    h2o = lambda q: np.array([q[0], -q[2], q[1]])
    conns = json.load(open(a.verify))['connectors']

    def on_stairs(p):
        for c in conns:
            (x0, y0), (x1, y1) = c['bbox_xy']; z0, z1 = c['z_range']
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and z0 - a.stair_pad_z <= p[2] <= z1 + a.stair_pad_z:
                return True
        return False

    eps = {str(e['episode_id']): e for e in json.load(gzip.open(a.val, 'rt'))['episodes']}
    by = {}
    for sd in a.snaps.split(','):
        sd = Path(sd.strip())
        if not (sd / 'snaps.jsonl').exists():
            continue
        for l in (sd / 'snaps.jsonl').open():
            r = json.loads(l); r['_dir'] = str(sd)
            by.setdefault(str(r['ep']), []).append(r)

    mesh = trimesh.load(a.mesh, force='mesh', process=False, skip_materials=True)
    RI = mesh.ray  # embreex 있으면 ray_pyembree (약 32 배 빠름)
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
    dp = AutoProcessor.from_pretrained(a.det); det = AutoModelForZeroShotObjectDetection.from_pretrained(a.det).to('cuda').eval()
    spn = SamProcessor.from_pretrained(a.sam); sam = SamModel.from_pretrained(a.sam).to('cuda').eval()
    text = '. '.join(VOCAB) + '.'
    print(f'[68] 메시 {len(mesh.faces)} faces · 광선 {type(RI).__module__} · ep {len(by)} · 어휘 {len(VOCAB)}', flush=True)

    rows = []
    for ep in sorted(by, key=int):
        e = eps[ep]
        gp = h2o(e['goals'][0]['position'])
        best = min(by[ep], key=lambda r: math.hypot(r['robot_pos'][0] - gp[0], r['robot_pos'][1] - gp[1]))
        cam_d = math.hypot(best['robot_pos'][0] - gp[0], best['robot_pos'][1] - gp[1])
        rec = {'episode_id': ep, 'frame': best['rgb'], 'dir': best['_dir'], 'cam_to_goal_m': round(cam_d, 2), 'A': []}
        if on_stairs(gp):
            rec['note'] = '계단 중간 — 단서 안 만듦'; rows.append(rec); print(f"  ep{ep:>4s} {rec['note']}", flush=True); continue
        if cam_d > 3.0:
            rec['note'] = f'목적지 프레임 없음 ({cam_d:.1f} m)'; rows.append(rec); print(f"  ep{ep:>4s} {rec['note']}", flush=True); continue

        img_path = Path(best['_dir']) / best['rgb']
        img = Image.open(img_path).convert('RGB')

        # Gemma 가 **무엇을** 멈출 대상으로 삼을지 정한다 (검출기보다 식별이 낫다).
        # 검출기는 **어디에 있는지**만 찾고, 거리는 메시 광선이 잰다.
        gem_name, gem_pt = None, None
        if a.vlm_url:
            try:
                gem_name = vlm_ask(a.vlm_url, a.vlm_model, img_path, Q_NAME, 40).strip().strip('."')
                if len(gem_name) > 48 or 'unable' in gem_name.lower():
                    gem_name = None
                if gem_name:
                    t = vlm_ask(a.vlm_url, a.vlm_model, img_path, Q_POINT.format(x=gem_name), 16)
                    f = re.findall(r'[01]?\.\d+|[01]\b', t)
                    if len(f) >= 2:
                        u_, v_ = float(f[0]) * (W - 1), float(f[1]) * (H - 1)
                        if 0 <= u_ < W and 0 <= v_ < H:
                            gem_pt = (u_, v_)
            except Exception as e:
                print(f'  ep{ep:>4s} Gemma 실패: {type(e).__name__}', flush=True)
        if gem_name:
            gem_name = re.sub(r'^(a|an|the)\s+', '', gem_name, flags=re.I).strip()
        rec['gemma_name'] = gem_name
        # Gemma 이름 **과** 어휘를 같이 묻는다. Gemma 만 쓰면 그 이름으로 검출이 안 될 때 후보가 0 이 되고
        # (ep5·11·21·27), 어휘만 쓰면 Gemma 가 보는 좋은 대상을 놓친다. 합치면 둘의 장점이 남는다.
        query = (gem_name.rstrip('.') + '. ' + text) if gem_name else text

        inp = dp(images=img, text=query, return_tensors='pt').to('cuda')
        with torch.no_grad():
            o = det(**inp)
        res = dp.post_process_grounded_object_detection(o, inp.input_ids, threshold=a.box_thr,
                                                        text_threshold=a.box_thr, target_sizes=[img.size[::-1]])[0]
        if len(res['scores']) == 0:
            rec['note'] = '후보 없음'; rows.append(rec); print(f"  ep{ep:>4s} 후보 없음", flush=True); continue

        boxes = [[float(v) for v in b] for b in res['boxes']]
        names = list(res.get('text_labels', res.get('labels')))
        si = spn(img, input_boxes=[boxes], return_tensors='pt').to('cuda')
        with torch.no_grad():
            so = sam(**si, multimask_output=False)
        masks = spn.image_processor.post_process_masks(so.pred_masks.cpu(), si['original_sizes'].cpu(),
                                                       si['reshaped_input_sizes'].cpu())[0].numpy()[:, 0]
        c2w = np.asarray(best['c2w']); org = c2w[:3, 3]

        def mask3d(m):
            vs, us = np.nonzero(m)
            if len(us) == 0: return None
            if len(us) > 600:
                k = np.random.RandomState(0).choice(len(us), 600, replace=False); us, vs = us[k], vs[k]
            d = np.stack([(us - (W - 1) / 2) / FX, (vs - (H - 1) / 2) / FX, np.ones(len(us))], axis=1) @ c2w[:3, :3].T
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            loc, _, _ = RI.intersects_location(np.repeat([org], len(d), axis=0), d, multiple_hits=False)
            return loc if len(loc) else None

        cands = []
        for i, (nm, bx) in enumerate(zip(names, boxes)):
            pts = mask3d(masks[i])
            if pts is None: continue
            area = float(masks[i].sum()) / (W * H) * 100
            pbar = np.median(pts, axis=0)
            dist = float(math.hypot(pbar[0] - gp[0], pbar[1] - gp[1]))
            spread = float(np.percentile(np.linalg.norm(pts - pbar, axis=1), 90))
            if dist > a.max_dist_m or area < a.area_min or spread > a.spread_max:
                continue
            u = (bx[0] + bx[2]) / 2
            th = math.degrees(math.atan((u - (W - 1) / 2) / FX))
            side = 'ahead' if abs(th) <= a.ahead_deg else ('left' if th < 0 else 'right')
            nlab, lab = cv2.connectedComponents(masks[i].astype(np.uint8))
            sizes = [int((lab == k).sum()) for k in range(1, nlab)]
            nm_c = canon(str(nm), VOCAB + ([gem_name] if gem_name else []))
            if nm_c is None:
                continue  # 어휘에 못 붙는 토막 (토큰 쓰레기)
            cands.append({'name': nm_c, 'raw_label': str(nm), 'dist_m': round(dist, 2), 'side': side,
                          'visible_in': ['60cm_15deg'], 'center_local': [round(float(t), 2) for t in pbar],
                          'area_pct': round(area, 2), 'spread_m': round(spread, 2),
                          'blob_pct': round(max(sizes) / sum(sizes) * 100, 1) if sizes else 0.0,
                          'score': round(float(res['scores'][i]), 3), 'yaw_deg': round(th, 1)})
        # 같은 이름이 목적지 inst_r 안에 여럿이면 헷갈린다 — 개수를 적어 둔다
        # 같은 이름이라도 **3D 위치가 겹치면 같은 물체를 두 번 센 것**이다 (검출기가 한 기둥을
        # 'pillar' 와 'pillar column' 으로 두 번 돌려주는 일이 있다). 1 m 보다 멀 때만 다른 개체로 센다.
        import math as _m
        for c in cands:
            n = 1
            for d2 in cands:
                if d2 is c or d2['name'] != c['name']:
                    continue
                if _m.dist(c['center_local'][:2], d2['center_local'][:2]) > 1.0:
                    n += 1
            c['n_same'] = n

        # 고르기 — 사람이 "저거다" 하고 알아볼 수 있는 것을 1 순위로.
        #   처음에는 (유일 > 퍼짐 작은 것 > 가까운 것) 으로 했는데, 화면의 10 % 를 차지하는
        #   score 0.30 짜리 'station map' 이 화면 절반을 채운 기둥을 제치는 일이 생겼다 (ep38).
        #   퍼짐만 보면 **작고 애매한 것**이 이긴다 -> 신뢰도와 크기를 먼저 본다.
        for c in cands:
            c['rank_score'] = (
                (0 if c['n_same'] == 1 else -1.0)      # 유일하면 가점
                + min(c['score'], 1.0) * 2.0           # 검출 신뢰도
                + min(c['area_pct'] / 20.0, 1.0)       # 화면에서 차지하는 크기 (20 % 에서 포화)
                + (0.8 if gem_name and gem_name.lower() in c['name'].lower() else 0.0)  # Gemma 가 고른 것이면 가점
                - min(c['spread_m'] / 3.0, 1.0)        # 퍼져 있으면 감점
                - min(c['dist_m'] / 2.0, 1.0) * 0.5    # 멀면 약간 감점
            )
        cands.sort(key=lambda c: -c['rank_score'])

        # 교차 검증 — Gemma 가 찍은 점을 같은 메시 광선으로 쏴서, 1 등 후보와 3D 위치가 가까운지 본다.
        # 둘 다 "어디를 가리키나" 만 다르고 거리 계산은 같다. 22 ep 실측: 차이 중앙 0.09 m · 19/22 가 0.5 m 안.
        # 어긋나는 것은 '바닥' 처럼 점이 하나로 안 정해지는 대상이라 버리는 게 맞다.
        if gem_pt is not None and cands:
            d = np.array([[(gem_pt[0] - (W - 1) / 2) / FX, (gem_pt[1] - (H - 1) / 2) / FX, 1.0]])
            d = d @ c2w[:3, :3].T
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            loc, _, _ = RI.intersects_location(np.array([org]), d, multiple_hits=False)
            if len(loc):
                gp3 = loc[0]
                rec['gemma_xyz'] = [round(float(t), 2) for t in gp3]
                rec['gemma_goal_m'] = round(float(math.hypot(gp3[0] - gp[0], gp3[1] - gp[1])), 2)
                for c in cands:
                    c['cross_m'] = round(float(math.dist(c['center_local'][:2], gp3[:2].tolist())), 2)
                keep = [c for c in cands if c['cross_m'] <= a.cross_m]
                rec['n_cross_drop'] = len(cands) - len(keep)
                if keep:
                    cands = keep
                else:
                    rec['note'] = f'Gemma 와 검출기가 {cands[0]["cross_m"]} m 어긋남 — 버림'
                    cands = []
        rec['A'] = cands[:a.max_n]
        rec['n_cand'] = len(cands)
        rec['all'] = [{k: c[k] for k in ('name', 'raw_label', 'dist_m', 'area_pct', 'spread_m',
                                         'score', 'n_same', 'rank_score')} for c in cands]  # 왜 그게 뽑혔는지 보려고
        rows.append(rec)
        top = rec['A'][0] if rec['A'] else None
        print(f"  ep{ep:>4s} 후보 {len(cands)} -> " + (f"'{top['name']}' {top['dist_m']} m {top['side']} (퍼짐 {top['spread_m']})" if top else '없음'), flush=True)

    json.dump(rows, open(out / 'stop_cues.json', 'w'), ensure_ascii=False, indent=1)
    n_ok = sum(1 for r in rows if r['A'])
    print(f"\n[68] {n_ok}/{len(rows)} 에피소드에 단서 생성 · 저장 {out}/stop_cues.json")


if __name__ == '__main__':
    main()
