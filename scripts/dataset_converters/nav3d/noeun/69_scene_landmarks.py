"""noeun 69 · 씬 랜드마크 목록 — "역 안에 무엇이 어디에 있나" 를 만든다.

왜: 지금까지는 목적지를 navmesh 에서 **무작위로** 뽑고(22_select_paths) 나중에 "쓸 만한 랜드마크가 2 m 안에
    있나" 를 봤다. 그래서 val_seen 40 중 21 개가 "A 가 멀다/범위가 넓다" 로 탈락했다.
    -> 순서를 뒤집는다. **랜드마크를 먼저 찾고 그 근처를 목적지로 삼으면** 버릴 것이 없다.

어휘를 사람이 짓지 않는다 (2026-11-13 사용자 결정):
  1) Gemma 가 스캔 사진마다 물체 이름을 자유롭게 뽑는다 — 역에 실제로 있는 것에서 나온다
  2) 이름마다 GroundingDINO 검출률·score 를 재서 **검출기가 아는 말만** 남긴다 (검출률 >= 80 % · score >= 0.35)
  3) 같은 물체를 가리키는 여러 표현은 **성적이 가장 좋은 것**을 대표로 삼아 매핑한다
     (실측: bench 0.32 -> wooden bench 0.72 · trash bin 0 % -> trash can 100 % · light fixture 0 % -> ceiling lights 100 %)
  4) 대표 어휘로 전체 사진을 처리 -> SAM 마스크 -> 메시 광선 -> 3D 위치
  5) 여러 사진에 찍힌 같은 물체를 3D 위치로 묶는다 (2 회 이상 본 것만 인정)

단계별로 중간 결과를 남겨 이어 돌릴 수 있다 (--stage).

    python3 scripts/dataset_converters/nav3d/noeun/69_scene_landmarks.py --snaps logs/scene_scan --out logs/scene_landmarks
"""
import argparse, base64, collections, json, math, re, urllib.request
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

W, H, HFOV_DEG = 640, 480, 68.7
FX = (W / 2) / math.tan(math.radians(HFOV_DEG) / 2)

# 랜드마크가 못 되는 것 — MP3D route.yaml 의 names 표가 wall·floor·ceiling·misc·void 를 뺀 것과 같은 이유.
# "저기서 멈춰라" 의 대상이 되려면 **점으로 지목할 수 있어야** 한다. 벽·바닥·천장은 어디에나 있다.
EXCLUDE_HEAD = {
    # 면 — 어디에나 있어 점으로 지목할 수 없다
    'wall', 'floor', 'ceiling', 'tile', 'tiles', 'paving', 'path', 'pavement', 'sidewalk',
    'surface', 'ground', 'strip', 'line', 'lines', 'band', 'stripe', 'edge', 'mat',
    # 건축 부재 — 벽의 일부라 "그 옆에 서라" 가 성립하지 않는다
    'baseboard', 'molding', 'trim', 'frame', 'panel', 'panels', 'partition', 'railing',
    'handrail', 'ceiling', 'beam', 'pipe', 'duct', 'vent', 'grille',
    # 천장 · 배경
    'light', 'lights', 'lighting', 'lamp', 'sky', 'tree', 'trees', 'building', 'buildings',
    'street', 'road', 'background', 'shadow', 'reflection', 'glass', 'window', 'windows',
    # 범위가 넓은 말
    'area', 'space', 'zone', 'section', 'platform', 'concourse', 'corridor', 'hallway',
}

Q_LIST = ("List the distinct physical objects visible in this subway station image that could serve as a landmark "
          "for telling someone where to stop. Use short common English noun phrases (1-3 words). "
          "Answer with ONLY a comma-separated list, at most 10 items. No explanation.")


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--snaps', required=True, help='noeun_scene_scan.py 결과 폴더')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mesh', default='data/data/noeun_eval/navmesh/noeun_station/noeun_station.glb')
    ap.add_argument('--det', default='IDEA-Research/grounding-dino-base')
    ap.add_argument('--sam', default='facebook/sam-vit-base')
    ap.add_argument('--vlm_url', default='http://127.0.0.1:8200/v1/chat/completions')
    ap.add_argument('--vlm_model', default='gemma-4-31B-it')
    ap.add_argument('--stage', default='all', choices=['all', 'names', 'vocab', 'detect', 'cluster'])
    ap.add_argument('--box_thr', type=float, default=0.25)
    # 어휘 통과 기준 (사용자 결정)
    ap.add_argument('--min_rate', type=float, default=0.80, help='Gemma 가 있다고 한 사진 중 검출된 비율')
    ap.add_argument('--min_score', type=float, default=0.35)
    ap.add_argument('--min_said', type=int, default=2, help='이름이 최소 몇 장에서 나와야 통계로 쓸지')
    # 묶기
    ap.add_argument('--merge_m', type=float, default=1.5, help='3D 중심이 이 거리 안이면 같은 물체')
    ap.add_argument('--min_views', type=int, default=2, help='이만큼 다른 사진에 찍혀야 인정 (사용자 결정)')
    ap.add_argument('--max_dist_m', type=float, default=12.0, help='카메라에서 이보다 먼 검출은 버린다')
    return ap.parse_args()


def vlm(url, model, path, prompt, mx=200, timeout=300):
    img = base64.b64encode(open(path, 'rb').read()).decode()
    body = {'model': model, 'max_tokens': mx, 'temperature': 0.0,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img}'}}]}]}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())['choices'][0]['message']['content'].strip()


def norm_name(x):
    x = re.sub(r'^(a|an|the)\s+', '', x.strip().strip('.').lower())
    return re.sub(r'\s+', ' ', x)


COLORS = {'red', 'blue', 'green', 'yellow', 'white', 'black', 'orange', 'purple',
          'brown', 'grey', 'gray', 'silver', 'gold', 'pink', 'beige'}


def singular(h):
    """아주 단순한 단수화. 'es' 를 무조건 자르면 tiles -> til 이 되므로 **s 만** 자르는 것을 기본으로 한다.
    'es' 는 sh/ch/s/x/z 로 끝나는 말에만 적용한다 (boxes -> box, benches -> bench)."""
    if h.endswith('ies') and len(h) > 4:
        return h[:-3] + 'y'
    if h.endswith('es') and len(h) > 3 and h[:-2].endswith(('sh', 'ch', 's', 'x', 'z')):
        return h[:-2]
    if h.endswith('s') and not h.endswith('ss') and len(h) > 3:
        return h[:-1]
    return h


def head_noun(n):
    """표현을 묶는 열쇠 — 마지막 명사(단수) + **색 수식어**.

    색까지 보는 이유: 'blue pillar' 와 'red pillar' 는 다른 물체다. 머리 명사만 보면 하나로 합쳐져
    엉뚱한 곳을 가리키게 된다 (실측: blue arrow -> red arrow 로 매핑되는 사고).
    색이 아닌 수식어('metal' · 'wooden')는 같은 물체의 다른 표현일 때가 많아 묶는다.
    """
    w = n.split()
    if not w:
        return n
    h = singular(w[-1])
    col = next((x for x in w[:-1] if x in COLORS), None)
    return f'{col} {h}' if col else h


def main():
    a = parse()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    snaps = [json.loads(l) for l in (Path(a.snaps) / 'snaps.jsonl').open()]
    sd = Path(a.snaps)
    print(f'[69] 사진 {len(snaps)} 장', flush=True)

    # ── 1) Gemma 가 이름을 뽑는다 ───────────────────────────────────────────
    f_names = out / 'names.json'
    # --stage all 로 다시 돌려도 names.json 이 있으면 **다시 묻지 않는다** (Gemma 질의 40 분을 아낀다).
    # 이름을 새로 뽑고 싶으면 names.json 을 지우거나 --stage names 로 명시한다.
    if a.stage == 'names' or not f_names.exists():
        per = {}
        for i, s in enumerate(snaps):
            p = str(sd / s['rgb'])
            try:
                t = vlm(a.vlm_url, a.vlm_model, p, Q_LIST)
                names = [norm_name(x) for x in t.split(',')]
                per[s['id']] = [n for n in names if 0 < len(n) <= 28]
            except Exception as e:
                per[s['id']] = []
                print(f"  {s['id']} Gemma 실패 {type(e).__name__}", flush=True)
            if (i + 1) % 50 == 0:
                print(f'  이름 뽑기 {i + 1}/{len(snaps)}', flush=True)
                json.dump(per, open(f_names, 'w'), ensure_ascii=False)
        json.dump(per, open(f_names, 'w'), ensure_ascii=False)
    per = json.load(open(f_names))
    cnt = collections.Counter(n for v in per.values() for n in set(v))
    print(f'[69] 이름 {len(cnt)} 종류 (상위: {", ".join(n for n, _ in cnt.most_common(8))})', flush=True)
    if a.stage == 'names':
        return

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
    dp = AutoProcessor.from_pretrained(a.det)
    det = AutoModelForZeroShotObjectDetection.from_pretrained(a.det).to('cuda').eval()

    def detect(img, phrase):
        inp = dp(images=img, text=phrase.rstrip('.') + '.', return_tensors='pt').to('cuda')
        with torch.no_grad():
            o = det(**inp)
        return dp.post_process_grounded_object_detection(
            o, inp.input_ids, threshold=a.box_thr, text_threshold=a.box_thr, target_sizes=[img.size[::-1]])[0]

    # ── 2) 이름별 검출률 -> 어휘 확정 ──────────────────────────────────────
    f_vocab = out / 'vocab.json'
    if a.stage in ('all', 'vocab') or not f_vocab.exists():
        def excluded(n):
            w = [singular(x) for x in n.split()]
            return any(x in EXCLUDE_HEAD for x in w)   # 'red floor strip' 처럼 가운데 말로도 걸린다

        stat = {n: {'said': 0, 'found': 0, 'scores': []} for n in cnt
                if cnt[n] >= a.min_said and not excluded(n)}
        print(f'[69] 제외 범주로 걸러낸 이름 {sum(1 for n in cnt if cnt[n] >= a.min_said) - len(stat)} 개', flush=True)
        by_id = {s['id']: s for s in snaps}
        for n in stat:
            for sid, lst in per.items():
                if n not in lst:
                    continue
                stat[n]['said'] += 1
                img = Image.open(sd / by_id[sid]['rgb']).convert('RGB')
                r = detect(img, n)
                if len(r['scores']):
                    stat[n]['found'] += 1
                    stat[n]['scores'].append(float(max(r['scores'])))
        rows = []
        for n, v in stat.items():
            rate = v['found'] / max(v['said'], 1)
            sc = float(np.mean(v['scores'])) if v['scores'] else 0.0
            rows.append({'name': n, 'said': v['said'], 'found': v['found'],
                         'rate': round(rate, 3), 'score': round(sc, 3),
                         'pass': bool(rate >= a.min_rate and sc >= a.min_score)})
        # 3) 같은 물체의 여러 표현 -> 성적 좋은 것을 대표로 (사용자 요청 매핑)
        groups = collections.defaultdict(list)
        for r in rows:
            groups[head_noun(r['name'])].append(r)
        mapping, vocab = {}, []
        for h, g in groups.items():
            ok = [r for r in g if r['pass']] or g
            best = max(ok, key=lambda r: (r['pass'], r['score'] * r['rate']))
            if not best['pass']:
                continue
            vocab.append(best['name'])
            for r in g:
                mapping[r['name']] = best['name']
        json.dump({'rows': sorted(rows, key=lambda r: -r['said']), 'vocab': sorted(vocab),
                   'mapping': mapping, 'criteria': {'min_rate': a.min_rate, 'min_score': a.min_score}},
                  open(f_vocab, 'w'), ensure_ascii=False, indent=1)
    V = json.load(open(f_vocab))
    print(f"[69] 어휘 {len(V['vocab'])} 개 확정 (검사한 이름 {len(V['rows'])})", flush=True)
    for n, m in sorted(V['mapping'].items()):
        if n != m:
            print(f'     매핑 {n:24s} -> {m}', flush=True)
    if a.stage == 'vocab':
        return

    # ── 4) 확정 어휘로 전체 검출 + 3D 위치 ────────────────────────────────
    spn = SamProcessor.from_pretrained(a.sam); sam = SamModel.from_pretrained(a.sam).to('cuda').eval()
    mesh = trimesh.load(a.mesh, force='mesh', process=False, skip_materials=True)
    RI = mesh.ray
    # GroundingDINO 의 텍스트 한도는 **256 토큰**이다. 어휘 136 개를 한 번에 넣으면 432 토큰이 되어
    #   RuntimeError: The size of tensor a (432) must match the size of tensor b (256)
    # -> 어휘를 덩어리로 쪼개 여러 번 질의하고 결과를 합친다.
    def chunk_vocab(vocab, max_chars=700):
        out, cur, n = [], [], 0
        for w in vocab:
            if cur and n + len(w) + 2 > max_chars:
                out.append(cur); cur, n = [], 0
            cur.append(w); n += len(w) + 2
        if cur:
            out.append(cur)
        return out

    chunks = ['. '.join(c) + '.' for c in chunk_vocab(V['vocab'])]
    print(f'[69] 어휘를 {len(chunks)} 덩어리로 나눠 질의 (DINO 텍스트 한도 256 토큰)', flush=True)
    f_det = out / 'detections.json'
    if a.stage in ('all', 'detect') or not f_det.exists():
        dets = []
        for i, s in enumerate(snaps):
            img = Image.open(sd / s['rgb']).convert('RGB')
            boxes, labs, scores = [], [], []
            for ch in chunks:
                r = detect(img, ch)
                if len(r['scores']) == 0:
                    continue
                boxes += [[float(v) for v in b] for b in r['boxes']]
                labs += list(r.get('text_labels', r.get('labels')))
                scores += [float(x) for x in r['scores']]
            if not boxes:
                continue
            r = {'scores': scores}
            si = spn(img, input_boxes=[boxes], return_tensors='pt').to('cuda')
            with torch.no_grad():
                so = sam(**si, multimask_output=False)
            masks = spn.image_processor.post_process_masks(
                so.pred_masks.cpu(), si['original_sizes'].cpu(), si['reshaped_input_sizes'].cpu())[0].numpy()[:, 0]
            c2w = np.asarray(s['c2w']); org = c2w[:3, 3]
            for j, (lab, bx) in enumerate(zip(labs, boxes)):
                nm = V['mapping'].get(norm_name(str(lab)))
                if nm is None:
                    continue
                vs, us = np.nonzero(masks[j])
                if len(us) == 0:
                    continue
                if len(us) > 400:
                    k = np.random.RandomState(0).choice(len(us), 400, replace=False); us, vs = us[k], vs[k]
                d = np.stack([(us - (W - 1) / 2) / FX, (vs - (H - 1) / 2) / FX, np.ones(len(us))], axis=1) @ c2w[:3, :3].T
                d /= np.linalg.norm(d, axis=1, keepdims=True)
                loc, _, _ = RI.intersects_location(np.repeat([org], len(d), axis=0), d, multiple_hits=False)
                if len(loc) == 0:
                    continue
                p = np.median(loc, axis=0)
                cam_d = float(np.linalg.norm(p - org))
                if cam_d > a.max_dist_m:
                    continue
                dets.append({'snap': s['id'], 'name': nm, 'raw': str(lab),
                             'score': round(float(scores[j]), 3),
                             'xyz': [round(float(t), 2) for t in p],
                             'cam_m': round(cam_d, 2),
                             'area_pct': round(float(masks[j].sum()) / (W * H) * 100, 2),
                             'spread_m': round(float(np.percentile(np.linalg.norm(loc - p, axis=1), 90)), 2)})
            if (i + 1) % 50 == 0:
                print(f'  검출 {i + 1}/{len(snaps)} · 누적 {len(dets)}', flush=True)
                json.dump(dets, open(f_det, 'w'), ensure_ascii=False)
        json.dump(dets, open(f_det, 'w'), ensure_ascii=False)
    dets = json.load(open(f_det))
    print(f'[69] 검출 {len(dets)} 건', flush=True)
    if a.stage == 'detect':
        return

    # ── 5) 같은 물체 묶기 ─────────────────────────────────────────────────
    out_lm = []
    for nm in sorted({d['name'] for d in dets}):
        g = [d for d in dets if d['name'] == nm]
        used = [False] * len(g)
        for i in range(len(g)):
            if used[i]:
                continue
            grp = [g[i]]; used[i] = True
            for j in range(i + 1, len(g)):
                if used[j]:
                    continue
                if math.dist(g[i]['xyz'], g[j]['xyz']) <= a.merge_m:
                    grp.append(g[j]); used[j] = True
            views = len({x['snap'] for x in grp})
            if views < a.min_views:
                continue
            P = np.array([x['xyz'] for x in grp])
            out_lm.append({'name': nm, 'xyz': [round(float(t), 2) for t in np.median(P, axis=0)],
                           'views': views, 'n_det': len(grp),
                           'score': round(float(np.median([x['score'] for x in grp])), 3),
                           'area_pct': round(float(np.median([x['area_pct'] for x in grp])), 2),
                           'spread_m': round(float(np.median([x['spread_m'] for x in grp])), 2),
                           'snaps': sorted({x['snap'] for x in grp})[:6]})
    out_lm.sort(key=lambda r: (r['name'], -r['views']))
    json.dump(out_lm, open(out / 'landmarks.json', 'w'), ensure_ascii=False, indent=1)
    c = collections.Counter(r['name'] for r in out_lm)
    print(f'\n[69] 랜드마크 {len(out_lm)} 개')
    for n, k in c.most_common():
        print(f'     {k:4d}  {n}')
    print(f'저장 {out}/landmarks.json')


if __name__ == '__main__':
    main()
