"""정지 단서 후보(축 1: 구현이 맞는가) 검증 — 학습 없음.

  coverage   : 후보가 1개 이상인 trajectory 비율, 후보 수 분포, 범주 분포
  GT recall  : GT 마지막 문장의 명사가 후보 이름(동의어 포함) 중에 있는 비율 (GT 는 **평가에만** 쓴다)
  precision  : (--vlm_check) gemma 에게 마지막 프레임을 주고 "Is there a {name} close to you{side}?" yes/no
  그림       : 마지막 프레임(정면/30°)에 후보 물체 중심을 찍은 이미지 (--viz N)

    /usr/bin/python scripts/dataset_converters/instruction_labeling/eval/check_stop_cues.py --split val_unseen --subset dev --cue A-grounded-v1 --viz 40 --vlm_check --out logs/instruction_labeling/stop_cues/A-grounded-v1_dev
"""

import argparse
import base64
import collections
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'label'))
import annotate as A  # noqa: E402
from sources import REPO, episode_track, gt_instructions, house_path, items  # noqa: E402
from stop_cues import FX, CX, CY, object_candidates, place_candidates, project  # noqa: E402

SYN = {'couch': {'couch', 'sofa', 'sofas', 'couches', 'loveseat'}, 'dresser': {'dresser', 'drawers', 'chest'},
       'tv': {'tv', 'television', 'monitor', 'screen'}, 'shelf': {'shelf', 'shelves', 'bookshelf', 'bookcase'},
       'door': {'door', 'doors', 'doorway', 'doorways', 'entrance'}, 'stairs': {'stairs', 'staircase', 'steps', 'stair'},
       'table': {'table', 'tables', 'desk'}, 'bench': {'bench', 'seat', 'seats'}, 'plant': {'plant', 'plants', 'tree'},
       'counter': {'counter', 'island', 'countertop'}, 'bathtub': {'bathtub', 'tub', 'bath'}, 'chair': {'chair', 'chairs', 'stool', 'stools'}}


PLACE_RE = {  # GT 마지막 문장이 같은 종류의 장소 단서를 쓰는가 (평가 전용)
    'stairs_top': lambda c: r'\b(top|head|up)\b[^.]*\b(stairs?|steps|staircase)\b|\b(stairs?|steps|staircase)\b[^.]*\btop\b',
    'stairs_bottom': lambda c: r'\b(bottom|foot|base)\b[^.]*\b(stairs?|steps|staircase)\b|\b(stairs?|steps|staircase)\b[^.]*\b(bottom|foot)\b',
    'doorway': lambda c: r'\b(doorway|door|entrance|threshold|entry|opening|archway)\b',
    'entrance': lambda c: r'\b(inside|entrance|enter|doorway|into)\b',
    'hall_end': lambda c: r'\bend of the hall(way)?\b|\bhall(way)?\b[^.]*\bend\b',
    'room': lambda c: r'\b' + re.escape(c['name'].split()[-1]) + r's?\b',
}


def last_sentence(t):
    s = [x for x in re.split(r'(?<=[.!?])\s+', t.strip()) if x.strip()]
    return s[-1].lower() if s else ''


def words_of(name):
    return SYN.get(name, {name, name + 's'})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='val_unseen')
    ap.add_argument('--subset', default='dev')
    ap.add_argument('--cue', default='A-grounded-v1')
    ap.add_argument('--viz', type=int, default=30)
    ap.add_argument('--vlm_check', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(HERE.parent / 'configs' / 'route.yaml'))['stop_cue'][a.cue]
    out = REPO / a.out
    (out / 'viz').mkdir(parents=True, exist_ok=True)
    its = A.select_subset(items(a.split), a.subset)
    gt = gt_instructions(a.split)
    by_t = collections.defaultdict(list)
    for it in its:
        by_t[it['ep']['trajectory_id']].append(it)
    rows, n_viz = [], 0
    rigs = ['125cm_0deg', '125cm_30deg']
    for tid, group in sorted(by_t.items()):
        it = group[0]
        tr = episode_track(it, rigs[0])
        T = len(tr['action'])
        poses = {r: tr['pose_of'](r)[T - 1] for r in rigs}
        depths = {r: cv2.imread(str(tr['depth'](T - 1, r)), -1).astype(np.float32) / 1000.0 for r in rigs}
        gts = [gt[g['ep']['episode_id']] for g in group]
        if cfg['kind'] == 'A':
            cands = object_candidates(it['ep'], house_path(it['ep']), poses, depths, cfg)
        elif cfg['kind'] in ('P', 'R'):
            cands = place_candidates(it['ep'], house_path(it['ep']), tr['pose_of'](rigs[0]), cfg)
            for c in cands:
                c['name'] = c['phrase']
        else:
            assert False, f"unreachable kind={cfg['kind']!r}"
        hits = []
        for g in gts:
            ls = last_sentence(g)
            if cfg['kind'] == 'A':
                hits.append(any(any(re.search(rf'\b{w}\b', ls) for w in words_of(c['name'])) for c in cands))
            else:
                hits.append(any(re.search(PLACE_RE[c['type']](c), ls) is not None for c in cands))
        keys = ('name', 'dist_m', 'side', 'visible_in') if cfg['kind'] == 'A' else ('name', 'type', 'kind')
        rows.append({'trajectory_id': tid, 'scene': it['scene'], 'n': len(cands),
                     'cands': [{k: c[k] for k in keys} for c in cands],
                     'gt_hit_any': any(hits), 'gt_hit_frac': float(np.mean(hits)), 'top1': cands[0]['name'] if cands else None,
                     'img': str(tr['img'](T - 1, rigs[0])), 'gt_last': [last_sentence(g) for g in gts]})
        if n_viz < a.viz:
            ims = []
            for r in rigs:
                im = cv2.imread(str(tr['img'](T - 1, r)))
                for c in (cands if cfg['kind'] == 'A' else []):
                    u, v, Z = project(np.array([c['center_local']]), poses[r])
                    if Z[0] > 0 and 0 <= u[0] < 640 and 0 <= v[0] < 480:
                        cv2.circle(im, (int(u[0]), int(v[0])), 7, (0, 255, 255), -1)
                        cv2.putText(im, f"{c['name']} {c['dist_m']}m {c['side']}", (int(u[0]) + 8, int(v[0])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(im, r, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ims.append(im)
            canvas = np.concatenate(ims, 1)
            if cfg['kind'] != 'A':
                for j, c in enumerate(cands):
                    cv2.putText(canvas, f"{c['kind']}: {c['name']}", (10, 60 + 28 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(canvas, ('GT: ' + gts[0][-110:]).encode('ascii', 'ignore').decode(), (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.imwrite(str(out / 'viz' / f'{it["scene"]}_traj{tid}.jpg'), canvas)
            n_viz += 1

    if a.vlm_check:
        from openai import OpenAI
        client = OpenAI(base_url='http://localhost:8100/v1', api_key='x')
        for r in rows:
            if not r['cands']:
                continue
            c = r['cands'][0]
            b64 = base64.b64encode(open(r['img'], 'rb').read()).decode()
            if cfg['kind'] == 'A':
                side = {'left': ' on your left', 'right': ' on your right', 'ahead': ' in front of you'}[c['side']]
                q = f'This is what a robot sees where it stopped. Is there a {c["name"]} within about two meters of the robot{side}? Answer only yes or no.'
            else:
                q = f'This is what a robot sees where it stopped. Is the robot standing {c["name"]}? Answer only yes or no.'
            resp = client.chat.completions.create(model='gemma-4-31B-it', max_tokens=3, temperature=0.0, messages=[
                {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                                             {'type': 'text', 'text': q}]}])
            r['vlm_yes'] = resp.choices[0].message.content.strip().lower().startswith('yes')

    n = len(rows)
    cats = collections.Counter(r['top1'] for r in rows if r['top1'])
    summ = {'trajectories': n, 'coverage': round(sum(r['n'] > 0 for r in rows) / n, 3),
            'n_cands_dist': dict(collections.Counter(min(r['n'], 4) for r in rows)),
            'top1_names': dict(cats.most_common(12)),
            'gt_recall_traj_any': round(np.mean([r['gt_hit_any'] for r in rows]), 3),
            'gt_recall_sentence': round(np.mean([r['gt_hit_frac'] for r in rows]), 3),
            'gt_recall_given_cand': round(np.mean([r['gt_hit_frac'] for r in rows if r['n']]), 3)}
    if a.vlm_check:
        v = [r['vlm_yes'] for r in rows if 'vlm_yes' in r]
        summ['vlm_precision_top1'] = round(float(np.mean(v)), 3)
        summ['vlm_checked'] = len(v)
    (out / 'summary.json').write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    with open(out / 'rows.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
