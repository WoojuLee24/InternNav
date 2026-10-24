"""3단계: GEPA adapter — 후보 글 두 개(style_instruction, facts_header) 로 경로마다 형제 3문장을 만들고 채점한다.

문장 생성은 instruction_labeling 과 같은 코드(e2e.generate → facts.build_request_facts, knobs.knob_style) 를 쓰고,
후보 글은 cfg 사본의 style.knobs.instruction / route.facts.header 만 바꿔 넣는다 → 최적화와 실제 라벨 생성이 같은 경로.
"""

import base64
import copy
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from gepa import EvaluationBatch, Image

from scorer import CFG, score_paths

import e2e  # noqa: E402  (scorer 가 instruction_labeling/label 을 sys.path 에 넣는다)
from sources import episode_track  # noqa: E402

PLACEHOLDERS = ('{granularity}', '{length}', '{ending}')


def apply_candidate(cfg: dict, cand: dict) -> dict:
    c = copy.deepcopy(cfg)
    c['style']['knobs']['instruction'] = cand['style_instruction']
    c['route']['facts']['header'] = cand['facts_header']
    return c


def check_candidate(cand: dict) -> list:
    L = CFG['gepa']['limits']
    bad = []
    si, fh = cand['style_instruction'], cand['facts_header']
    for p in PLACEHOLDERS:
        if si.count(p) != 1:
            bad.append(f'style_instruction must contain {p} exactly once')
    if set(re.findall(r'\{[^}]*\}', si)) - set(PLACEHOLDERS):
        bad.append('style_instruction has curly braces other than the three placeholders')
    if '{' in fh or '}' in fh:
        bad.append('facts_header must not contain curly braces')
    if len(si.split()) > L['style_instruction_max_words']:
        bad.append(f'style_instruction is longer than {L["style_instruction_max_words"]} words')
    if len(fh.split()) > L['facts_header_max_words']:
        bad.append(f'facts_header is longer than {L["facts_header_max_words"]} words')
    return bad


def contact_sheet(content: list, out: Path, h: int = 200, per_row: int = 4) -> Path:
    """요청 content 의 사진들을 한 장으로 (사례당 1장 → reflection 요청이 이미지 한도 안)."""
    tiles = []
    for x in content:
        if x['type'] != 'image_url':
            continue
        img = cv2.imdecode(np.frombuffer(base64.b64decode(x['image_url']['url'].split(',', 1)[1]), np.uint8), cv2.IMREAD_COLOR)
        t = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
        cv2.putText(t, str(len(tiles) + 1), (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        tiles.append(t)
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    tiles += [np.full_like(tiles[0], 255)] * ((-len(tiles)) % per_row)
    sheet = np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return out


class NavLabelAdapter:
    """DataInst = {'traj': trajectory_id, 'items': [sources item, ...] (형제)}."""

    propose_new_texts = None  # GEPA 기본 제안기(instruction_proposal) 를 쓴다

    def __init__(self, cfg: dict, client, run_dir: Path):
        self.cfg, self.client, self.run_dir = cfg, client, run_dir
        self.workers = CFG['gepa']['workers']
        self.rc = cfg['route']

    def evaluate(self, batch, candidate, capture_traces=False):
        bad = check_candidate(candidate)
        if bad:  # 규칙을 어긴 후보: 생성하지 않고 0 점, 이유만 돌려준다
            fb = 'INVALID CANDIDATE: ' + '; '.join(bad)
            return EvaluationBatch(outputs=[None] * len(batch), scores=[0.0] * len(batch),
                                   trajectories=[{'feedback': fb, 'sheet': None, 'path': d} for d in batch] if capture_traces else None)
        cfg = apply_candidate(self.cfg, candidate)
        if cfg['delivery']['type'] == 'joint':  # 형제를 한 요청으로: 경로마다 joint 호출 (annotate 용 캐시는 쓰지 않는다 — 후보마다 새로 만든다)
            return self._evaluate_joint(batch, cfg, capture_traces)
        jobs = [(i, it) for i, d in enumerate(batch) for it in d['items']]

        def gen(job):
            i, it = job
            track = episode_track(it, cfg['frames']['rig'])
            try:
                r = e2e.generate(self.client, cfg, it['ep'], track)
            except Exception as ex:  # noqa: BLE001 — 후보 글이 format 을 깨면 여기서 걸린다
                r = {'text': '', 'error': repr(ex)[:200], 'inputs': {}}
            return i, it, r

        with ThreadPoolExecutor(self.workers) as ex:
            res = list(ex.map(gen, jobs))
        per_path = [[] for _ in batch]
        for i, it, r in res:
            per_path[i].append((it, r))
        paths = [{'eps': [it['ep'] for it, _ in pp], 'texts': [r['text'] for _, r in pp],
                  'knobs': [r.get('inputs', {}).get('knobs') for _, r in pp]} for pp in per_path]
        sc = score_paths(paths, self.rc)
        trajs = None
        if capture_traces:
            trajs = []
            for d, pp, s in zip(batch, per_path, sc):
                it0 = pp[0][0]
                content, _, _ = e2e.build_request(cfg, it0['ep'], episode_track(it0, cfg['frames']['rig']))
                sheet = contact_sheet(content, self.run_dir / 'sheets' / f'{d["traj"]}.jpg')
                trajs.append({'feedback': s['feedback'], 'sheet': str(sheet), 'path': d})
        outputs = [{'texts': p['texts'], 'knobs': p['knobs'], 'fact': s['fact'], 'knob': s['knob'], 'div': s['div']} for p, s in zip(paths, sc)]
        return EvaluationBatch(outputs=outputs, scores=[s['score'] for s in sc], trajectories=trajs)

    def _evaluate_joint(self, batch, cfg, capture_traces):
        import joint

        def gen(d):
            it0 = d['items'][0]
            try:
                r = joint._generate_traj(self.client, cfg, it0['ep'], episode_track(it0, cfg['frames']['rig']), len(d['items']))
            except Exception as ex:  # noqa: BLE001 — 후보 글이 format 을 깨면 여기서 걸린다
                r = {'texts': [''] * len(d['items']), 'knobs': [None] * len(d['items']), 'error': repr(ex)[:200]}
            return r

        with ThreadPoolExecutor(self.workers) as ex:
            res = list(ex.map(gen, batch))
        paths = [{'eps': [it['ep'] for it in d['items']], 'texts': r['texts'], 'knobs': r['knobs']} for d, r in zip(batch, res)]
        sc = score_paths(paths, self.rc)
        trajs = None
        if capture_traces:
            trajs = []
            for d, s in zip(batch, sc):
                it0 = d['items'][0]
                content, _, _, _ = joint.build_joint_request(cfg, it0['ep'], episode_track(it0, cfg['frames']['rig']), len(d['items']))
                sheet = contact_sheet(content, self.run_dir / 'sheets' / f'{d["traj"]}.jpg')
                trajs.append({'feedback': s['feedback'], 'sheet': str(sheet), 'path': d})
        outputs = [{'texts': p['texts'], 'knobs': p['knobs'], 'fact': s['fact'], 'knob': s['knob'], 'div': s['div']} for p, s in zip(paths, sc)]
        return EvaluationBatch(outputs=outputs, scores=[s['score'] for s in sc], trajectories=trajs)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        recs = []
        for t, o in zip(eval_batch.trajectories, eval_batch.outputs):
            r = {'Feedback': t['feedback']}
            if t['sheet']:
                r = {'Photos along the path (numbered in order; last = destination)': Image(path=t['sheet']),
                     'Requested styles and generated instructions': '\n'.join(f'{k}: {x}' for k, x in zip(o['knobs'], o['texts'])),
                     'Feedback': t['feedback']}
            recs.append(r)
        return {c: recs for c in components_to_update}


def reflection_lm(client, model: str):
    R = CFG['gepa']['reflection']

    def call(prompt):
        msgs = [{'role': 'user', 'content': prompt}] if isinstance(prompt, str) else prompt
        r = client.chat.completions.create(model=model, messages=msgs, max_tokens=R['max_tokens'], temperature=R['temperature'])
        return r.choices[0].message.content or ''
    return call
