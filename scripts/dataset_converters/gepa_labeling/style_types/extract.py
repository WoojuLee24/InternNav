"""0단계: 사람(GT) 문장마다 말투 특징을 뽑는다.

  --stage rules : 코드로 세는 겉모양 (길이·첫머리·끝맺음·정지 위치를 무엇으로 말하나·회전을 경로보다 덜 말했나·숫자·대문자 ...)
  --stage judge : gemma 가 고르는 의미 축 (목적지만 말하는 문장인가: yes / no), 프롬프트 2개로 두 번
출력: <out_dir>/<split>_rules.jsonl, <split>_judge.jsonl (episode 하나에 한 줄)

    /home/irteam/venvs/gepa/bin/python scripts/dataset_converters/gepa_labeling/style_types/extract.py --split train --stage rules
    /home/irteam/venvs/gepa/bin/python scripts/dataset_converters/gepa_labeling/style_types/extract.py --split train --stage judge
"""
import argparse
import gzip
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'scripts/dataset_converters/instruction_labeling/label'))
import config as il_config  # noqa: E402
from route_words import route_geometry, said  # noqa: E402

CFG = yaml.safe_load(open(HERE.parent / 'configs' / 'style_types.yaml'))


def load_gt(split: str) -> list:
    return json.load(gzip.open(REPO / CFG['raw_root'] / split / f'{split}.json.gz'))['episodes']


def sents(t: str) -> list:
    return [x for x in re.split(r'(?<=[.!?])\s+', t.strip()) if x]


def bin_of(x: float, edges: list) -> int:
    return sum(x > e for e in edges)


def rule_features(e: dict, rc: dict) -> dict:
    R = CFG['rules']
    t = e['instruction']['instruction_text'].strip()
    tl = t.lower()
    words = re.findall(r"[a-z']+", tl)
    ss = sents(t)
    head = tl.lstrip(' "\'')
    opener = next((name for name, rx in R['openers'] if re.match(rx, head)), 'other')
    last = ss[-1].lower() if ss else ''
    ending = next((name for name, rx in R['endings'] if re.search(rx, last)), 'none')
    g = route_geometry(e)
    path_turns = sum(said(a, rc) for a in g['turns'].values()) + int(said(g['initial'], rc))
    told = len(re.findall(R['turn_words'], tl))
    n_cmd, n_lm = len(re.findall(R['turn_cmd'], tl)), len(re.findall(R['landmark_phrase'], tl))
    if abs(g['initial']) >= R['start_turn_min_deg']:
        st = 'yes' if ss and re.search(R['turn_words'], ss[0].lower()) else 'no'
    else:
        st = 'na'  # 출발 때 돌 필요가 없는 경로
    lo, lr = bool(re.search(R['object_words'], last)), bool(re.search(R['room_words'], last))
    stop_ref = 'both' if lo and lr else 'object' if lo else 'room' if lr else 'none'  # 마지막 문장이 정지 위치를 무엇으로 말하나
    letters = [c for c in t if c.isalpha()]
    return {
        'episode_id': e['episode_id'], 'trajectory_id': e['trajectory_id'], 'text': t,
        'n_words': len(words), 'log_words': round(math.log(max(1, len(words))), 4), 'n_sents': len(ss),
        'length': ['short', 'medium', 'long'][bin_of(len(words), R['length_bins'])],
        'sent_bin': ['1', '2-3', '4+'][bin_of(len(ss), R['sent_bins'])],
        'opener': opener, 'ending': ending, 'stop_ref': stop_ref,
        'path_turns': path_turns, 'told_turns': told,
        'turn_ratio': round(min(2.0, told / path_turns) if path_turns else (1.0 if told == 0 else 2.0), 4),
        'start_turn_said': st,
        'n_turn_cmd': n_cmd, 'n_landmark': n_lm,
        'step_share': round(n_cmd / (n_cmd + n_lm), 4) if n_cmd + n_lm else 0.5,  # 1 = 회전 명령만, 0 = 지나는 것만, 섞이면 중간
        'numbers': bool(re.search(R['number_words'], tl)),
        'allcaps': bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8,
        'then': bool(re.search(r'\bthen\b', tl)),
        'n_objects': len(re.findall(R['object_words'], tl)), 'n_rooms': len(re.findall(R['room_words'], tl)),
        'initial_turn_deg': round(g['initial'], 1),
    }


def parse_judge(s: str) -> dict:
    m = re.search(r'\{.*\}', s or '', re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    ax = CFG['judge']['axes']
    return {k: (str(d.get(k, '')).strip().lower() if str(d.get(k, '')).strip().lower() in v else None) for k, v in ax.items()}


def judge(rows: list) -> list:
    from openai import OpenAI
    J = CFG['judge']
    client = OpenAI(base_url=J['base_url'], api_key=J['api_key'])

    def one(r):
        out = {'episode_id': r['episode_id']}
        for name, tmpl in J['variants'].items():
            res = None
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=J['model'], messages=[{'role': 'user', 'content': tmpl.format(text=r['text'].replace('"', "'"))}],
                        max_tokens=J['max_tokens'], temperature=J['temperature'], seed=attempt)
                    res = parse_judge(resp.choices[0].message.content)
                    if all(v is not None for v in res.values()):
                        break
                except Exception as ex:  # noqa: BLE001 — 기록하고 재시도
                    res = {'error': repr(ex)[:200]}
            out[name] = res
        return out

    with ThreadPoolExecutor(J['workers']) as ex:
        return list(ex.map(one, rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', required=True, choices=['train', 'val_seen', 'val_unseen'])
    ap.add_argument('--stage', required=True, choices=['rules', 'judge'])
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    out = REPO / CFG['out_dir']
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == 'rules':
        rc = il_config.load(CFG['route_method'], [])['route']
        rows = [rule_features(e, rc) for e in load_gt(a.split)]
        rows = rows[:a.limit] if a.limit else rows
        dst = out / f'{a.split}_rules.jsonl'
    elif a.stage == 'judge':
        rows = [json.loads(l) for l in open(out / f'{a.split}_rules.jsonl')]
        rows = judge(rows[:a.limit] if a.limit else rows)
        dst = out / f'{a.split}_judge{"_limit%d" % a.limit if a.limit else ""}.jsonl'
    else:
        assert False, f'unreachable stage={a.stage!r}'
    with open(dst, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{dst} ({len(rows)})')


if __name__ == '__main__':
    main()
