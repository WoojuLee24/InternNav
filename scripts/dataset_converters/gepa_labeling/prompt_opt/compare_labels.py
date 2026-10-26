"""relabel 라벨 비교 표: 2x2 (label→GT, label→label, GT→label=정보량) + 문장 지표 (val_unseen 1,839, 규칙 채점기).

    /home/irteam/venvs/gepa/bin/python scripts/dataset_converters/gepa_labeling/prompt_opt/compare_labels.py
2x2 는 eval_dashboard reader(build_task) 로 checkpoints/relabel_half 의 로그에서 읽는다 (같은 셀이 여러 번이면 가장 최근 완료본).
문장 지표: 사실 오류(규칙: 회전 방향·순서, 계단, "stop in the <방>", 최소 정보), 형제 첫 3단어 같음·내용어 겹침,
사람(GT) 과의 분포 차이(끝맺음·설명 방식·길이, TV×100), 평균 단어 수, 어휘 수.
출력: 표(stdout) + logs/gepa_labeling/compare_labels.md
"""
import collections
import glob
import gzip
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scorer import REPO, diversity, knob_of, path_facts, rule_fact_issues  # noqa: E402

sys.path.insert(0, str(REPO / 'scripts/eval_dashboard'))
import data  # noqa: E402
import config as il_config  # noqa: E402
from sources import gt_instructions, items  # noqa: E402

LAB = Path('/home/irteam/data-vol2/relabel_half/labels/val_unseen')
CK = '/home/irteam/data-vol2/checkpoints/relabel_half/'
B1 = 'b0_slight-omit_fewshot-3_keyframe_deg-0_rooms-house-v2'  # b1 과 같은 생성 (2x2 는 이 이름으로 돌았다)
J = 'b1_facts-memo-v1_knobs-v2_delivery-joint3-v1'
ROWS = [  # (표 이름, 라벨 파일 이름, 2x2 셀 이름)
    ('v278 (v278_reproduce)', 'v278_reproduce', 'v278_reproduce'),
    ('v278fix', 'v278fix', 'v278fix'),
    ('b0', 'b0', 'b0'),
    ('b1', 'b1', B1),
    ('b1 + 정지 단서 A', 'b1_stop_cue-A-grounded-v1', B1 + '_stop_cue-A-grounded-v1'),
    ('b1 + 형제 서술방식 (실험1)', 'b1_sibling-profiles-v1_fewshot_match-gtstyle-v1', 'b1_sibling-profiles-v1_fewshot_match-gtstyle-v1'),
    ('b1 + 출발 주변사진 (실험2)', 'b1_select-keyframes-startsurround-v1', 'b1_select-keyframes-startsurround-v1'),
    ('b1 + 주변사진 + 서술방식v2 (실험3)', 'b1_select-keyframes-startsurround-v1_sibling-profiles-v2_fewshot_match-gtstyle-v1',
     'b1_select-keyframes-startsurround-v1_sibling-profiles-v2_fewshot_match-gtstyle-v1'),
    ('b1 + pathrule v1', 'b1_select-keyframes-startsurround-v1_pathrule-v1_fewshot_match-gtstyle-v1',
     'b1_select-keyframes-startsurround-v1_pathrule-v1_fewshot_match-gtstyle-v1'),
    ('작성자 B (화살표)', 'b1_overlay-arrow-v1_motion-none', 'b1_overlay-arrow-v1_motion-none'),
    ('작성자 C (평면도)', 'b1_select-topdown-v1_motion-none', 'b1_select-topdown-v1_motion-none'),
    ('대조군2 사실 메모', 'b1_facts-memo-v1', 'b1_facts-memo-v1'),
    ('대조군3 + 손잡이', 'b1_facts-memo-v1_knobs-v1', 'b1_facts-memo-v1_knobs-v1'),
    ('대조군3 + GEPA (gemma)', 'b1_facts-memo-v1_knobs-v1_writer-gepa-v4-c3-gemma', 'b1_facts-memo-v1_knobs-v1_writer-gepa-v4-c3-gemma'),
    ('대조군3 + GEPA (Sonnet)', 'b1_facts-memo-v1_knobs-v1_writer-gepa-v4-c3-sonnet', 'b1_facts-memo-v1_knobs-v1_writer-gepa-v4-c3-sonnet'),
    ("'전' 3개 한번에", J, J),
    ("'후' 3개 한번에 + GEPA", J + '_writer-gepa-v3-joint-gemma', J + '_writer-gepa-v3-joint-gemma'),
]
AX = {'ending': ['stop', 'wait', 'none'], 'granularity': ['turn', 'mixed', 'landmark'], 'length': ['short', 'medium', 'long']}
STOPW = set('the a an and to of in on at by you your into from with then until is are it this that there'.split())


def sr(pattern):
    """가장 최근에 1,839 개 모두 끝난 로그의 SR. 없으면 진행 중이면 (n), 아무것도 없으면 None."""
    best = None
    for d in sorted(glob.glob(pattern)):
        t = data.build_task(Path(d))
        e = t['_episodes']
        if not e:
            continue
        v = 100 * sum(x['success'] for x in e) / len(e)
        best = f'{v:.1f}' if len(e) == 1839 else f'{v:.1f}({len(e)})'
    return best or '—'


def two_by_two(cell):
    lg = sr(f'{CK}dualvln_mini_half_s1.train.{cell}_eval.gt_internvla-n1-system2_*/logs/relabel_half_dualvln_mini_half_s1.train.{cell}_eval.gt_internvla-n1-system2')
    ll = sr(f'{CK}dualvln_mini_half_s1.train.{cell}_eval.gt_internvla-n1-system2_*/logs/relabel_half_dualvln_mini_half_s1.train.{cell}_eval.{cell}_internvla-n1-system2')
    gl = sr(f'{CK}dualvln_mini_half_s1.train.gt_eval.gt_internvla-n1-system2_*/logs/relabel_half_dualvln_mini_half_s1.train.gt_eval.{cell}_internvla-n1-system2')
    return lg, ll, gl


def text_metrics(texts, eps, rc, facts, by):
    fact_err = sum(bool(rule_fact_issues(texts[k], facts[k])) for k in eps) / len(eps)
    f3s, jacs = [], []
    for ks in by.values():
        _, f3, jac = diversity([texts[k] for k in ks])
        f3s.append(f3)
        jacs.append(jac)
    kn = {k: knob_of(texts[k], eps[k], rc) for k in eps}
    dist = {a: [sum(kn[k][a] == v for k in eps) / len(eps) for v in vs] for a, vs in AX.items()}
    words = sum(len(texts[k].split()) for k in eps) / len(eps)
    vocab = len({w for k in eps for w in re.findall(r"[a-z']+", texts[k].lower()) if w not in STOPW and len(w) > 2})
    first = collections.Counter(texts[k].lower().split()[0] if texts[k].split() else '' for k in eps).most_common(3)
    return {'fact': 100 * fact_err, 'f3': 100 * sum(f3s) / len(f3s), 'jac': sum(jacs) / len(jacs), 'dist': dist,
            'words': words, 'vocab': vocab, 'first': first}


def main():
    rc = il_config.load('b1', [])['route']
    eps = {it['ep']['episode_id']: it['ep'] for it in items('val_unseen')}
    by = collections.defaultdict(list)
    for k, ep in eps.items():
        by[ep['trajectory_id']].append(k)
    facts = {k: path_facts(ep, rc) for k, ep in eps.items()}
    gt = gt_instructions('val_unseen')
    ref = text_metrics({k: gt[k].strip() for k in eps}, eps, rc, facts, by)
    tv = lambda m: ' / '.join(f'{50 * sum(abs(x - y) for x, y in zip(m["dist"][a], ref["dist"][a])):.0f}' for a in AX)
    gg = sr(f'{CK}dualvln_mini_half_s1.train.gt_eval.gt_internvla-n1-system2_*/logs/relabel_half_dualvln_mini_half_s1.train.gt_eval.gt_internvla-n1-system2')
    hdr = '| 라벨 | label→GT | label→label | 정보량 (GT→label) | 사실 오류 (규칙) | 형제 첫3 같음 | 형제 겹침 | 사람과 분포 차이 끝맺음/설명/길이 | 평균 단어 | 어휘 | 첫 단어 상위 |'
    out = [hdr, '|' + '---|' * 11,
           f'| **GT (사람)** | 42.5 | — | {gg} (463 ep 만 기록) | {ref["fact"]:.1f} % | {ref["f3"]:.1f} % | {ref["jac"]:.2f} | 0 / 0 / 0 | {ref["words"]:.1f} | {ref["vocab"]} | '
           + ', '.join(f'{w} {100 * c / len(eps):.0f}%' for w, c in ref['first']) + ' |']
    for name, lab, cell in ROWS:
        f = LAB / f'val_unseen_{lab}.json.gz'
        if not f.exists():
            out.append(f'| {name} | 라벨 파일 없음 |' + ' |' * 9)
            continue
        d = {e['episode_id']: e['instruction']['instruction_text'].strip() for e in json.load(gzip.open(f))['episodes']}
        if set(d) != set(eps):
            out.append(f'| {name} | episode 수 {len(d)} (≠ 1,839) |' + ' |' * 9)
            continue
        m = text_metrics(d, eps, rc, facts, by)
        lg, ll, gl = two_by_two(cell)
        out.append(f'| {name} | {lg} | {ll} | {gl} | {m["fact"]:.1f} % | {m["f3"]:.1f} % | {m["jac"]:.2f} | {tv(m)} | {m["words"]:.1f} | {m["vocab"]} | '
                   + ', '.join(f'{w} {100 * c / len(eps):.0f}%' for w, c in m['first']) + ' |')
        print(out[-1], flush=True)
    dst = REPO / 'logs/gepa_labeling/compare_labels.md'
    dst.write_text('\n'.join(out) + '\n')
    print('\n'.join(out[:3]))
    print(f'-> {dst}')


if __name__ == '__main__':
    main()
