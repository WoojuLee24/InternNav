"""noeun · 사람이 리포트에서 고친 내용(`corrections.json`)을 랜드마크 목록에 적용한다.

왜 파일 하나로 모으나:
    같은 결정을 70(`.house`)과 73(목적지 후보)이 **똑같이** 반영해야 한다.
    명령줄 인자로 따로 넘기면 한쪽만 고쳐지는 사고가 난다. 그래서 한 파일을 둘이 같이 읽는다.

파일 모양 (리포트의 "고친 내용" 상자가 그대로 내려준다):
    {
      "exclude_types": ["pillar", "car"],      # 머리 명사 — white pillar · concrete pillar 전부
      "exclude_names": ["white sign"],         # 종류 이름 그대로 — green sign 은 남는다
      "exclude_ids":   [372, 418],             # 랜드마크 하나하나 (landmarks.json 안의 순서, 0 부터)
      "rename":        {"digital sign": "exit sign"},   # 그 종류 전부의 대표 이름
      "rename_ids":    {"501": "ticket gate"},          # 랜드마크 하나의 대표 이름
      "aliases":       {"exit sign": ["station sign"]}, # 그 종류 전부의 다른 이름
      "alias_ids":     {"501": ["gate"]},               # 랜드마크 하나의 다른 이름
      "common":        {"exit sign": "sign"},           # 공통 class (.house 의 mpcat40)
      "goal_exclude_types": ["arrow"],  # 73 목적지에서만 뺀다 (.house 에는 남음). 머리 명사 규칙은 exclude_types 와 같음
      "add":           [{"id": 100000, "name": "bench", "xyz": [x,y,z], "points": [[x,y,z],...]}]
                       # 걸어 보기 도구에서 화면을 눌러 추가한 물체. 점 = 분할 모양을 메쉬에 쏜 3D 점.
                       # 다른 이름·공통 class 는 alias_ids·common_ids 에 같은 id 로 들어간다.
    }

한 물체가 이름을 여럿 갖는 이유:
    같은 물건을 사람마다 다르게 부른다. "exit sign" 이라고만 적어 두면 문장이 늘 한 가지로만 나온다.
    `names` 를 여럿 두면 문장 만들 때 골라 쓸 수 있고, `common` 으로 묶으면
    "어느 표지판이든" 같은 넓은 지시도 가능하다. `.house` 는 범주마다 이름 두 개
    (`raw_name` · `mpcat40_name`)를 담을 수 있어 대표 이름과 공통 class 가 그대로 들어간다.

적용 순서는 **이름 먼저, 빼기 나중**이다. 리포트에서 보이는 이름이 곧 거르는 대상이기 때문이다.
(`digital sign` 을 `exit sign` 으로 고친 뒤 `sign` 을 빼면 둘 다 빠진다 — 화면과 같은 결과다.)
"""
import collections
import json
import re
from pathlib import Path


def load(path):
    """corrections.json -> dict. 경로가 비었으면 빈 결정."""
    empty = {'exclude_types': [], 'exclude_names': [], 'exclude_ids': [], 'rename': {}, 'rename_ids': {}}
    if not path:
        return empty
    c = json.load(open(path))
    for k, v in empty.items():
        c.setdefault(k, v)
    return c


def apply(lms, c, log=print):
    """랜드마크 목록에 적용해 **남은 것**을 돌려준다. `lms` 의 각 항목에 'id' 가 있어야 한다.
    `add` (걸어 보기 도구에서 카메라 화면을 눌러 추가한 물체) 는 `lms` 에 **제자리로** 덧붙인다 —
    부르는 쪽이 같은 목록으로 뺀 것까지 그릴 수 있게. 추가한 것도 아래 이름 고치기·빼기를 똑같이 받는다."""
    known = {l['id'] for l in lms}
    for x in c.get('add') or []:
        if int(x['id']) not in known:
            lms.append({'id': int(x['id']), 'name': str(x['name']).strip().lower(), 'xyz': x['xyz'],
                        'points': x.get('points') or [x['xyz']], 'views': x.get('views', 1),
                        'score': x.get('score', 1.0), 'n_det': 0, 'added': True})
    n_add = len(lms) - len(known)
    if n_add:
        log(f'[보정] 화면에서 추가한 물체 {n_add} 개')
    all_ids = {l['id'] for l in lms}
    n_all = len(lms)

    # 1) 이름 고치기
    ri = {int(k): str(v).strip().lower() for k, v in (c.get('rename_ids') or {}).items()}
    rn = {str(k).strip().lower(): str(v).strip().lower() for k, v in (c.get('rename') or {}).items()}
    n_rn = 0
    for l in lms:
        new = ri.get(l['id']) or rn.get(l['name'].strip().lower())
        if new and new != l['name']:
            l.setdefault('orig_name', l['name'])   # 검출 점 구름은 원래 이름으로 묶여 있다
            l['name'] = new
            n_rn += 1
    if n_rn:
        log(f'[보정] 이름 고침 {n_rn} 개 (종류 {len(rn)} · 낱개 {len(ri)})')
        for o, nw in sorted(rn.items()):
            log(f'[보정]   {o} -> {nw}')

    # include_ids: 종류째 · 이름째 빼기에서 이 랜드마크만 살린다 (걸어 보기 도구에서 되살린 것). 없으면 기존 동작.
    inc = {int(x) for x in (c.get('include_ids') or [])}

    # 2) 머리 명사로 빼기
    ex = [w.strip().lower() for w in (c.get('exclude_types') or []) if str(w).strip()]
    if ex:
        pat = re.compile(r'\b(' + '|'.join(re.escape(w) for w in ex) + r')s?\b', re.I)
        gone = collections.Counter(l['name'] for l in lms if pat.search(l['name']) and l['id'] not in inc)
        lms = [l for l in lms if not pat.search(l['name']) or l['id'] in inc]
        log(f'[보정] 묶음 제외 ({",".join(ex)}): {sum(gone.values())} 개 / 종류 {len(gone)}')

    # 3) 종류 이름 그대로 빼기
    nm = {str(w).strip().lower() for w in (c.get('exclude_names') or []) if str(w).strip()}
    if nm:
        gone = collections.Counter(l['name'] for l in lms if l['name'].lower() in nm and l['id'] not in inc)
        lms = [l for l in lms if l['name'].lower() not in nm or l['id'] in inc]
        log(f'[보정] 이름 제외: {sum(gone.values())} 개 / 종류 {len(gone)} -> '
            + ', '.join(f'{k} {v}' for k, v in gone.most_common(8)))
        unknown = nm - {k.lower() for k in gone}
        if unknown:
            log(f'[보정]   경고 — 없는 이름 {len(unknown)}: ' + ', '.join(sorted(unknown)))

    # 4) 번호로 빼기
    ids = {int(x) for x in (c.get('exclude_ids') or [])}
    if ids:
        gone = collections.Counter(l['name'] for l in lms if l['id'] in ids)
        lms = [l for l in lms if l['id'] not in ids]
        log(f'[보정] 번호 제외: {sum(gone.values())} 개 / 종류 {len(gone)} -> '
            + ', '.join(f'{k} {v}' for k, v in gone.most_common(8)))
        missing = ids - all_ids
        if missing:
            log(f'[보정]   경고 — 범위 밖 번호 {len(missing)}: ' + ', '.join(map(str, sorted(missing)[:10])))

    # 5) 다른 이름 · 공통 class 를 각 랜드마크에 붙인다 (거르지는 않는다)
    ai = {int(k): [str(x).strip().lower() for x in v] for k, v in (c.get('alias_ids') or {}).items()}
    al = {str(k).strip().lower(): [str(x).strip().lower() for x in v]
          for k, v in (c.get('aliases') or {}).items()}
    cm = {str(k).strip().lower(): str(v).strip().lower() for k, v in (c.get('common') or {}).items()}
    ci = {int(k): str(v).strip().lower() for k, v in (c.get('common_ids') or {}).items()}     # 낱개 공통 class
    ov = {int(k): v for k, v in (c.get('override_ids') or {}).items()}                        # 낱개 중심·반지름
    n_a = n_c = 0
    for l in lms:
        extra = ai.get(l['id']) or al.get(l['name'].lower()) or []
        l['names'] = [l['name']] + [x for x in extra if x and x != l['name']]
        if len(l['names']) > 1:
            n_a += 1
        if l['id'] in ov:
            l['override'] = ov[l['id']]          # 70 이 .house 중심(center)·반지름(radii) 대신 쓴다
            if ov[l['id']].get('points'):        # 화면에서 다시 맞춘 모양의 3D 점 -> 상자를 이 점으로 다시 계산
                l['points'] = ov[l['id']]['points']
        common = ci.get(l['id']) or cm.get(l['name'].lower())
        if common:
            l['common'] = common
            n_c += 1
    if n_a or n_c:
        log(f'[보정] 다른 이름 붙은 랜드마크 {n_a} · 공통 class 붙은 랜드마크 {n_c} (종류 {len(cm)})')

    log(f'[보정] 랜드마크 {n_all} -> {len(lms)}')
    return lms
