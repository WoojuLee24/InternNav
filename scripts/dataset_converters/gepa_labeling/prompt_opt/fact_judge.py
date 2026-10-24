"""모델 사실 검사 (score.fact_checker 가 rule 이 아닐 때 scorer.py 가 쓴다; claude_judge.py 시험도 같은 코드).

경로 사실(출발 회전, 회전과 그 자리 방, 계단, 들어가는 방, 끝 방을 순서대로) 과 문장을 모델에 주고 사실과 어긋나는 말만 찾게 한다.
  1단계 SYSTEM       : 오류 찾기 (batch 문장씩)
  2단계 VERIFY_SYSTEM: (verify=true) 지적한 오류마다 진짜 모순인지 같은 모델에 다시 물어 "keep" 만 남김
모델 설정 jc (configs/gepa.yaml fact_checkers.<이름>): backend (claude = `claude -p` 새 대화·CLAUDE.md·도구 없음 / openai = vLLM 등),
  base_url, model, effort, facts (v1 | v2), verify, batch, workers, cwd, retry_missing (답에서 빠진 문장만 다시 묻는 횟수, 기본 2)
"""
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'scripts/dataset_converters/instruction_labeling/label'))
from rooms import load_regions, room_at  # noqa: E402,F401  (load_regions 는 claude_judge 도 쓴다)
from route_words import route_geometry  # noqa: E402
from sources import house_path  # noqa: E402


SYSTEM = """You check navigation instructions written for a path inside a house against the TRUE FACTS of that path.
Find only statements that CONTRADICT the facts. Leaving things out is fine: people skip turns, rooms and details.

What you can check (from the facts): the start turn, the direction and order of turns, going up or down stairs,
the rooms the path goes through, and the room where the path ends.
What you cannot check: objects and landmarks, and left/right that describes where something is
("the sofa on your left", "the door to the right") rather than a turn. Ignore those.

People speak loosely, so do not report:
- a turn of about 30-60 degrees described as a turn, a slight turn, or just "go/walk/head" in that direction;
- "turn around" for a turn of about 100 degrees or more; several small turns in a row described as one turn;
- a start turn left implicit ("walk toward X") or folded into the first move;
- similar room names (living room / family room / lounge / den / tv room; hallway / corridor / landing / hall;
  entryway / foyer / entrance; bathroom / toilet / restroom; dining room / dining area; office / study / library);
- stopping at the doorway or entrance of the final room, or "stop in the hallway" when the path ends right at a hallway edge
  is not certain from the facts (only report a room contradiction when it is clear).

Report a contradiction when, for example: a turn goes the opposite way from every turn that could match it at that point
in the route; turns are said in an order the route cannot have; the instruction goes up stairs when the path only goes down
or has no stairs (or the reverse); the instruction says the path ends / you stop / you wait in a room that is not the final
room (including a room the path passed earlier); it says to enter a room the path never visits.

Answer with ONLY JSON, one entry per item, in the given order:
{"results": [{"id": "<id>", "errors": [{"type": "turn|order|stairs|room|other", "phrase": "<words from the instruction>", "why": "<short>"}]}]}
Use "errors": [] when nothing contradicts the facts."""



UNKNOWN = {  # 방 정보(MP3D region) 가 없는 점을 경로 사실에 어떻게 쓰나
    'v1': {'start': 'Start in: unknown room.', 'turn': ' (in the unknown room)', 'end': 'End (stop) in: unknown room.'},
    'v2': {'start': 'Start in: (room not annotated; do not judge the start room).', 'turn': '',
           'end': 'End (stop) in: (room not annotated; do not judge the final room).'},
}


def route_events(ep, rc, facts='v1'):
    """경로 사실을 순서대로: 출발 회전, 회전 (그 자리의 방), 걷기 (계단·들어가는 방), 끝 방.
    facts=v2: 방 정보가 없는 곳은 "unknown room" 대신 판단하지 말라고 쓴다 (모델이 unknown 을 다른 방으로 보고 경보함)."""
    U = UNKNOWN[facts]
    g = route_geometry(ep)
    rr = rc['rooms']
    regions = load_regions(str(house_path(ep)))
    p = ep['reference_path']

    def room(i):
        r = room_at(regions, p[i], rr['max_dz_m'])
        return rr['names'].get(r['label']) if r else None

    def side(a):
        return f"{'right' if a > 0 else 'left'} {abs(a):.0f} degrees"

    lines = [f'Start in: {room(0)}.' if room(0) else U['start']]
    a0 = g['initial']
    lines.append(f'Start turn: the path begins {side(a0)} from the way you face.' if abs(a0) >= rc['turn_deg']
                 else 'Start turn: none (the path begins roughly straight ahead).')
    cur, walk = room(0), 0.0
    for i, sg in enumerate(g['segments']):
        if i in g['turns'] and abs(g['turns'][i]) >= rc['turn_deg']:
            if walk:
                lines.append(f'Walk {walk:.1f} m.')
                walk = 0.0
            lines.append(f'Turn {side(g["turns"][i])}' + (f' (in the {room(i)})' if room(i) else U['turn']) + '.')
        nxt = room(i + 1)
        if abs(sg['dy']) > rc['stairs_dy_m']:
            if walk:
                lines.append(f'Walk {walk:.1f} m.')
                walk = 0.0
            lines.append(f'Go {"up" if sg["dy"] > 0 else "down"} stairs ({abs(sg["dy"]):.1f} m).')
        else:
            walk += sg['dist']
        if nxt and nxt != cur:
            if walk:
                lines.append(f'Walk {walk:.1f} m.')
                walk = 0.0
            lines.append(f'Enter the {nxt}.')
            cur = nxt
    if walk:
        lines.append(f'Walk {walk:.1f} m.')
    lines.append(f'End (stop) in: {room(len(p) - 1)}.' if room(len(p) - 1) else U['end'])
    return '\n'.join(lines)



VERIFY_SYSTEM = """A first checker compared a navigation instruction for a path inside a house with the TRUE FACTS of that path
and claimed some errors. For each claimed error decide whether it is a real CONTRADICTION: the statement cannot be true given the facts.

Answer "drop" when the claim is really one of these (they are NOT errors):
- an omission: the instruction skips a room, a turn, stairs or other details (people leave things out);
- a similar, broader or more specific name for the same place (pantry or kitchen island area in a kitchen, living area, den,
  foyer, landing, ...). Room labels come from a coarse annotation, so an open area may carry a different label than a person would use;
- a place the facts do not annotate (no room given), or anything the facts cannot settle;
- loose turn wording: about 30-60 degrees called a turn or slight turn, about 100 degrees or more called "turn around",
  several small turns said as one, a start turn left implicit;
- left/right that tells where an object is, or claims about objects and landmarks.
Answer "keep" only when the facts clearly rule the statement out (opposite turn direction, impossible order, stairs that the
path does not have, stopping in or entering a clearly different room).

Answer with ONLY JSON: {"results": [{"id": "<id>", "verdicts": ["keep" or "drop", one per claimed error in the given order]}]}"""


def batch_msg(batch, ev):
    return 'Check each item. Items:\n\n' + '\n\n'.join(
        f'### id: {x["id"]}\nTRUE FACTS:\n{ev[x["ep"]]}\nINSTRUCTION: "{x["text"].strip()}"' for x in batch)


def verify_msg(batch, ev):
    return 'Judge each claimed error. Items:\n\n' + '\n\n'.join(
        f'### id: {x["id"]}\nTRUE FACTS:\n{ev[x["ep"]]}\nINSTRUCTION: "{x["text"].strip()}"\nCLAIMED ERRORS:\n'
        + '\n'.join(f'{j + 1}. [{e.get("type")}] "{e.get("phrase", "")}": {e.get("why", "")}' for j, e in enumerate(x['errors_raw']))
        for x in batch)


def parse(text):
    m = re.search(r'\{.*\}', text, re.S)
    return {d['id']: d.get('errors', []) for d in json.loads(m.group(0))['results']}


def parse_verify(text):
    m = re.search(r'\{.*\}', text, re.S)
    return {d['id']: d['verdicts'] for d in json.loads(m.group(0))['results']}


def ask(system, msg, parse_fn, n, jc):
    """한 번 호출. backend=claude: `claude -p` (새 대화, CLAUDE.md·도구 없음) / openai: OpenAI 호환 서버 (vLLM gemma 등)."""
    err = ''
    for attempt in range(3):
        t0 = time.time()
        try:
            if jc['backend'] == 'claude':
                cmd = ['claude', '-p', '--model', jc['model'], '--effort', jc['effort'], '--safe-mode', '--tools', '', '--system-prompt', system,
                       '--output-format', 'json', '--no-session-persistence']
                r = subprocess.run(cmd, input=msg, capture_output=True, text=True, cwd=jc['cwd'], timeout=900)
                out = json.loads(r.stdout)
                text, usage, cost = out['result'], out['usage'], out.get('total_cost_usd')
            elif jc['backend'] == 'openai':
                from openai import OpenAI
                client = OpenAI(base_url=jc['base_url'], api_key='token-abc123')
                r = client.chat.completions.create(model=jc['model'], temperature=0.0, max_tokens=4000, seed=attempt,
                                                   messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': msg}])
                text, cost = r.choices[0].message.content or '', 0.0
                usage = {'input_tokens': r.usage.prompt_tokens, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0,
                         'output_tokens': r.usage.completion_tokens}
            else:
                assert False, f'unreachable backend={jc['backend']!r}'
            return {'ok': True, 'res': parse_fn(text), 'usage': usage, 'cost': cost, 'sec': time.time() - t0, 'n': n, 'attempt': attempt}
        except Exception as e:  # noqa: BLE001 — 형식이 깨지면 다시 묻는다
            err = f'{type(e).__name__}: {e}'
    return {'ok': False, 'res': {}, 'err': err, 'n': n}


def run_stage(items, msg_fn, system, parse_fn, ev, jc):
    batches = [items[i:i + jc['batch']] for i in range(0, len(items), jc['batch'])]
    with ThreadPoolExecutor(jc['workers']) as ex:
        calls = list(ex.map(lambda b: ask(system, msg_fn(b, ev), parse_fn, len(b), jc), batches))
    return batches, calls


def judge(items, ev, jc):
    """items = [{'id', 'ep' (ev 의 키), 'text'}] 에 'errors_raw' (1단계), 'errors' (최종), 'verdicts' 를 채운다. -> 호출 기록 list.
    답을 못 받은 문장은 'errors' 가 없다 (부르는 쪽에서 감점하지 않음). 2단계 답을 못 받으면 1단계 지적을 그대로 둔다."""
    calls, todo = [], items
    for r in range(1 + jc.get('retry_missing', 2)):  # 모델이 답에서 빠뜨린 문장(id 누락·형식 깨짐) 만 다시 묻는다 (다시 물을 땐 한 문장씩)
        batches, cs = run_stage(todo, batch_msg, SYSTEM, parse, ev, jc if r == 0 else dict(jc, batch=1))
        calls += cs
        for b, c in zip(batches, cs):
            for x in b:
                if x['id'] in c['res']:
                    x['errors'] = x['errors_raw'] = c['res'][x['id']]
        todo = [x for x in items if 'errors' not in x]
        if not todo:
            break
    if jc['verify']:
        flagged = [x for x in items if x.get('errors_raw')]
        vb, vc = run_stage(flagged, verify_msg, VERIFY_SYSTEM, parse_verify, ev, jc)
        for b, c in zip(vb, vc):
            for x in b:
                v = c['res'].get(x['id'])
                if v is not None and len(v) == len(x['errors_raw']):
                    x['errors'] = [e for e, k in zip(x['errors_raw'], v) if str(k).lower() == 'keep']
                    x['verdicts'] = v
        calls = calls + vc
    return calls
