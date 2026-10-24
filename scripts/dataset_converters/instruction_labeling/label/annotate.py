"""P2: 방법(method) + 그룹별 인자로 split 의 지시문을 생성한다 (VLM 은 vLLM OpenAI 호환 서버).

    /usr/bin/python scripts/dataset_converters/instruction_labeling/label/annotate.py --method b0 --split val_unseen --subset dev
    /usr/bin/python scripts/dataset_converters/instruction_labeling/label/annotate.py --method b0 --set route.start_turn=gtrate-v1 --split val_unseen --subset dev

방법 = configs/methods/<method>.yaml (그룹: frames / route / delivery / style / model / gen). `--set` 으로 인자를 바꾸면
결과 이름에 suffix 가 붙는다 (b0_start_turn-gtrate-v1). 해석 규칙은 label/config.py.

출력 (`--out_root`, 기본 data/InternData-N1-v0.5-mini/instruction_labels)
  <name>/<split>[_<subset>]/meta/ep_<episode_id>.json   episode 별 생성 기록 (입력 프레임, route words, 시도, 품질)
  <name>/<split>[_<subset>]/<split>_<name>.json.gz      GT 키 + 새 instruction (". " 끝), instruction_vocab, _generation_meta
  전체 split(subset 없음)이면 `relabel_vlnce/build_relabel_dataset.py --label_dir <out_root>/<name>` 이 읽는 위치와 같다.

재실행: 같은 gen_hash 의 meta 가 있으면 건너뛴다 (중단 후 재개). gen_hash 가 다른 meta 가 있으면 거부한다.
조용한 실패 방지: 예외/빈 응답 비율이 `fail_abort_rate` 를 넘으면 중단, 품질 실패·예외 수는 요약에 남긴다.
"""

import argparse
import gzip
import json
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import backend  # noqa: E402
import config  # noqa: E402
import e2e  # noqa: E402
import two_stage  # noqa: E402
import joint  # noqa: E402
from route_words import CODE_VERSION  # noqa: E402
from sources import REPO, episode_track, items  # noqa: E402

GENERATORS = {'image': e2e.generate, 'two_stage': two_stage.generate, 'joint': joint.generate}  # delivery 정의의 type 으로 선택
DEV_SUBSET = HERE.parent / 'eval' / 'dev_subset.json'
VOCAB_SOURCE = REPO / 'data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r/val_unseen/val_unseen.json.gz'


def git_sha():
    try:
        return subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def normalize(text: str) -> str:
    """= relabel_vlnce/build_relabel_dataset.normalize: GT 처럼 '. ' 로 끝나게 (evaluator 가 [:-1] 로 자른다)."""
    t = text.strip()
    if t and t[-1] not in '.!?':
        t += '.'
    return t + ' '


def select_subset(its, subset):
    if not subset:
        return its
    if subset == 'dev':
        ids = set(json.loads(DEV_SUBSET.read_text())['trajectory_ids'])
        return [it for it in its if it['ep']['trajectory_id'] in ids]
    assert False, f'unreachable subset={subset!r}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', required=True, help='configs/methods/<method>.yaml (b0, ...)')
    ap.add_argument('--set', action='append', default=[], help='<group>.<arg>=<value> 또는 <group>=<value> (반복 가능)')
    ap.add_argument('--split', required=True, help='train | val_unseen')
    ap.add_argument('--subset', default='', help="'' (전체) 또는 'dev' (eval/dev_subset.json)")
    ap.add_argument('--limit', type=int, default=0, help='앞에서 N episode 만 (스모크용)')
    ap.add_argument('--out_root', default='data/InternData-N1-v0.5-mini/instruction_labels')
    ap.add_argument('--backend', default='server', choices=['server', 'managed'],
                    help='server: 떠 있는 vLLM 서버 사용 / managed: 이 job 이 서버를 띄우고 끝나면 내림 (GPU 반납)')
    ap.add_argument('--gpus', default=None, help='managed 일 때 CUDA_VISIBLE_DEVICES (기본: model yaml)')
    args = ap.parse_args()

    cfg = config.load(args.method, args.set)
    assert cfg['delivery']['type'] in GENERATORS, f'unknown delivery type={cfg["delivery"]["type"]!r}'
    ghash = config.gen_hash(cfg, CODE_VERSION)
    name = cfg['name']
    its = select_subset(items(args.split), args.subset)
    if args.limit:
        its = its[:args.limit]
    tag = args.split + (f'_{args.subset}' if args.subset else '') + (f'_limit{args.limit}' if args.limit else '')
    out_dir = REPO / args.out_root / name / tag
    meta_dir = out_dir / 'meta'
    meta_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for it in its:
        mp = meta_dir / f'ep_{it["ep"]["episode_id"]:06d}.json'
        if mp.is_file():
            old = json.loads(mp.read_text())
            if old['gen_hash'] != ghash:
                raise SystemExit(f'{mp} 의 gen_hash {old["gen_hash"]} != {ghash}: 설정이 바뀌었다. 다른 label 이름을 쓴다.')
            continue
        todo.append(it)
    print(f'[{name}/{tag}] gen_hash={ghash} episodes={len(its)} todo={len(todo)}', flush=True)

    rig_list = cfg['frames'].get('rigs') or [cfg['frames']['rig']]
    t0 = time.time()

    def run(client):
        def work(it):
            track = episode_track(it, rig_list[0])
            r = GENERATORS[cfg['delivery']['type']](client, cfg, it['ep'], track)
            m = {'episode_id': it['ep']['episode_id'], 'trajectory_id': it['ep']['trajectory_id'], 'scene': it['scene'],
                 'episode_index': it['episode_index'], 'gen_hash': ghash, 'name': name, 'generated': r}
            (meta_dir / f'ep_{it["ep"]["episode_id"]:06d}.json').write_text(json.dumps(m, ensure_ascii=False))
            return r

        n_err, n_done = 0, 0
        with ThreadPoolExecutor(int(cfg['workers'])) as ex:
            futs = [ex.submit(work, it) for it in todo]
            for fu in as_completed(futs):
                r = fu.result()
                n_done += 1
                n_err += r['error'] is not None
                if n_done >= 50 and n_err / n_done > float(cfg['fail_abort_rate']):
                    for f in futs:
                        f.cancel()
                    raise SystemExit(f'예외/빈 응답 {n_err}/{n_done} > {cfg["fail_abort_rate"]}: 중단. 마지막 오류: {r["error"]}')
                if n_done % 100 == 0 or n_done == len(todo):
                    print(f'  {n_done}/{len(todo)} err={n_err} {time.time() - t0:.0f}s', flush=True)

    if todo:
        with backend.client(cfg['vllm'], args.backend, out_dir, args.gpus) as client:
            run(client)

    # assemble
    vocab = json.load(gzip.open(VOCAB_SOURCE))['instruction_vocab']
    eps, issues, n_bad, n_err, n_parse_fail = [], Counter(), 0, 0, 0
    for it in its:
        m = json.loads((meta_dir / f'ep_{it["ep"]["episode_id"]:06d}.json').read_text())
        g = m['generated']
        n_bad += not g['quality_ok']
        n_err += g['error'] is not None
        issues.update(g['quality_issues'])
        n_parse_fail += g['inputs'].get('scene_parse_fail', 0)
        ep = dict(it['ep'])
        ep['instruction'] = {'instruction_text': normalize(g['text']), 'instruction_tokens': []}
        eps.append(ep)
    out = {'episodes': eps, 'instruction_vocab': vocab,
           '_generation_meta': {'name': name, 'method': args.method, 'sets': args.set, 'gen_hash': ghash, 'config': cfg, 'git': git_sha(),
                                'split': args.split, 'subset': args.subset, 'limit': args.limit,
                                'created_utc': time.strftime('%Y-%m-%d %H:%M:%S'), 'n': len(eps),
                                'quality_fail': n_bad, 'errors': n_err, 'issues': dict(issues),
                                'scene_parse_fail_frames': n_parse_fail}}
    dst = out_dir / f'{args.split}_{name}.json.gz'
    with gzip.open(dst, 'wt') as f:
        json.dump(out, f)
    if not args.subset and not args.limit:  # build_relabel_dataset 위치 (<label_dir>/<split>/<split>_<label>.json.gz)
        alt = REPO / args.out_root / name / args.split / f'{args.split}_{name}.json.gz'
        if alt != dst:
            alt.parent.mkdir(parents=True, exist_ok=True)
            alt.write_bytes(dst.read_bytes())
    print(json.dumps(out['_generation_meta'] | {'config': '...'}, ensure_ascii=False))
    print(f'-> {dst}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
