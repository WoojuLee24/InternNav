"""noeun 64b · 랜드마크 학습 데이터 — 62b 라벨로 S2 학습용 vln_ce 데이터셋을 만든다 (64 와 같은 방식, 64 는 그대로).

runner 의 train_label 규칙(`<mini>/vln_ce` -> `<mini>/vln_ce_<train_label>`)과 로더 `r2r_60cm_15_15`(traj_data/r2r, 60cm_15deg)를 쓴다.
  원본 : <mini>/vln_ce_noeun_src/landmark_t12/traj_data/r2r/noeunstation/   (vlnce_train_t12 를 rsync — 노드 로컬 실물)
  학습 : <mini>/vln_ce_noeun-lm<N>-<label>/traj_data/r2r/landmark_s<k>/      (k = 형제 0, 1, 2)
scene 폴더 = 형제 k: data · videos 는 원본으로 상대 링크, meta 는 복사 후 episodes.jsonl · tasks.jsonl 만 형제 k 문장으로.
N = 앞에서 경로 N 개 (episode_index < N; 51b 렌더 순서 = 경로 고른 순서). 로더는 meta/episodes.jsonl 에 있는 ep 만 읽는다.
val 데이터셋은 만들지 않는다 — noeun 셀(noeun_base.stage1)은 val_ratio 0 (데이터 100 % 학습), 64 의 -val 도 학습에 안 쓰였다.

라벨 (62b 출력 이름 · gen_hash, 2026-11-15 사용자 결정):
  b1     : b1, 구역 v2                          (rooms-noeun-v2)
  b1lm   : b1, 구역 v2 + 멈출 곳 랜드마크 줄      (+ stop_cue-noeun-paths-v1)
  post   : 후, 구역 v2
  postlm : 후, 구역 v2 + 멈출 곳 랜드마크 줄
  모두 --landmark_nouns noeun-v1 (출력 <labels_root> = .../instruction_labels/nouns-noeun-v1)

사용 (저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/64b_build_train_landmark.py --src <DIR>/vlnce_train_t12 --labels_root <DIR>/instruction_labels/nouns-noeun-v1
    (시험) ... --mini <빈 폴더> --test_eps   : 라벨이 있는 ep 만으로 (62b --ep_list 시험 라벨 확인용)

평가 A · B (--split eval | eval_seen, 2026-11-15 추가 · 기본 train 은 위와 같음):
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/64b_build_train_landmark.py --split eval --src <vlnce_eval_t12> --labels_root <...>/instruction_labels/nouns-noeun-v1
  원본 <mini>/vln_ce_noeun_src/landmark_t12_<split>/ , 데이터셋 <mini>/vln_ce_noeun-lmevalA-<label>/ (eval_seen = lmevalB) — 경로 전부 (--sizes 안 씀).
  · 열린 루프 (scripts/eval/noeun_openloop_pixel.py --scenes landmark_s0,landmark_s1,landmark_s2) : traj_data/r2r/landmark_s<k> (학습과 같은 형식)
  · 닫힌 루프 VLN-PE (scripts/eval/run_noeun_vlnpe_eval.sh, NOEUN_DATA=<데이터셋>/raw_data/r2r) : raw_data/r2r/val_seen/val_seen.json.gz
    경로당 형제 --pe_sibs (기본 0 = 첫 문장만, 경로 80 개) · 예전 v6_b1 val_seen 과 같은 칸 (+ key · kind · sib_idx)
"""

import argparse
import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path

POST = 'b1_facts-memo-v1_knobs-v2_delivery-joint3-v1_writer-gepa-v3-joint-gemma_model-gemma31b-bf16_rig-60cm_15deg_rooms-noeun-v2'
B1 = 'b1_model-gemma31b-bf16_rig-60cm_15deg_rooms-noeun-v2'
LM = '_stop_cue-noeun-paths-v1'
LABELS = {'b1': (B1, '1dd95593a6ff'), 'b1lm': (B1 + LM, '8ee0be42a6e4'),
          'post': (POST, 'b7b15a339bc3'), 'postlm': (POST + LM, '7743e94b44d6')}
DS = 'landmark_t12'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='vlnce_train_t12 (Gdrive tar 를 푼 곳)')
    ap.add_argument('--labels_root', required=True, help='62b 출력 .../instruction_labels/nouns-noeun-v1')
    ap.add_argument('--mini', default='/home/irteam/data-vol1/InternData-N1-v0.5-mini')
    ap.add_argument('--labels', default='b1,b1lm,post,postlm')
    ap.add_argument('--sizes', default='400,1200', help='앞에서 경로 N 개')
    ap.add_argument('--n_sib', type=int, default=3)
    ap.add_argument('--test_eps', action='store_true', help='라벨이 있는 ep 만 (시험용; sizes 무시)')
    ap.add_argument('--split', default='train', choices=['train', 'eval', 'eval_seen'],
                    help='train (기본) | eval (평가 A, 안 본 랜드마크) | eval_seen (평가 B, 본 랜드마크) — 평가는 경로 전부 + VLN-PE raw json')
    ap.add_argument('--pe_sibs', default='0', help='평가 VLN-PE raw json 에 넣을 형제 번호 (쉼표)')
    return ap.parse_args()


def read_jsonl(p):
    return [json.loads(x) for x in open(p)]


def write_jsonl(p, rows):
    Path(p).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def texts(a, label: str) -> dict:
    """{episode_index: [형제 0, 1, 2 문장]} — gen_hash 가 정한 값과 같아야 한다."""
    name, gh = LABELS[label]
    d = json.load(gzip.open(Path(a.labels_root) / name / f'{DS}_{a.split}' / f'{DS}_{a.split}.json.gz'))
    assert d['_generation_meta']['gen_hash'] == gh, f'{label}: gen_hash {d["_generation_meta"]["gen_hash"]} != {gh} (설정이 다르다)'
    T = {}
    for e in d['episodes']:
        T.setdefault(e['raw_episode_id'], {})[e['sib_idx']] = e['instruction']['instruction_text'].strip()
    a._raw[label] = d['episodes']
    out = {}
    for i, s in T.items():
        assert sorted(s) == list(range(a.n_sib)) and all(s.values()), f'{label} ep {i}: 형제 {sorted(s)} 또는 빈 문장'
        out[i] = [s[k] for k in range(a.n_sib)]
    return out


def copy_src(a) -> Path:
    src = Path(a.src) / 'traj_data' / 'r2r' / 'noeunstation'
    if a.test_eps:                       # 시험: 28 GB 를 복사하지 않고 원본을 그대로 가리킨다
        return src.resolve()
    dst = Path(a.mini) / 'vln_ce_noeun_src' / (DS if a.split == 'train' else f'{DS}_{a.split}') / 'traj_data' / 'r2r' / 'noeunstation'
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', f'{src}/', f'{dst}/'], check=True)
    print(f'src: {dst}')
    return dst


def build(a, src: Path, label: str, n: int, T: dict):
    if a.split == 'train':
        tl = f'noeun-lm{n}-{label}' if not a.test_eps else f'noeun-lmtest-{label}'
    elif a.split == 'eval':
        tl = f'noeun-lmevalA-{label}' if not a.test_eps else f'noeun-lmevalAtest-{label}'
    elif a.split == 'eval_seen':
        tl = f'noeun-lmevalB-{label}' if not a.test_eps else f'noeun-lmevalBtest-{label}'
    else:
        assert False, f'unreachable split={a.split!r}'
    out = Path(a.mini) / f'vln_ce_{tl}' / 'traj_data' / 'r2r'
    eps = read_jsonl(src / 'meta' / 'episodes.jsonl')
    eps = [e for e in eps if e['episode_index'] in T] if a.test_eps else [e for e in eps if e['episode_index'] < n]
    assert a.test_eps or len(eps) == n, f'{tl}: ep {len(eps)} != {n}'
    assert all(e['episode_index'] in T for e in eps), f'{tl}: 라벨 없는 ep {[e["episode_index"] for e in eps if e["episode_index"] not in T][:10]}'
    for k in range(a.n_sib):
        sc = out / f'landmark_s{k}'
        (sc / 'meta').mkdir(parents=True, exist_ok=True)
        for sub in ('data', 'videos'):
            if not (sc / sub).is_symlink():
                os.symlink(os.path.relpath(src / sub, sc), sc / sub)
        for f in (src / 'meta').iterdir():
            if f.name not in ('episodes.jsonl', 'tasks.jsonl'):
                shutil.copy2(f, sc / 'meta' / f.name)
        rows = [{'episode_index': e['episode_index'], 'tasks': [T[e['episode_index']][k]], 'length': e['length']} for e in eps]
        write_jsonl(sc / 'meta' / 'episodes.jsonl', rows)
        write_jsonl(sc / 'meta' / 'tasks.jsonl', [{'task_index': r['episode_index'], 'task': r['tasks'][0]} for r in rows])
    info = {'train_label': tl, 'label': label, 'label_name': LABELS[label][0], 'gen_hash': LABELS[label][1], 'n_paths': len(eps),
            'n_sib': a.n_sib, 'episodes': len(eps) * a.n_sib, 'frames': sum(e['length'] for e in eps) * a.n_sib,
            'source': str(src), 'loader': 'r2r_60cm_15_15'}
    if a.split != 'train':
        info['split'] = a.split
        info['vlnpe_raw'] = write_pe_raw(a, out.parent.parent, label, {e['episode_index'] for e in eps})
    (out.parent.parent / 'noeun_train_source.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(f'{out.parent.parent.name}: 경로 {len(eps)} x 형제 {a.n_sib} = ep {info["episodes"]} · 프레임 {info["frames"]}')


def write_pe_raw(a, ds_dir: Path, label: str, keep: set) -> str:
    """평가 닫힌 루프 (VLN-PE) 입력: <ds_dir>/raw_data/r2r/val_seen/val_seen.json.gz — 62b 라벨 json 의 ep 중 형제 --pe_sibs 만.
    칸은 예전 노은역 평가 파일 (vln_pe_noeun_nav3d_rbq_v6_b1 val_seen) 과 같고 key · kind · sib_idx 를 더한다."""
    sibs = {int(x) for x in a.pe_sibs.split(',')}
    meta = Path(a.src) / 'noeun_meta'
    RM = json.loads((meta / 'render_meta.json').read_text())['episodes']
    P = {e['episode_id']: e for e in json.loads((meta / 'paths_meshfloor_straight_v5.json').read_text())['episodes']}
    rows = []
    for e in a._raw[label]:
        i = e['raw_episode_id']
        if i not in keep or e['sib_idx'] not in sibs:
            continue
        key = RM[i]['key']
        rows.append({k: e[k] for k in ('episode_id', 'trajectory_id', 'scene_id', 'start_position', 'start_rotation', 'info', 'goals',
                                       'reference_path', 'instruction')} | {'key': key, 'kind': P[int(key.split(':')[1])]['kind'], 'sib_idx': e['sib_idx']})
    assert len(rows) == len(keep) * len(sibs), f'{label}: VLN-PE ep {len(rows)} != {len(keep)} x {len(sibs)}'
    p = ds_dir / 'raw_data' / 'r2r' / 'val_seen' / 'val_seen.json.gz'
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, 'wt') as f:
        json.dump({'episodes': rows, '_label': {'label': label, 'name': LABELS[label][0], 'gen_hash': LABELS[label][1],
                                                'split': a.split, 'sibs': sorted(sibs)}}, f)
    print(f'  VLN-PE raw: {p} ({len(rows)} ep)')
    return str(p)


def main():
    a = parse()
    a._raw = {}
    src = copy_src(a)
    for label in a.labels.split(','):
        T = texts(a, label)
        if a.test_eps:
            sizes = [0]
        elif a.split == 'train':
            sizes = [int(x) for x in a.sizes.split(',')]
        else:
            sizes = [len(T)]                    # 평가: 경로 전부
        for n in sizes:
            build(a, src, label, n, T)


if __name__ == '__main__':
    main()
