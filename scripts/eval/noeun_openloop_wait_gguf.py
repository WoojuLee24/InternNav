"""노은역 열린 루프 S2 평가의 GGUF(llama.cpp) 판 — noeun_openloop_wait.py 와 같은 프레임·같은 입력을 llama-s2-runner 에 넣는다.

경량화 검증용: HF 판(noeun_openloop_wait.py)과 프레임 선택·정답 종류·프롬프트가 같고, 모델 호출만 runner 로 바뀐다.
--hf_rows 를 주면 같은 (scene, episode, frame) 끼리 HF 출력과 비교해 compare.json 을 쓴다.

HF 와 맞춘 것 (internvla_n1_policy_llamacpp_hfparity.py 와 같은 설정):
  이미지 = resize x resize 로 줄인 뒤 HF processor 의 smart_resize (384 -> 392, BICUBIC), MTMD_QWEN25VL_HF_ALIGN=1
  채팅 형식 = S2_HF_CHAT=1 (Qwen2-VL 기본 system 메시지), 프롬프트 = split_and_clean 으로 이어 붙임
  생성 = greedy, repetition_penalty 1.05 를 문맥 전체에 (repeat_last_n = n_ctx), max_new_tokens 128
  프레임마다 RESET (HF 판처럼 매번 새 대화)

사용 (vln-deploy/ 에서, isaacsim 이미지):
    quantization_kit/convenv_x86/bin/python ../scripts/eval/noeun_openloop_wait_gguf.py --text <text-Q8_0-embf16.gguf> --mmproj <mmproj-f16.gguf> --data_root <val> --out <dir> [--hf_rows <hf>/rows.jsonl]
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from noeun_openloop_wait import CONJ, KEEP, PROMPT, WAIT_ACTION, kind_out  # noqa: E402

from internnav.model.basemodel.internvla_n1.internvla_n1_policy_llamacpp_hfparity import hf_processor_resize  # noqa: E402
from internnav.model.utils.vln_utils import split_and_clean  # noqa: E402

MEDIA = '<__media__>'
BIN = ROOT / 'vln-deploy' / 'quantization_kit' / 'llama.cpp' / 'build_x86' / 'bin'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--text', required=True, help='text GGUF (build_qwen_s2_gguf.sh 의 text-Q8_0-embf16.gguf)')
    ap.add_argument('--mmproj', required=True)
    ap.add_argument('--bin', default=str(BIN), help='llama-s2-runner 가 있는 폴더')
    ap.add_argument('--n_ctx', type=int, default=8192)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--scenes', default='hunav_s0')
    ap.add_argument('--rig', default='60cm_15deg')
    ap.add_argument('--every', type=int, default=4)
    ap.add_argument('--num_history', type=int, default=8)
    ap.add_argument('--resize', type=int, default=384)
    ap.add_argument('--max_eps', type=int, default=0)
    ap.add_argument('--wait_keep', default=str(KEEP))
    ap.add_argument('--hf_rows', default='', help='HF 판 rows.jsonl -> compare.json')
    ap.add_argument('--out', required=True)
    return ap.parse_args()


class Runner:
    def __init__(self, a):
        env = dict(os.environ, LD_LIBRARY_PATH=a.bin, MTMD_QWEN25VL_HF_ALIGN='1', S2_HF_CHAT='1')
        args = [str(Path(a.bin) / 'llama-s2-runner'), '-m', a.text, '--mmproj', a.mmproj, '-ngl', '99', '-c', str(a.n_ctx),
                '-n', '128', '--fit', 'off', '--temp', '0', '--repeat-penalty', '1.05', '--repeat-last-n', str(a.n_ctx)]
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True, bufsize=1)
        self.tmp = tempfile.mkdtemp(prefix='noeun_gguf_')

    def cmd(self, line):
        self.p.stdin.write(line + '\n')
        self.p.stdin.flush()
        r = self.p.stdout.readline().rstrip('\n')
        assert r.startswith('OK'), f's2-runner: {r!r}'
        return r[3:]

    def text(self, prompt, images):
        self.cmd('RESET')
        paths = []
        for k, im in enumerate(images):
            paths.append(os.path.join(self.tmp, f'i{k}.bmp'))
            im.save(paths[-1])
        esc = prompt.replace('\\', '\\\\').replace('\n', '\\n').replace('\t', ' ')
        return self.cmd(f"TEXT\t{','.join(paths)}\t{esc}").strip()


def pixel(text):
    m = re.findall(r'\d+', text)
    return np.array([int(m[-2]), int(m[-1])], float) if len(m) >= 2 else None


def compare(rows, hf_rows):
    """같은 (scene, episode, frame) 끼리 GGUF vs HF: 문자열 정확 일치 · 출력 종류 일치 · pixel goal 거리."""
    hf = {(r['scene'], r['episode'], r['frame']): r for r in hf_rows}
    keys = [(r['scene'], r['episode'], r['frame']) for r in rows if (r['scene'], r['episode'], r['frame']) in hf]
    g = {(r['scene'], r['episode'], r['frame']): r for r in rows}
    exact = sum(g[k]['out'] == hf[k]['out'] for k in keys)
    kind = sum(g[k]['pred'] == hf[k]['pred'] for k in keys)
    d = [np.linalg.norm(pixel(g[k]['out']) - pixel(hf[k]['out'])) for k in keys
         if g[k]['pred'] == hf[k]['pred'] == 'pixel' and pixel(g[k]['out']) is not None and pixel(hf[k]['out']) is not None]
    d = np.array(d) if d else np.zeros(0)
    kind_conf = collections.Counter(f"{hf[k]['pred']}->{g[k]['pred']}" for k in keys if g[k]['pred'] != hf[k]['pred'])
    return {'frames': len(keys), 'missing_in_hf': len(rows) - len(keys),
            'exact': exact, 'exact_rate': exact / len(keys) if keys else None,
            'kind_match': kind, 'kind_rate': kind / len(keys) if keys else None, 'kind_mismatch_hf->gguf': dict(kind_conf),
            'pixel_frames': int(d.size), 'pixel_mean': float(d.mean()) if d.size else None,
            'pixel_p95': float(np.percentile(d, 95)) if d.size else None, 'pixel_max': float(d.max()) if d.size else None,
            'pixel_gt5': int((d > 5).sum())}


def main():
    a = parse()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = {tuple(k) for k in json.load(open(a.wait_keep))['val_seen']} if a.wait_keep else None
    runner = Runner(a)
    rows, t0 = [], time.time()
    for sc in a.scenes.split(','):
        root = Path(a.data_root) / 'traj_data' / 'r2r' / sc
        eps = [json.loads(l) for l in open(root / 'meta' / 'episodes.jsonl')]
        for e in eps[:a.max_eps] if a.max_eps else eps:
            i, instr = e['episode_index'], e['tasks'][0].strip() + ' '
            t = pq.read_table(root / 'data' / 'chunk-000' / f'episode_{i:06d}.parquet',
                              columns=['action', f'relative_goal_frame_id.{a.rig}']).to_pydict()
            acts = [int(x) for x in t['action']][1:] + [0]
            rel = [int(x) for x in t[f'relative_goal_frame_id.{a.rig}']]
            n = len(acts)
            img = lambda f: hf_processor_resize(Image.open(root / 'videos' / 'chunk-000' / f'observation.images.rgb.{a.rig}' / f'episode_{i:06d}_{f}.jpg').convert('RGB').resize((a.resize, a.resize)))
            frames = sorted({f for f in range(n) if acts[f] == WAIT_ACTION} | set(range(0, n - 1, a.every)) | {n - 1})
            base = re.sub(r'_s\d+$', '', sc)
            for f in frames:
                if acts[f] == WAIT_ACTION:
                    gt = 'wait' if keep is None or (base, i, f) in keep else 'wait_skip'
                else:
                    gt = 'stop' if f == n - 1 else ('pixel' if rel[f] >= 0 else 'turn')
                text = PROMPT.replace('<instruction>.', instr[:-1])
                hist = [] if f == 0 else sorted(np.unique(np.linspace(0, f - 1, a.num_history, dtype=np.int32)).tolist())
                if hist:
                    text += f" These are your historical observations: {('<image>' + chr(10)) * len(hist)}."
                text += f' {CONJ}<image>.'
                prompt = ''.join(MEDIA if p == '<image>' else p for p in split_and_clean(text))
                o = runner.text(prompt, [img(h) for h in hist] + [img(f)])
                rows.append({'scene': sc, 'episode': i, 'frame': f, 'gt': gt, 'out': o, 'pred': kind_out(o)})
            print(f'  {sc} ep {i}: 프레임 {len(frames)} {time.time() - t0:.0f}s', flush=True)
    skipped = [r for r in rows if r['gt'] == 'wait_skip']
    rows_all, rows = rows, [r for r in rows if r['gt'] != 'wait_skip']
    conf = collections.Counter((r['gt'], r['pred']) for r in rows)
    n_gt_w = sum(r['gt'] == 'wait' for r in rows)
    n_pr_w = sum(r['pred'] == 'wait' for r in rows)
    tp = conf[('wait', 'wait')]
    summary = {'text': a.text, 'mmproj': a.mmproj, 'data_root': a.data_root, 'scenes': a.scenes, 'frames': len(rows),
               'wait_frames': n_gt_w, 'wait_recall': tp / n_gt_w if n_gt_w else None, 'wait_precision': tp / n_pr_w if n_pr_w else None,
               'false_wait_rate': (n_pr_w - tp) / max(1, len(rows) - n_gt_w), 'wait_skip_frames': len(skipped),
               'confusion': {f'{g}->{p}': c for (g, p), c in sorted(conf.items())}, 'seconds': round(time.time() - t0)}
    (out / 'rows.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows_all) + '\n')
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if a.hf_rows:
        c = compare(rows_all, [json.loads(l) for l in open(a.hf_rows) if l.strip()])
        (out / 'compare.json').write_text(json.dumps(c, ensure_ascii=False, indent=1))
        print(json.dumps(c, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
