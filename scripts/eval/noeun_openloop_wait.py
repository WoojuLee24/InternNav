"""노은역 열린 루프(open-loop) S2 평가 — 기록된 프레임을 한 장씩 넣고 S2 첫 출력이 WAIT / STOP / pixel goal / 회전 중 무엇인지 잰다.

닫힌 루프(Isaac, 다른 노드)에서는 WAIT 의 옳고 그름을 직접 잴 수 없어 충돌률로 본다. 여기서는 "대기로 기록된 프레임에서
WAIT 를 내는가" (recall) 와 "대기가 아닌 프레임에서 WAIT 를 내는가" (오발) 를 학습이 제대로 됐는지 빠르게 확인하는 용도로 잰다.

입력은 habitat 평가기(`habitat_vln_evaluator._run_eval_dual_system`)와 같게 만든다:
  프롬프트 = 같은 conversation 문구 + '<instruction>.' 치환 + 지난 프레임 linspace(0, t-1, num_history) + 현재 프레임,
  rgb 는 resize_w x resize_h, greedy generate (max_new_tokens 128). 연결 문구(conjunction)는 첫 번째로 고정 (재현성).
정답 종류(프레임 t, 학습 로더와 같은 시프트 actions = action[1:] + [0]):
  wait = actions[t] == 4 · stop = 마지막 프레임 · pixel = relative_goal_frame_id >= 0 · turn = 나머지
출력 종류: 'WAIT' 포함 -> wait · 'STOP' -> stop · '↓' 또는 숫자 -> pixel · ←→↑ -> turn · 그 외 other

사용 (저장소 루트, GPU 1 장):
    /usr/bin/python scripts/eval/noeun_openloop_wait.py --model_path <ckpt> --data_root data/InternData-N1-v0.5-mini/vln_ce_noeun-hunav-b1-val --out <dir>
"""

import argparse
import collections
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from internnav.model.utils.vln_utils import split_and_clean  # noqa: E402

PROMPT = ("You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? "
          "Please output the next waypoint\'s coordinates in the image. Please output STOP when you have successfully completed the task.")
CONJ = 'you can see '   # habitat 평가기 conjunctions[0]
WAIT_ACTION = 4


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--data_root', required=True, help='64_build_train.py 의 -val 데이터셋 (traj_data/r2r/<ds>_s<k>)')
    ap.add_argument('--scenes', default='hunav_s0', help='쉼표. 기본 = hunav 형제 0 문장')
    ap.add_argument('--rig', default='60cm_15deg')
    ap.add_argument('--every', type=int, default=4, help='대기가 아닌 프레임 표본 간격 (대기 프레임은 전부)')
    ap.add_argument('--num_history', type=int, default=8)
    ap.add_argument('--resize', type=int, default=384)
    ap.add_argument('--max_eps', type=int, default=0)
    ap.add_argument('--out', required=True)
    return ap.parse_args()


def kind_out(text: str) -> str:
    if 'WAIT' in text:
        return 'wait'
    if 'STOP' in text:
        return 'stop'
    if '↓' in text or re.search(r'\d', text):
        return 'pixel'
    if re.search('[←→↑]', text):
        return 'turn'
    return 'other'


def main():
    a = parse()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(a.model_path)
    processor.tokenizer.padding_side = 'left'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model_path, torch_dtype=torch.bfloat16,
                                                               attn_implementation='flash_attention_2', device_map={'': 'cuda:0'}).eval()
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
            img = lambda f: Image.open(root / 'videos' / 'chunk-000' / f'observation.images.rgb.{a.rig}' / f'episode_{i:06d}_{f}.jpg').convert('RGB').resize((a.resize, a.resize))
            frames = sorted({f for f in range(n) if acts[f] == WAIT_ACTION} | set(range(0, n - 1, a.every)) | {n - 1})
            for f in frames:
                gt = 'wait' if acts[f] == WAIT_ACTION else ('stop' if f == n - 1 else ('pixel' if rel[f] >= 0 else 'turn'))
                text = PROMPT.replace('<instruction>.', instr[:-1])
                hist = [] if f == 0 else sorted(np.unique(np.linspace(0, f - 1, a.num_history, dtype=np.int32)).tolist())
                if hist:
                    text += f" These are your historical observations: {('<image>' + chr(10)) * len(hist)}."
                text += f' {CONJ}<image>.'
                images = [img(h) for h in hist] + [img(f)]
                content, k = [], 0
                for p in split_and_clean(text):
                    if p == '<image>':
                        content.append({'type': 'image', 'image': images[k]})
                        k += 1
                    else:
                        content.append({'type': 'text', 'text': p})
                msg = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[msg], images=images, return_tensors='pt').to(model.device)
                with torch.no_grad():
                    ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True)
                o = processor.tokenizer.decode(ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                rows.append({'scene': sc, 'episode': i, 'frame': f, 'gt': gt, 'out': o, 'pred': kind_out(o)})
            print(f'  {sc} ep {i}: 프레임 {len(frames)} (대기 {sum(acts[f] == WAIT_ACTION for f in frames)}) {time.time() - t0:.0f}s', flush=True)
    conf = collections.Counter((r['gt'], r['pred']) for r in rows)
    n_gt_w = sum(r['gt'] == 'wait' for r in rows)
    n_pr_w = sum(r['pred'] == 'wait' for r in rows)
    tp = conf[('wait', 'wait')]
    summary = {'model_path': a.model_path, 'data_root': a.data_root, 'scenes': a.scenes, 'frames': len(rows),
               'wait_frames': n_gt_w, 'wait_recall': tp / n_gt_w if n_gt_w else None, 'wait_precision': tp / n_pr_w if n_pr_w else None,
               'false_wait_rate': (n_pr_w - tp) / max(1, len(rows) - n_gt_w),
               'confusion': {f'{g}->{p}': c for (g, p), c in sorted(conf.items())}, 'seconds': round(time.time() - t0)}
    (out / 'rows.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
