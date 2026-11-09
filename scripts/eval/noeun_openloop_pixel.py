"""노은역 열린 루프 S2 — '↓' 다음 둘째 답(픽셀 좌표)까지 받아 정답 좌표와 비교하고, 끝 근처 프레임의 STOP 시점을 잰다.

`noeun_openloop_wait.py` 는 첫 답만 기록한다 (pixel 이면 '↓' 에서 끝). 여기서는 평가 agent(`internvla_n1_s2_agent._query`)와
같이 '↓' 를 assistant 로 붙이고 look-down 이미지(60cm_15deg 는 현재 = look-down, `lookdown_same_view`)로 둘째 답을 받는다.
학습 로더(`NavPixelGoalDataset`)의 둘째 정답 = goal.<rig>[t] (같은 순서의 두 정수).

프레임: range(0, n-1, every) + 끝 --tail 프레임 전부 (STOP 시점). 각 행에 정답 종류 · 좌표 오차 · 정답 지점까지 높이 변화(계단)를 남긴다.

사용 (저장소 루트, GPU 1 장, habitat 이미지):
    /usr/bin/python scripts/eval/noeun_openloop_pixel.py --model_path <ckpt> --data_root <val_b1> --out <dir>
"""

import argparse
import json
import math
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from internnav.model.utils.vln_utils import split_and_clean  # noqa: E402
from noeun_openloop_wait import CONJ, PROMPT, kind_out  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--data_root', required=True, help='64_build_train.py 의 -val 데이터셋 (traj_data/r2r/<ds>_s<k>)')
    ap.add_argument('--scenes', default='hunav_s0')
    ap.add_argument('--rig', default='60cm_15deg')
    ap.add_argument('--every', type=int, default=4, help='표본 간격 (학습 sample_step 4)')
    ap.add_argument('--tail', type=int, default=12, help='끝에서 이 수만큼 프레임은 전부 (STOP 시점)')
    ap.add_argument('--num_history', type=int, default=8)
    ap.add_argument('--resize', type=int, default=384)
    ap.add_argument('--max_eps', type=int, default=0)
    ap.add_argument('--out', required=True)
    return ap.parse_args()


def last_pixel(text):
    m = re.findall(r'(\d+)\s+(\d+)', text)
    return [int(m[-1][0]), int(m[-1][1])] if m else None


def main():
    a = parse()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(a.model_path)
    processor.tokenizer.padding_side = 'left'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model_path, torch_dtype=torch.bfloat16,
                                                               attn_implementation='flash_attention_2', device_map={'': 'cuda:0'}).eval()

    def generate(messages, images):
        msg = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[msg], images=images, return_tensors='pt').to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True)
        return processor.tokenizer.decode(ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    f_rows = open(out / 'rows.jsonl', 'w')
    t0 = time.time()
    for sc in a.scenes.split(','):
        root = Path(a.data_root) / 'traj_data' / 'r2r' / sc
        eps = [json.loads(l) for l in open(root / 'meta' / 'episodes.jsonl')]
        for e in eps[:a.max_eps] if a.max_eps else eps:
            i, instr = e['episode_index'], e['tasks'][0].strip() + ' '
            t = pq.read_table(root / 'data' / 'chunk-000' / f'episode_{i:06d}.parquet',
                              columns=['action', f'pose.{a.rig}', f'goal.{a.rig}', f'relative_goal_frame_id.{a.rig}']).to_pydict()
            acts = [int(x) for x in t['action']][1:] + [0]
            rel = [int(x) for x in t[f'relative_goal_frame_id.{a.rig}']]
            goal = t[f'goal.{a.rig}']
            P = np.array([np.asarray(p)[:3, 3] for p in t[f'pose.{a.rig}']])
            n = len(acts)
            length_m = float(np.sum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)))
            raw = lambda f: Image.open(root / 'videos' / 'chunk-000' / f'observation.images.rgb.{a.rig}' / f'episode_{i:06d}_{f}.jpg').convert('RGB')
            img = lambda f: raw(f).resize((a.resize, a.resize))
            frames = sorted(set(range(0, n - 1, a.every)) | set(range(max(0, n - a.tail), n)))
            for f in frames:
                gt = 'stop' if f == n - 1 else ('pixel' if rel[f] >= 0 else 'turn')
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
                messages = [{'role': 'user', 'content': content}]
                o1 = generate(messages, images)
                o2, err = '', None
                if o1.strip() == '↓':
                    look = raw(f)  # lookdown_same_view · 원본 크기 (학습 로더 · agent 모두 look-down 은 resize 하지 않는다)
                    messages += [{'role': 'assistant', 'content': [{'type': 'text', 'text': o1}]},
                                 {'role': 'user', 'content': [{'type': 'text', 'text': CONJ}, {'type': 'image', 'image': look}, {'type': 'text', 'text': '.'}]}]
                    o2 = generate(messages, images + [look])
                    p2 = last_pixel(o2)
                    if gt == 'pixel' and p2 is not None:
                        err = math.dist(p2, goal[f])
                j = min(n - 1, f + max(rel[f], 1)) if rel[f] >= 0 else f
                row = {'scene': sc, 'episode': i, 'frame': f, 'n': n, 'length_m': round(length_m, 1), 'gt': gt, 'out': o1, 'out2': o2,
                       'pred': kind_out(o1 + o2), 'gt_pixel': goal[f] if rel[f] >= 0 else None, 'pred_pixel': last_pixel(o2),
                       'pixel_err': None if err is None else round(err, 1), 'up_m': round(float(P[j, 2] - P[f, 2]), 2),
                       'dz_ep_m': round(float(P[-1, 2] - P[0, 2]), 2), 'to_end_m': round(float(np.linalg.norm(P[-1, :2] - P[f, :2])), 2)}
                f_rows.write(json.dumps(row, ensure_ascii=False) + '\n')
                f_rows.flush()
            print(f'  {sc} ep {i}: 프레임 {len(frames)} {time.time() - t0:.0f}s', flush=True)
    f_rows.close()


if __name__ == '__main__':
    main()
