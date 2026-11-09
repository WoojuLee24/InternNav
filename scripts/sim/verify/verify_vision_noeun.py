"""노은 S2 vision encoder 일치: HF model.visual vs llama.cpp mmproj (runner EMBD), 같은 사전 리사이즈 이미지.

정면 = 384x384 -> 392x392, look-down = 640x480 -> 644x476 (internvla_n1_s2_gguf_agent.hf_processor_resize).
토큰별 cosine 을 내고, look-down 은 행(row) 별 평균도 낸다 (세로 위치 처리 오류면 특정 행에서 무너진다).

  # [internnav-isaacsim] HF 는 GPU2, runner 는 GPU3
  CUDA_VISIBLE_DEVICES=2 RUNNER_GPU=3 /workspace/isaaclab/_isaac_sim/python.sh scripts/sim/verify/verify_vision_noeun.py --model <ckpt>
"""
import argparse
import base64
import glob
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.getcwd())


def runner_embd(paths, a):
    from internnav.agent.internvla_n1_s2_gguf_agent import GGUF_DEFAULTS

    b = GGUF_DEFAULTS['runner_bin']
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(b), MTMD_QWEN25VL_HF_ALIGN='1', S2_EMBD_CACHE='64',
               CUDA_VISIBLE_DEVICES=os.environ.get('RUNNER_GPU', '3'))
    cmd = [b, '-m', f'{a.gguf_dir}/{a.text}', '--mmproj', f'{a.gguf_dir}/mmproj-f16.gguf', '-ngl', '99', '-c', '4096']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True, bufsize=1)
    out = []
    for f in paths:
        p.stdin.write(f'EMBD\t{f}\n')
        p.stdin.flush()
        ok, shape, data = p.stdout.readline().rstrip('\n').split('\t')
        assert ok == 'OK', ok
        nx, ny, d = map(int, shape.split(','))
        out.append((nx, ny, np.frombuffer(base64.b64decode(data), np.float32).reshape(-1, d)))
    p.stdin.write('QUIT\n')
    p.stdin.flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--gguf_dir', default='vln-deploy/quantization_kit/out/gguf_noeun_v01_b1wait')
    ap.add_argument('--text', default='text-Q8_0-embf16.gguf')
    ap.add_argument('--frames', default='logs/noeun_smoke/agent/ep0000')
    ap.add_argument('--n', type=int, default=4)
    a = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from internnav.agent.internvla_n1_s2_gguf_agent import hf_processor_resize

    frames = sorted(glob.glob(f'{a.frames}/rgb_*.jpg'))[:: max(1, 180 // a.n)][: a.n]
    imgs, paths = [], []
    os.makedirs('/dev/shm/vis_noeun', exist_ok=True)
    for i, f in enumerate(frames):
        im = Image.open(f).convert('RGB')
        for kind, x in (('front', hf_processor_resize(im.resize((384, 384)))), ('look', hf_processor_resize(im))):
            path = f'/dev/shm/vis_noeun/{i}_{kind}.png'
            x.save(path)
            imgs.append((kind, x))
            paths.append(path)
    g = runner_embd(paths, a)

    proc = AutoProcessor.from_pretrained(a.model)
    vis = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float32).visual.to('cuda:0')
    for (kind, x), (nx, ny, e_g) in zip(imgs, g):
        inp = proc.image_processor(images=[x], return_tensors='pt')
        with torch.no_grad():
            e_h = vis(inp['pixel_values'].to('cuda:0'), grid_thw=inp['image_grid_thw'].to('cuda:0')).cpu().numpy()
        cos = (e_g * e_h).sum(1) / (np.linalg.norm(e_g, axis=1) * np.linalg.norm(e_h, axis=1))
        line = f'{kind:5s} {x.size} grid {inp["image_grid_thw"].tolist()} gguf nx,ny {nx},{ny}  cos med {np.median(cos):.4f} min {cos.min():.4f}'
        if kind == 'look':
            rows = cos.reshape(ny, nx).mean(1)
            line += '  row-mean ' + ' '.join(f'{r:.3f}' for r in rows)
        print(line, flush=True)


if __name__ == '__main__':
    main()
