"""노은 S2: HF bf16 vs GGUF 출력 일치 검증 (open-loop replay).

저장된 에피소드 프레임(agent save_images 의 rgb_XXXX.jpg)을 순서대로 쌓으며 `every` 프레임마다 S2 를 한 번 질의한다.
질의는 에이전트의 `_query` 그대로 (프롬프트 · history linspace · conjunction rng 동일). 두 백엔드를 따로 돌려 json 으로 남기고
cmp 로 비교한다. '↓' 이어가기는 같은 현재 프레임을 look-down 이미지로 한 번 더 질의한다 (lookdown_same_view 와 같다).

  # [internnav-isaacsim] 두 백엔드를 다른 GPU 에서 동시에
  CUDA_VISIBLE_DEVICES=2 /workspace/isaaclab/_isaac_sim/python.sh scripts/sim/verify/verify_s2_gguf_noeun.py run --backend hf --model <ckpt>
  CUDA_VISIBLE_DEVICES=3 /workspace/isaaclab/_isaac_sim/python.sh scripts/sim/verify/verify_s2_gguf_noeun.py run --backend gguf --gguf_dir <dir>
  /workspace/isaaclab/_isaac_sim/python.sh scripts/sim/verify/verify_s2_gguf_noeun.py cmp --ref hf --test gguf
  (F16 GGUF: --gguf_dir <dir>_f16 --text text-f16.gguf --tag gguf_f16)
"""
import argparse
import glob
import json
import os
import re
import sys
import time

OUT = 'scripts/sim/verify/out/s2_gguf_noeun'


def run(a):
    import numpy as np
    from PIL import Image

    from internnav.configs.agent import AgentCfg

    ms = {'model_path': a.model, 'device': 'cuda:0', 'log_dir': ''}
    if a.backend == 'hf':
        from internnav.agent.internvla_n1_s2_agent import InternVLAN1S2Agent as Cls
    elif a.backend == 'hf_fp16':  # 같은 HF, fp16 — llama.cpp(f16/f32 계산) 와 같은 정밀도 조건의 기준
        from internnav.agent.internvla_n1_s2_agent import InternVLAN1S2Agent

        class Cls(InternVLAN1S2Agent):
            def _load_model(self):
                import torch
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

                self.processor = AutoProcessor.from_pretrained(self.model_path)
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=torch.float16, attn_implementation='flash_attention_2',
                    device_map={'': self.device},
                ).eval()

    elif a.backend == 'hf_sdpa':  # 같은 HF bf16, attention 커널만 sdpa — HF 자체의 수치 흔들림(기준 잡음)
        from internnav.agent.internvla_n1_s2_agent import InternVLAN1S2Agent

        class Cls(InternVLAN1S2Agent):
            def _load_model(self):
                import torch
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

                self.processor = AutoProcessor.from_pretrained(self.model_path)
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': self.device}
                ).eval()

    elif a.backend == 'gguf':
        from internnav.agent.internvla_n1_s2_gguf_agent import InternVLAN1S2GGUFAgent as Cls

        ms.update(gguf_text=f'{a.gguf_dir}/{a.text}', gguf_mmproj=f'{a.gguf_dir}/mmproj-f16.gguf', runner_args=a.runner_args)
    else:
        assert False, f'unreachable backend={a.backend!r}'
    agent = Cls(AgentCfg(server_port=0, model_name='x', ckpt_path='', model_settings=ms))
    frames = sorted(glob.glob(os.path.join(a.frames, 'rgb_*.jpg')))[: a.max_frames]
    print(f'[{a.backend}] {len(frames)} frames', flush=True)
    rows = []
    for k, f in enumerate(frames):
        img = Image.open(f).convert('RGB')
        agent.rgb_list.append(img.resize((agent.s['resize_w'], agent.s['resize_h'])))
        if k % a.every:
            continue
        if a.single_look:  # 대화 이어가기 없이 look-down 이미지 하나로 단일 턴 (2 번째 턴 문제 분리용)
            from internnav.agent.internvla_n1_s2_agent import PROMPT

            text = PROMPT.replace('<instruction>.', a.instruction[:-1]) + ' you can see'
            msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': text}, {'type': 'image', 'image': img}]}]
            t0 = time.time()
            out = agent._generate(msgs, [img])
            rows.append({'k': k, 'out': out, 'lat': round(time.time() - t0, 3)})
            print(f'[{a.backend}] k={k} single_look {out!r}', flush=True)
            continue
        agent.step_id, agent.messages, agent.prev_action = k, [], None
        lat = agent._query(a.instruction)
        out = agent.llm_outputs
        row = {'k': k, 'out': out, 'lat': round(lat, 3)}
        if '↓' in out:  # look-down 이어가기
            agent.prev_action, agent.look_down_image = 5, img
            row['lat_ld'] = round(agent._query(a.instruction), 3)
            row['out_ld'] = agent.llm_outputs
        rows.append(row)
        print(f"[{a.backend}] k={k} {row['lat']:.2f}s {out!r}" + (f" | ld {row['out_ld']!r}" if 'out_ld' in row else ''),
              flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({'backend': a.backend, 'rows': rows, 'instruction': a.instruction},
              open(f'{OUT}/{a.tag or a.backend}.json', 'w'), ensure_ascii=False, indent=1)


def kind(out):
    """S2 답 종류: pixel / action 문자열 / wait / stop / other"""
    if re.search(r'\d', out):
        return 'pixel'
    if 'WAIT' in out:
        return 'wait'
    acts = ''.join(c for c in out if c in '↑←→↓')
    if acts:
        return acts
    return 'stop' if 'STOP' in out else 'other'


def cmp(a):
    import numpy as np

    ref = {r['k']: r for r in json.load(open(f'{OUT}/{a.ref}.json'))['rows']}
    tst = {r['k']: r for r in json.load(open(f'{OUT}/{a.test}.json'))['rows']}
    ks = sorted(set(ref) & set(tst))
    same_text = same_kind = 0
    dpx, lat_r, lat_t = [], [], []
    for k in ks:
        r, t = ref[k], tst[k]
        out_r, out_t = r.get('out_ld', r['out']), t.get('out_ld', t['out'])  # 최종 답 (↓ 면 look-down 답)
        same_text += r['out'] == t['out'] and r.get('out_ld') == t.get('out_ld')
        same_kind += kind(out_r) == kind(out_t)
        if kind(out_r) == kind(out_t) == 'pixel':
            pr = [int(c) for c in re.findall(r'\d+', out_r)][:2]
            pt = [int(c) for c in re.findall(r'\d+', out_t)][:2]
            dpx.append(float(np.hypot(pr[0] - pt[0], pr[1] - pt[1])))
        lat_r.append(r['lat'] + r.get('lat_ld', 0))
        lat_t.append(t['lat'] + t.get('lat_ld', 0))
        if out_r != out_t:
            print(f'k={k:4d}  {a.ref}={out_r!r:16}  {a.test}={out_t!r}')
    n = len(ks)
    rep = {
        'n_query': n,
        'text_match': same_text / n,
        'kind_match': same_kind / n,
        'pixel_n': len(dpx),
        'pixel_exact': float(np.mean([d == 0 for d in dpx])) if dpx else None,
        'pixel_dist_median': float(np.median(dpx)) if dpx else None,
        'pixel_within_10px': float(np.mean([d <= 10 for d in dpx])) if dpx else None,
        f'latency_p50_{a.ref}': float(np.median(lat_r)),
        f'latency_p50_{a.test}': float(np.median(lat_t)),
    }
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(f'{OUT}/report_{a.test}_vs_{a.ref}.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'cmp'])
    ap.add_argument('--backend', choices=['hf', 'hf_fp16', 'hf_sdpa', 'gguf'])
    ap.add_argument('--model', default='')
    ap.add_argument('--gguf_dir', default='vln-deploy/quantization_kit/out/gguf_noeun_v01_b1wait')
    ap.add_argument('--text', default='text-Q8_0-embf16.gguf', help='gguf_dir 안의 text decoder 파일')
    ap.add_argument('--runner_args', default='', help='[gguf] llama-s2-runner 추가 인자 (예: "-ctk f32 -ctv f32")')
    ap.add_argument('--single_look', action='store_true', help='look-down 이미지 단일 턴만 (대화 이어가기 제외)')
    ap.add_argument('--tag', default='', help='결과 이름 (기본 = backend)')
    ap.add_argument('--ref', default='hf')
    ap.add_argument('--test', default='gguf')
    ap.add_argument('--frames', default='logs/noeun_smoke/agent/ep0000')
    ap.add_argument('--every', type=int, default=4)
    ap.add_argument('--max_frames', type=int, default=200)
    ap.add_argument('--instruction', default='Walk straight ahead and stop in front of the ticket gate.')
    a = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    {'run': run, 'cmp': cmp}[a.cmd](a)
