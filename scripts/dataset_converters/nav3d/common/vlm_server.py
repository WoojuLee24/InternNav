"""07 — 로컬 VLM 을 OpenAI 호환 `/v1/chat/completions` 로 띄운다 (instruction_labeling `label/backend.py --backend server` 가 그대로 붙는다).

이 PC 에는 labeling 기본 모델(Gemma-4-31B, vLLM, 8 GPU)이 없다 → 로컬에 있는 **Qwen3-VL-8B-Instruct** 를 transformers 로 돌린다.
요청 하나씩 순서대로 처리한다(GPU 1 장). 생성 코드(e2e.generate)는 바뀌지 않는다 — 모델 이름만 cfg 에서 바뀐다.

지원하는 요청 필드: messages(user content = text / image_url(data URI) 리스트), max_tokens, temperature, seed.
temperature > 0 이면 seed 로 torch 난수를 고정한다(같은 요청 = 같은 문장).

사용 (habitat 컨테이너, 저장소 루트; transformers 4.57 은 따로 설치한 경로를 PYTHONPATH 로):
    PYTHONPATH=/ws/src/InternNav/.venv_label/site /usr/bin/python scripts/dataset_converters/nav3d/common/vlm_server.py
"""

import argparse
import base64
import io
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI, Request
from PIL import Image


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model_path', default='checkpoints/Qwen/Qwen3-VL-8B-Instruct')
    ap.add_argument('--served_name', default='qwen3-vl-8b-instruct', help='configs/model/qwen3vl8b-local.yaml 의 model 과 같게')
    ap.add_argument('--port', type=int, default=8200)
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--attn', default='sdpa', help='sdpa | flash_attention_2')
    return ap.parse_args()


def to_messages(msgs: list) -> tuple:
    """OpenAI messages -> Qwen chat template messages + PIL 이미지 목록 (순서 유지)."""
    out, images = [], []
    for m in msgs:
        c = m['content']
        if isinstance(c, str):
            out.append({'role': m['role'], 'content': [{'type': 'text', 'text': c}]})
            continue
        parts = []
        for p in c:
            if p['type'] == 'text':
                parts.append({'type': 'text', 'text': p['text']})
            elif p['type'] == 'image_url':
                url = p['image_url']['url']
                assert url.startswith('data:'), f'data URI 만 지원: {url[:40]}'
                images.append(Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert('RGB'))
                parts.append({'type': 'image'})
            else:
                assert False, f"unreachable content type={p['type']!r}"
        out.append({'role': m['role'], 'content': parts})
    return out, images


def main():
    args = parse()
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, dtype=getattr(torch, args.dtype),
                                                            attn_implementation=args.attn, device_map='cuda:0').eval()
    proc = AutoProcessor.from_pretrained(args.model_path)
    print(f'[07 server] {args.model_path} 로드 {time.time() - t0:.0f}s · GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB', flush=True)
    lock = threading.Lock()
    app = FastAPI()
    stats = {'n': 0, 'sec': 0.0}

    @app.get('/v1/models')
    def models():
        return {'object': 'list', 'data': [{'id': args.served_name, 'object': 'model'}]}

    @app.post('/v1/chat/completions')
    async def chat(req: Request):
        body = await req.json()
        msgs, images = to_messages(body['messages'])
        temp = float(body.get('temperature', 0.0))
        with lock:
            t = time.time()
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[text], images=images or None, return_tensors='pt').to(model.device)
            if body.get('seed') is not None:
                torch.manual_seed(int(body['seed']))
            kw = {'do_sample': True, 'temperature': temp, 'top_p': 1.0, 'top_k': 0} if temp > 0 else {'do_sample': False}
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=int(body.get('max_tokens', 256)), **kw)
            gen = out[0, inp['input_ids'].shape[1]:]
            reply = proc.decode(gen, skip_special_tokens=True).strip()
            stats['n'] += 1
            stats['sec'] += time.time() - t
            if stats['n'] % 20 == 0:
                print(f"[07 server] {stats['n']} 요청 · 평균 {stats['sec'] / stats['n']:.1f}s · 입력 토큰 {inp['input_ids'].shape[1]}", flush=True)
        return {'id': f"chatcmpl-{stats['n']}", 'object': 'chat.completion', 'model': args.served_name,
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': reply}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': int(inp['input_ids'].shape[1]), 'completion_tokens': int(len(gen))}}

    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
