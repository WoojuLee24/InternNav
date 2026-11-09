"""S2 단독 에이전트의 GGUF(llama.cpp) 백엔드 — 경량화 모델 (`internvla_n1_s2` 와 같은 대화 · 같은 행동 규칙).

`InternVLAN1S2Agent` 를 상속해 모델 로딩(`_load_model`)과 생성(`_generate`)만 바꾼다. 프롬프트 · history 선택 ·
'↓' 대화 이어가기 · WAIT · pixel goal 추종은 부모 코드를 그대로 쓴다 (HF 와 같은 코드 경로).

생성은 vln-deploy 의 `llama-s2-runner` (stdin/stdout 줄 프로토콜) 로 한다:
  RESET                         대화 · KV 비움
  TEXT \t img1,img2 \t prompt   user 턴 하나를 KV 에 이어 넣고 답을 생성 (답도 KV 에 남는다)
부모의 messages 는 새 질의면 [user], '↓' 이어가기면 [user, assistant, user] 다. 이어가기의 앞 두 턴은 이미 runner KV 에
있으므로 마지막 user 턴만 보낸다.

HF 와 맞추는 설정 (vln-deploy report §5-2 의 HF parity):
  MTMD_QWEN25VL_HF_ALIGN=1  이미지 28 px 정렬 (HF smart_resize: 384 -> 392, 640x480 -> 644x476)
  S2_HF_CHAT=1              Qwen 기본 system 메시지 + 답의 EOG 를 KV 에 유지
  --temp 0 --repeat-penalty 1.05 --repeat-last-n <n_ctx>   ckpt generation_config (do_sample=False, repetition_penalty 1.05)

model_settings 추가 키 (기본값은 GGUF_DEFAULTS):
  gguf_text / gguf_mmproj   GGUF 파일 (tools/build_qwen_s2_gguf.sh 산출물)
  runner_bin                llama-s2-runner 경로. 같은 폴더의 .so 를 LD_LIBRARY_PATH 로 쓴다
  n_ctx                     KV 크기 (이미지 9장 x 196 토큰 + 텍스트 < 4096)
  embd_cache                vision 임베딩 캐시 개수 (history 이미지 재인코딩 생략, 출력 동일). 0 = 끔
  img_dir                   runner 에 넘길 이미지 임시 폴더 (PNG, 무손실)
  runner_args               runner 에 덧붙일 llama.cpp 인자 (공백 구분)
`device` 는 쓰지 않는다 — runner 는 CUDA_VISIBLE_DEVICES 로 보이는 GPU 를 쓴다.
"""
import os
import subprocess
import tempfile

from internnav.agent.base import Agent
from internnav.agent.internvla_n1_s2_agent import InternVLAN1S2Agent

MEDIA_MARKER = '<__media__>'  # mtmd 기본 이미지 자리표시 (qwen2vl: <|vision_start|><|image_pad|><|vision_end|> 로 바뀜)
_QK = 'vln-deploy/quantization_kit'
GGUF_DEFAULTS = {
    'gguf_text': f'{_QK}/out/gguf_noeun_v01_b1wait/text-Q8_0-embf16.gguf',
    'gguf_mmproj': f'{_QK}/out/gguf_noeun_v01_b1wait/mmproj-f16.gguf',
    'runner_bin': f'{_QK}/llama.cpp/build_x86/bin/llama-s2-runner',
    'n_ctx': 8192,
    'embd_cache': 512,
    'img_dir': '',  # 비면 /dev/shm 아래 임시 폴더
    'runner_args': '',  # runner 에 덧붙일 인자 (예: '-ctk f32 -ctv f32 -fa off'). 비면 llama.cpp 기본
}


def hf_processor_resize(img):
    """HF Qwen2VLImageProcessor 와 같은 리사이즈 (smart_resize 28 px 정렬 + PIL BICUBIC): 384x384 -> 392x392,
    640x480 -> 644x476. llama.cpp 는 HF_ALIGN 이면 28 정렬 이미지를 그대로 쓰므로, 리사이즈를 여기서 HF 와 같게 한다
    (llama.cpp 자체 bicubic 은 HF 와 픽셀이 달라 출력이 갈린다 — verify_s2_gguf_noeun.py)."""
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    h, w = smart_resize(img.height, img.width, factor=28, min_pixels=3136, max_pixels=12845056)
    return img if (w, h) == img.size else img.resize((w, h), Image.BICUBIC)


@Agent.register('internvla_n1_s2_gguf')
class InternVLAN1S2GGUFAgent(InternVLAN1S2Agent):
    def _load_model(self):
        ms = dict(self.config.model_settings)
        g = {k: ms.get(k) if ms.get(k) is not None else v for k, v in GGUF_DEFAULTS.items()}
        self.gguf = g
        self.img_dir = g['img_dir'] or tempfile.mkdtemp(prefix='s2gguf_', dir='/dev/shm')
        os.makedirs(self.img_dir, exist_ok=True)
        bin_dir = os.path.dirname(os.path.abspath(g['runner_bin']))
        env = dict(os.environ)
        env['LD_LIBRARY_PATH'] = bin_dir + ':' + env.get('LD_LIBRARY_PATH', '')
        env['MTMD_QWEN25VL_HF_ALIGN'] = '1'
        env['S2_HF_CHAT'] = '1'
        env['S2_EMBD_CACHE'] = str(int(g['embd_cache']))
        cmd = [
            g['runner_bin'], '-m', g['gguf_text'], '--mmproj', g['gguf_mmproj'],
            '-ngl', '99', '-c', str(int(g['n_ctx'])), '-n', str(int(self.s['max_new_tokens'])),
            # repeat-last-n = n_ctx (HF 처럼 전체 시퀀스). -1 은 이 llama.cpp 에서 0(penalty 꺼짐)이 된다
            # (common.cpp 의 -1 -> n_ctx 변환이 주석 처리됨, llama-sampler.cpp max(last_n, 0))
            '--temp', '0', '--repeat-penalty', '1.05', '--repeat-last-n', str(int(g['n_ctx'])),
        ] + g['runner_args'].split()  # fmt: skip
        self.runner = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True, bufsize=1
        )
        assert self._call('RESET') == 'OK', 'llama-s2-runner failed to start'
        self.n_img = 0

    def _call(self, line):
        self.runner.stdin.write(line + '\n')
        self.runner.stdin.flush()
        resp = self.runner.stdout.readline()
        if not resp:
            raise RuntimeError(f'llama-s2-runner exited (code {self.runner.poll()})')
        return resp.rstrip('\n')

    def _generate(self, messages, images):
        if len(messages) == 1:  # 새 질의
            assert self._call('RESET') == 'OK'
        elif len(messages) >= 3 and messages[-2]['role'] == 'assistant':  # '↓' 이어가기: 앞 턴은 KV 에 있다
            pass
        else:
            assert False, f"unreachable messages roles={[m['role'] for m in messages]!r}"
        files, prompt = [], ''
        for part in messages[-1]['content']:
            if part['type'] == 'image':
                path = os.path.join(self.img_dir, f'{self.n_img % 64:02d}.png')  # 한 턴 최대 9장 < 64
                self.n_img += 1
                hf_processor_resize(part['image']).save(path)
                files.append(path)
                prompt += MEDIA_MARKER
            elif part['type'] == 'text':
                prompt += part['text']
            else:
                assert False, f"unreachable content type={part['type']!r}"
        resp = self._call('TEXT\t' + ','.join(files) + '\t' + prompt.replace('\n', '\\n'))
        assert resp.startswith('OK\t') or resp == 'OK', f'llama-s2-runner: {resp[:200]}'
        return resp[3:]

    def __del__(self):
        r = getattr(self, 'runner', None)
        if r is not None and r.poll() is None:
            try:
                r.stdin.write('QUIT\n')
                r.stdin.flush()
                r.wait(timeout=5)
            except Exception:
                r.kill()
