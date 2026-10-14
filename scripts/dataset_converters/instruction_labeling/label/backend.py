"""VLM 백엔드 — 생성 코드는 OpenAI 호환 client 하나만 본다. 백엔드는 실행 옵션(`annotate.py --backend`)이며
생성 결과에는 영향을 주지 않으므로 결과 이름·gen_hash 에 들어가지 않는다.

  server  : 이미 떠 있는 vLLM 서버에 요청 (model yaml 의 base_url). GPU 는 서버가 계속 잡고 있다.
  managed : job 이 시작할 때 vLLM 서버를 **자식 프로세스로 직접 띄우고**, 끝나면(예외·Ctrl-C 포함) 내려서 GPU 를 반납한다.
            queue 에서 학습 job 과 번갈아 돌릴 때 쓴다. 서버 인자는 model yaml 의 `managed:` 블록.
            (in-process `vllm.LLM` 대신 이 방식을 쓰는 이유: labeling 코드는 habitat 파이썬(pandas/pyarrow)에서 돌고
             vllm 은 별도 venv 에 있어서, 한 프로세스에 둘을 섞지 않고 HTTP 로 잇는다. 생성 코드는 그대로다.)
"""

import contextlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _ready(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url + '/models', timeout=5) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


@contextlib.contextmanager
def managed_server(model_cfg: dict, log_path: Path, gpus: str = None):
    """vLLM 서버를 띄우고 base_url 을 돌려준다. with 블록을 나가면 프로세스 그룹째 종료."""
    m = model_cfg['managed']
    gpus = gpus or m['gpus']
    n_gpu = len([g for g in str(gpus).split(',') if g.strip()])
    tp = int(m['tensor_parallel_size'])
    assert n_gpu % tp == 0, f'GPU {n_gpu} 장이 tensor_parallel_size {tp} 의 배수가 아니다'
    dp = n_gpu // tp  # GPU 수에 맞춰 data parallel 을 정한다 (--gpus 로 줄여도 동작)
    port = _free_port()
    cmd = [m['vllm_bin'], 'serve', m['path'], '--served-model-name', model_cfg['model'], '--port', str(port),
           '--tensor-parallel-size', str(m['tensor_parallel_size']),
           '--data-parallel-size', str(dp),
           '--max-model-len', str(m['max_model_len']),
           '--limit-mm-per-prompt', json.dumps(m['limit_mm_per_prompt']),
           '--gpu-memory-utilization', str(m['gpu_memory_utilization'])] + [str(x) for x in m.get('extra_args', [])]
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(gpus)
    env.pop('PYTHONPATH', None)  # 호스트 PYTHONPATH(ROS 등)가 venv 에 섞이지 않게
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, 'a')
    print(f'[backend] managed vLLM 시작: GPU {env["CUDA_VISIBLE_DEVICES"]} port {port} 로그 {log_path}', flush=True)
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    base_url = f'http://127.0.0.1:{port}/v1'
    try:
        t0 = time.time()
        while not _ready(base_url):
            if proc.poll() is not None:
                raise RuntimeError(f'vLLM 서버가 준비 전에 종료됨 (code {proc.returncode}). 로그: {log_path}')
            if time.time() - t0 > float(m['startup_timeout_s']):
                raise TimeoutError(f'vLLM 서버가 {m["startup_timeout_s"]} 초 안에 준비되지 않음. 로그: {log_path}')
            time.sleep(5)
        print(f'[backend] 준비 완료 ({time.time() - t0:.0f}s)', flush=True)
        yield base_url
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        logf.close()
        print(f'[backend] managed vLLM 종료 (code {proc.returncode})', flush=True)


@contextlib.contextmanager
def client(model_cfg: dict, backend: str, log_dir: Path, gpus: str = None):
    from openai import OpenAI
    if backend == 'server':
        yield OpenAI(base_url=model_cfg['base_url'], api_key=model_cfg['api_key'], timeout=300)
    elif backend == 'managed':
        with managed_server(model_cfg, log_dir / 'vllm_server.log', gpus) as url:
            yield OpenAI(base_url=url, api_key='managed', timeout=300)
    else:
        assert False, f'unreachable backend={backend!r}'
