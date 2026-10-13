# repeat_train.sh 실행 실패 원인 (2026-09-19)

## 직접 원인
`ModuleNotFoundError: No module named 'flash_attn'` — 8개 rank 전부 import 단계에서 사망.

```
internnav/trainer/internvla_n1_trainer.py:31   from qwenvl_base import replace_qwen2_vl_attention_class
internnav/trainer/qwenvl_base.py:4             from flash_attn.flash_attn_interface import flash_attn_varlen_func
```

optional import이 아님:
- `internvla_n1_trainer.py:288` → `train(attn_implementation="flash_attention_2")` 하드코딩
- `internvla_n1_trainer.py:219` → `replace_qwen2_vl_attention_class()` 로 Qwen attention을 flash_attn_varlen_func 로 교체

`pyproject.toml` / requirements 에 flash-attn 선언이 없어서 `pip install -e .` 로는 절대 안 깔림.

## 환경
python 3.12.3 (/usr/bin/python), torch 2.9.0+cu130, transformers 4.51.0,
deepspeed 0.18.8, accelerate 1.13.0, cxx11abi=True, x86_64, H200 x8

주의: CLAUDE.md 는 habitat 환경 pip 를 `/usr/bin/pip` 라고 적었지만 이 머신엔 해당 파일이 없음.
→ `/usr/bin/python -m pip` 사용해야 함.

## 해결
정확히 맞는 prebuilt wheel 존재 (HTTP 200, 244,226,500 bytes 확인):

```
/usr/bin/python -m pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

`pip install flash-attn` (PyPI) 는 금지 — sdist 뿐이라 소스 컴파일로 30분~2시간 소요.

## 스크립트 자체 문제
`while true` 루프에 실패 처리가 없음. 240초 동안 66회 재시도하며 매번 8프로세스를 띄움.
에러가 나도 멈추지 않으므로 로그가 폭주함.

## flash_attn 뒤에 대기 중인 블로커 (이 머신은 환경 세팅이 안 됨)
1. `checkpoints/InternVLA-N1-System2` 없음 → 실물: `/home/irteam/data-vol2/checkpoints/InternVLA-N1-System2`
2. `data/` 디렉터리 자체가 없음 → 실물: `/home/irteam/data-vol1/InternData-N1-v0.5-mini`
   (둘 다 .gitignore 에 등록된 로컬 심볼릭 링크 대상)
3. `vln_ce/traj_data/r2r` 가 아직 .tar.gz 상태 → `scripts/dataset_converters/extract_dataset.sh` 로 풀어야 함
   (dataset 정의: `internnav/dataset/internvla_n1_lerobot_dataset.py` 의 `data_path: "traj_data/r2r"`)

---

## 조치 내역 (2026-09-20)

### 1. flash_attn 설치 (완료)
```
/usr/bin/python -m pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```
→ `Successfully installed flash-attn-2.8.3`, `qwenvl_base` import 정상 확인.

### 2. 심볼릭 링크 (완료)
repo `checkpoints/` 는 스크립트가 만든 빈 디렉터리뿐(파일 0개)이어서 제거 후 링크로 교체.
```
data        -> /home/irteam/data-vol1
checkpoints -> /home/irteam/data-vol2/checkpoints
```

### 3. 2차 블로커: 데이터가 압축 상태 (완료)
flash_attn 해결 후 모델 로드까지는 통과했으나 다음 에러 발생:
```
sampling 0 examples from dataset {'data_path': 'traj_data/r2r', ...}
ZeroDivisionError: division by zero   (internvla_n1_lerobot_dataset.py:1595)
```
로더가 `os.path.isdir` 로 scene 디렉터리만 세는데(`:801`) tar.gz 61개뿐이고 디렉터리 0개였음.

```
bash scripts/dataset_converters/extract_dataset.sh data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r 24
```
→ 61/61 해제, 실패 0건, 312G → 640G.

### 4. 결과: 학습 정상 시작 확인
```
sampling 34359 examples (r2r_125cm_0_30) + 40305 examples (r2r_60cm_15_15)
{'loss': 1.1926, 'grad_norm': 17.5156, 'learning_rate': 0.0,     'epoch': 0.0}
{'loss': 1.2024, 'grad_norm': 18.5693, 'learning_rate': 2e-05,   'epoch': 0.0}
{'loss': 1.0838, 'grad_norm': 15.3574, 'learning_rate': 4e-05,   'epoch': 0.01}
1/1515 steps, 24.53s/it -> 15.22s/it, H200 x8 각 63~72GB, Traceback 0건
```
검증용으로 timeout 걸어 돌린 것이라 3스텝 후 정리함. GPU 8장 모두 0MiB 로 해제 확인.

### 5. 남은 확인 사항
- `while true` 루프에 실패 처리 없음. 에러가 나도 멈추지 않고 재시도(240초에 66회) → 사용자 확인 필요.
