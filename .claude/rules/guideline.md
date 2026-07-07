# Project Guidelines

## Python / pip 실행 경로 (Isaaclab 설치 환경)
- `python` / `python3` → `/workspace/isaaclab/_isaac_sim/python.sh`
- `pip` / `pip3` → `/workspace/isaaclab/_isaac_sim/python.sh -m pip`
- `.bashrc`의 alias는 non-interactive shell에서 적용되지 않으므로 항상 full path 사용

## Python / pip 실행 경로 (Habitat 설치 환경)
- `python` / `python3` → `/usr/bin/python`
- `pip` / `pip3` → `/usr/bin/pip`


## 개발 제한사항

### 1. 코드 변경은 최소한으로 (Conservative Changes)

- **기존 함수/클래스의 시그니처를 바꾸지 않는다.**
  - 새로운 옵션이 필요하면 필수 인자가 아닌 **기본값이 있는 optional argument**로 추가한다.
    - 예: `def __init__(self, ..., use_depth: bool = False)`
  - 기존 호출부(caller)는 인자를 추가하지 않아도 기존과 동일하게 동작해야 한다. (하위 호환성 보장)
  - **기존 main branch**에 있었던 코드와 비교하고 수정을 최소화한다.
- **분기(if문) 규칙: guard clause는 허용, 로직 내부 침투는 금지**
  - **허용**: 함수 맨 앞에서 조건을 확인해 완전히 별도의 함수(또는 새 파일의 메서드)로 위임하고, 그 안에서 기존 로직을 전혀 건드리지 않는 경우.
```python
    # 허용 - guard clause + 위임
    def forward(self, x):
        if self.use_depth:
            return self._forward_with_depth(x)  # 새 파일/새 메서드로 완전히 위임
        return self._forward_default(x)          # 기존 로직 그대로, 손 대지 않음
```
  - **금지**: 기존 로직 중간에 분기를 끼워 넣어 변수/텐서/흐름을 새 로직과 공유하는 경우. (default 경로가 진짜로 안 변했는지 코드만 봐서는 보장이 안 되고, 새 기능이 추가될수록 분기 조합이 기하급수적으로 늘어나 검증이 어려워짐)
```python
    # 금지 - 로직 내부에 분기가 섞여 흐름/변수를 공유
    def forward(self, x):
        feat = self.backbone(x)
        if self.use_depth:
            feat = self.depth_fusion(feat, x)   # 이후 코드가 이 변경된 feat을 그대로 사용
        feat = self.neck(feat)
        return self.head(feat)
```
  - 판단이 애매하면: "`use_depth=False`일 때 이 함수가 diff 이전과 100% 동일한 코드 경로를 타는가?"를 기준으로 확인한다. 아니라면 guard clause + 위임 구조로 리팩터링한다.

- **기존 파일의 수정 범위를 스스로 판단하는 기준**
  - 위 guard clause 한 줄 추가 정도를 넘어서서, 새 기능 관련 로직이 여러 곳에 흩어지거나 기존 함수 본문을 수정해야 하면 → 새 파일로 분리한다.

### 2. Config/Argument로 조절 가능하게

- 새 기능의 활성화 여부와 파라미터는 **하드코딩 금지**, config 파일(yaml/json 등)에서 조정 가능해야 한다.
- config에 새 필드를 추가할 때, **기존 config 파일에 필드가 없어도 에러 없이 기본 동작(default)** 으로 돌아가야 한다.
  - 예: `cfg.get("use_depth", False)` 형태로 안전하게 접근.
- 기존 config 스키마의 필드명/구조를 변경하지 않는다. (필드 추가는 되지만, 이름 변경/삭제는 금지)

### 3. 새 파일 생성 시 상속을 우선 사용

- 새로운 기능은 원칙적으로 **새 파일 + 클래스 상속**으로 구현한다.
  - 예: `class DepthAwareEncoder(BaseEncoder): ...` 처럼 기존 클래스를 상속하고, 필요한 메서드만 오버라이드한다.
  - 오버라이드하는 메서드는 가능하면 `super().method()`를 호출해 기존 로직을 재사용하고, 그 위에 추가 로직만 덧붙인다.

- **컴포지션은 상속이 불가능하거나 부적절한 경우에만 예외적으로 사용한다.**
  - 예: 기존 클래스가 서드파티/외부 라이브러리라 상속이 불가능할 때
  - 예: 여러 기존 객체(BEV encoder + Depth encoder 등)를 조합해 하나의 새 provider를 만들어야 할 때
  - 예: 기존 클래스의 `__init__`이 복잡해서 상속 시 생성자 오버라이드가 오히려 지저분해질 때

- 새 파일은 기존 파일과 동일한 디렉토리 구조/네이밍 컨벤션을 따른다. (예: `encoders/base_encoder.py` → `encoders/depth_encoder.py`)
- 새 클래스를 기존 코드에 연결할 때는, 기존 코드에서 조건 분기로 새 클래스를 import하는 대신 **factory 함수나 registry 패턴**으로 등록해서 기존 코드 수정을 최소화한다.
  - 예: `ENCODER_REGISTRY = {"base": BaseEncoder, "depth": DepthAwareEncoder}` 후 config의 `encoder_type` 값으로 선택.

### 4. 검증

- 변경 후에는 반드시 명령어를 실행해 **기존 기능(default 경로)이 이전과 동일하게 동작하는지**, **신규 기능이 의도대로 동작하는지** 둘 다 확인한다.
- 기존 테스트가 있다면 새 코드 작성 전/후 모두 통과하는지 확인한다.
- 개발 목표와 관련된 코드는 시각화해서 저장해서 디버깅해라. 


## Command
### Training & evaluation
`python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/bev/base_s1.fpv_s2.fpv_rgb_gt.py --machine 5090` 
### Training only (debugging)
`--no-eval --debug-dir logs/260615_base_s1.fpv_s2.fpv_rgb_gt`  
### Evaluation only (debugging)
` --no-train  --debug-dir logs/260615_base_s1.fpv_s2.fpv_rgb_gt`


## 명령어 (command) 출력 규칙
- 멀티라인 `\` 이음 명령어는 copy-paste 시 `\` 뒤 공백으로 실패하므로 **항상 한 줄**로 제공

## md 파일 작성/업데이트 위치
- memory·계획·정리 등 md 파일을 작성하거나 update할 때는 **`InternNav/.claude/` 하위에 작성** (repo에 체크인되도록)
  - 예: memory → `.claude/memory/`, 태스크 → `.claude/tasks/`, 이해 문서 → `.claude/understanding/`
- auto-memory(`/root/.claude/projects/...`)에만 쓰지 말고 항상 repo `.claude/` 쪽도 동기화

## 결과 보고 및 정리: 명료하고 짧게 작성
- 이전과 비교해서 어떤 점이 달라졌는지 
- command 명령을 작성하고 각 argument의 의미
- 시각화 및 디버깅 결과 보고
- 결과 보고는 tasks/{task명}_result.md에 작성해라.

## r2r_h1_replay 데이터 수집 (eval_r2r_h1_replay.py)
```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/eval/eval_r2r_h1_replay.py --config scripts/eval/configs/h1_internvla_n1_async_cfg_orig.py --r2r_dir data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r --save_dir data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r_h1_replay --scenes <SCENE> --max_eps <N>
```
- `--scenes` 생략 시 전체 scene, `--max_eps` 생략 시 scene당 전체 episode
- `--skip_existing` : 이미 저장된 scene/episode 건너뜀

## debug 이미지 저장 (save_debug_images.py)
```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/eval/save_debug_images.py --src_dir data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r --scan <SCENE> --ep <N>
```
- `--src_dir` 에 `r2r` 또는 `r2r_h1_replay` 경로를 지정
- `--scan` / `--ep` 생략 시 전체 처리
- 출력: `traj_data/r2r_debug/<dataset_name>/<scan>/episode_XXXXXX/rgb/frame_XXXX.jpg` 및 `depth/frame_XXXX.png`
