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

### 4. 확실하지 않으면 질문
- 확실하지 않은 상태에서 개발하지 말고 user에게 질문해서 구체화한 후 개발할 것

### 5. Training과 evaluation은 일치해야 됨
- 입력, 출력 데이터 format이 일치해야 됨
- **train, eval(habitat), eval(isaac) 세 경로 모두 동일한 조건·파라미터로 모델에 동일한 입력 형식이 들어가야 한다.** 한쪽만 바꾸고 다른 쪽을 누락하면 train==eval 불변식이 깨짐.
- 예외적인 경우: 

### 5. 검증

- 변경 후에는 반드시 명령어를 실행해 **기존 기능(default 경로)이 이전과 동일하게 동작하는지**, **신규 기능이 의도대로 동작하는지** 둘 다 확인한다.
- 기존 테스트가 있다면 새 코드 작성 전/후 모두 통과하는지 확인한다.
- 개발 목표와 관련된 코드는 시각화해서 저장해서 디버깅해라. 

### 6. 분기(if/elif) 작성: 값이 유한하면 exhaustive if/elif/else + assert로 명시 (exhaustiveness check / assert_never 패턴)

- 변수가 가질 수 있는 값이 이미 검증되어 있다면, 남은 경우를 fallthrough/암묵적 else로 처리하지 말고 **모든 값을 `if/elif/.../else`로 나열**하고 마지막 `else`는 `assert False, f"unreachable {var}={var!r}"`로 막는다.
- 이유: 가독성. 어떤 값이 어떤 분기로 가는지 코드만 보고 바로 알 수 있어야 한다.
```python
if self.s1_combine == 'replace':
    ...
elif self.s1_combine == 'concat':
    ...
else:
    assert False, f"unreachable s1_combine={self.s1_combine!r}"
```

### 7. 상속을 코드 재사용 목적으로 썼다면, `isinstance`로 동작을 게이팅하지 않는다

- 클래스 A가 클래스 B를 상속하는 이유가 "B의 메서드를 `super()`로 재사용하기 위해서"일 뿐, A가 B의 모든 인스턴스를 의미론적으로 대표하지 않는다면(A가 B보다 더 넓은 범위를 표현한다면), 다른 코드에서 `isinstance(obj, B)`로 "이 객체가 B의 동작을 해야 하는가"를 판단하면 안 된다. A의 인스턴스는 상속 관계 때문에 `isinstance(obj, B)`가 **항상 True**가 되어버려서, 실제로 B의 동작(예: B 전용 디버그 저장)이 필요 없는 A 인스턴스에도 그 동작이 걸린다.
  - 실제로 겪은 사례: `UnifiedImageProvider`가 코드 재사용을 위해 `BEVImageProvider`를 상속했는데, `UnifiedImageProvider`는 BEV를 전혀 안 쓰는 조합(`s1_view='fpv'`)도 표현한다. `habitat_vln_evaluator_unified.py`가 "BEV 디버그 이미지를 저장할지"를 `isinstance(self.provider, BEVImageProvider)`로 판단했는데, 이 provider는 항상 `UnifiedImageProvider`라서 이 체크는 상수 True였고, `s1_view='fpv'`(디버그 스텝 카운터가 증가 안 하는 경로)에서도 디버그 저장이 호출되어 매 스텝 같은 파일명으로 덮어써지는 버그가 있었다.
  - 해결: `isinstance(obj, B)` 대신 **실제 동작을 결정하는 config/상태 값**(예: `obj.s1_view == 'bev'`)으로 직접 체크한다. 이 패턴은 이미 `internvla_n1_unified_provider.py`(학습 코드)와 `internnav/dataset/internvla_n1_lerobot_dataset.py`(데이터셋 코드)에 정확하게 적용되어 있었음 — evaluator만 예외적으로 틀린 패턴을 쓰고 있었다.
  - 더 근본적인 해결: 애초에 A가 B보다 의미론적으로 넓은 범위를 표현한다면(A의 일부 인스턴스만 B 역할을 함), A는 B를 상속하지 말고 필요한 로직만 A 안에 직접 작성하거나 B의 하위 로직을 호출하는 방식으로 가져온다. "코드 재사용" 하나만으로 상속 여부를 정하지 말 것 — `isinstance`/타입 계층이 실제 동작을 정확히 반영하는지도 같이 판단해야 한다.


### 8. 코드 주석은 최대 2줄

- Claude가 추가/수정하는 코드의 주석(블록·인라인 모두)은 **최대 2줄**로 제한한다.
- 배경 설명·근거·실험 결과 등 긴 서사는 코드 주석이 아니라 `.claude/memory/` 문서나 커밋 메시지에 쓴다.
- 주석은 코드만 봐서는 알 수 없는 제약/이유 하나만 압축해서 적는다.

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
  - 진행 중인 태스크 추적 파일(auto-dev용, `<branch-name>.md`) → `.claude/tasks/`
  - 완료된 결과 보고, 코드 이해/분석 문서 등 Claude가 작성하는 기록 전반 → `.claude/memory/` (`understanding_*.md`, `<날짜>_..._result.md` 등)
- auto-memory(`/root/.claude/projects/...`)에만 쓰지 말고 항상 repo `.claude/` 쪽도 동기화
- `.claude/memory/`는 Claude가 작성/관리하는 메모리. 사용자가 직접 쓰는 메모리는 `.claude/memory_user/`에 대응되게 둔다 — 세션 시작 시 참고할 컨텍스트가 필요하면 이 폴더도 확인한다.

## 결과 보고 및 정리: 명료하고 짧게 작성
- 이전과 비교해서 어떤 점이 달라졌는지 
- command 명령을 작성하고 각 argument의 의미
- 시각화 및 디버깅 결과 보고
- 결과 보고는 `.claude/memory/{task명}_result.md`에 작성해라.

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
