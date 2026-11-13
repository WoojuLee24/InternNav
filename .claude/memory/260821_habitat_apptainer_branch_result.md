# apptainer 분기 + habitat 설치 스크립트 — 결과

_2026-08-21_

## 왜

val_unseen 프레임 렌더링에 habitat-sim 이 필요한데, 현재 `internnav` 인스턴스에는
Isaac Sim 5.1 + torch 2.7.0+cu128 + InternNav(editable) 가 이미 들어 있다.
→ **overlay 만 새로 분기**해서 habitat 을 거기에만 설치한다. sandbox base(37G, isaac-sim 19G)는 공유.

## 환경 구조 (실측)

```
~/workspace/sandbox/internnav              sandbox base 37G  (컨테이너 /,  isaac-sim/ 19G)
~/workspace/sandbox/internnav_260723.img   기존 overlay 10G  ← 기본 internnav 인스턴스
~/workspace/sandbox/internnav_habitat_260820.img  신규 overlay 20G ← habitat 분기 (스크립트가 생성)
~/workspace                                컨테이너 /ws (bind, 3.3T)
```

- `/workspace/isaaclab/_isaac_sim` 는 `/isaac-sim` 으로의 symlink. torch 는 site-packages 가 아니라
  `/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch` (base) 에 있다.
- pip 설치 대상은 `/isaac-sim/kit/python/lib/python3.11/site-packages` → **overlay 에 기록**.
  그래서 overlay 를 갈아끼우면 base 공유 + 패키지 집합 분리가 된다. 37G 복사 불필요.

## 만든 파일 3개 (기존 파일 무수정)

| 파일 | 역할 |
|---|---|
| `docker/start-apptainer-habitat-instance.sh` | 호스트 실행. overlay 생성 → 기존 `start-apptainer-instance.sh` 에 `NAV_OVERLAY` 로 위임 |
| `docker/install_habitat_6000pro.sh` | 분기 인스턴스 안에서 실행. `install_habitat_5090.sh` 를 이 환경에 맞게 수정 |
| `docker/smoke_habitat_render.py` | import 만으로는 부족 — 실제 mp3d scene 을 열어 RGB 1장 렌더/저장 |

### 5090 스크립트가 이 환경에서 깨지던 지점

| # | 문제 | 수정 |
|---|---|---|
| 1 | `python` 이 PATH 에 **아예 없음** → 검증 전부 실패 | `PY=/workspace/isaaclab/_isaac_sim/python.sh` |
| 2 | `pip` → `/usr/bin/pip` (system py3.12, torch·internnav 없음) → **엉뚱한 인터프리터에 설치** | `PIP="$PY -m pip"` |
| 3 | `INTERNNAV_DIR=$BASE_DIR/InternNav` → `/ws/src/InternNav/InternNav` 를 찾다 exit 1 | `BASE_DIR=/ws/src` |
| 4 | `CMAKE_CUDA_ARCHITECTURES=90` (GPU 불일치) | torch 로 자동 감지 → **120** (Blackwell sm_120) |
| 5 | `[0/3]` 데이터 링크가 무조건 exit 1 (`/home/irteam/data-vol1` 없음) | 이미 있으면 skip 하는 guard |
| 6 | `pip install . 2>/dev/null` 로 빌드 에러를 삼킴 | `build_install.log` 로 tee |

추가: `PATH` 에 `.../kit/python/bin` prepend (ninja 가 거기에만 있음), `MAX_JOBS=16`.
**모든 pip 설치의 `--no-deps` 는 유지** — `habitat_requirements.txt` 가 accelerate==1.4.0 /
flash_attn==2.7.4.post1 / diffusers==0.32.2 를 핀하는데 현재는 1.13.0 / 2.8.3+cu128torch2.7 이다.

### 오설치 방지 guard

`NAV_BRANCH=habitat`(래퍼가 `APPTAINERENV_` 로 주입) 또는 `TMUX_TMPDIR=/ws/.tmux/instance-*habitat*`
둘 중 하나라도 있어야 진행. 둘 다 없으면 중단(`FORCE=1` 로 우회 가능).

## 검증 (여기서 가능한 것까지)

- 세 파일 `bash -n` / `py_compile` 통과
- `apptainer` stub 으로 조립 명령 확인 → `--overlay .../internnav_habitat_260820.img`,
  인스턴스명 `internnav-habitat`, `CUDA_VISIBLE_DEVICES=0`, `TMUX_TMPDIR=/ws/.tmux/instance-internnav-habitat`
  (기본 overlay `internnav_260723.img` 를 쓰지 않음을 확인)
- **guard 실동작 확인**: 현재 internnav 인스턴스에서 실행 → `감지된 인스턴스 = 'internnav'` 로 exit 1
- `CUDA_ARCH` 자동감지 → `120`, `$PY -m pip` → python 3.11 의 pip 24.3.1, ninja 경로 해결 확인
- **설치 전 baseline** `logs/260820_baseline_before_habitat.txt`:
  torch 2.7.0+cu128 / numpy 1.26.4 / accelerate 1.13.0 / transformers 4.51.0
- **미검증**: 실제 `apptainer instance start` 와 habitat-sim 빌드 (컨테이너 안에 apptainer CLI 없음)

## 실행 순서

1. 호스트: `bash ~/workspace/src/InternNav/docker/start-apptainer-habitat-instance.sh`
2. 호스트: `dexec internnav-habitat`
3. 그 안: `bash docker/install_habitat_6000pro.sh 2>&1 | tee logs/260821_install_habitat_6000pro.log`
4. 그 안: `/workspace/isaaclab/_isaac_sim/python.sh docker/smoke_habitat_render.py`
5. **원래 internnav 인스턴스**에서 오염 확인: `import habitat_sim` 이 **실패해야 정상**,
   torch/numpy/accelerate/transformers 버전이 baseline 과 동일해야 함

`dstart internnav-habitat` 은 안 된다 — VLN-Challenge alias 의 dispatch 가 `internnav|nav*` 라 거부됨.
기동은 위 래퍼로, 셸 열기는 `dexec`(이름만 봄)로.

## 리스크

- **habitat-sim v0.3.3 × python 3.11**: 공식 지원은 3.9 계열. 빌드 실패 가능성이 남아 있고
  fallback(`setup.py install --headless --with-bullet`)도 안 되면 패치 필요. 4번(실제 렌더)까지 통과해야 성공.
- `/ws` 는 두 분기가 공유 → `habitat-sim`/`habitat-lab` 소스와 `internnav.egg-info` 는 원래 인스턴스에도
  파일로 보인다. site-packages 등록은 overlay 별이라 `import habitat_sim` 은 여전히 실패(= 정상).
