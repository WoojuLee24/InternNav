# Codex 노은역 gs_vlnpe 초기 감사

- 작성자: Codex
- 작성일: 2026-08-13 UTC
- 범위: `scripts/dataset_converters/gs_vlnpe`
- 기준 브랜치: `feature/input_v0.1` (`6ac34a9`)
- 목적: 기존 Claude/박사님 기록과 섞지 않고, 노은역 USDZ 데이터셋 작업의 현재 상태와 다음 수정 원칙을 기록한다.

## 결론

현재 작업은 실패한 상태가 아니라 **Stage 02~04의 실험 산출물이 이미 존재하지만, 검증과 변경 관리가 완료되지 않은 상태**다.

확인된 노은역 결과는 다음과 같다.

| 단계 | 현재 결과 | 해석 |
|---|---:|---|
| Scene meta | `passed: true` | Z-up, meters-per-unit 1, logical MID ROI가 source bounds 안에 있음 |
| ESDF | sanity PASS, 전체 판정 `null` | GT가 없어서 clearance gate를 수행하지 못했으므로 `UNVERIFIED`가 정확한 표현 |
| Random path | 20/20, hard collision 0 | 안전 관련 hard gate는 통과 |
| Random 분포 | `distribution_ok: false` | 기존 Matterport 기준 분포와 다름. 새 역 공간에 같은 기준을 그대로 적용할 근거는 없음 |
| Isaac observation | D455 nominal 출력 약 3.0GB 존재 | 출력 파일은 있으나 run-level 완료 manifest/report가 없어 완전성 판정이 어려움 |

따라서 다음 목표는 기능을 더 추가하는 것이 아니라, **현재 변경을 원본 호환 경로와 노은역 전용 경로로 분리하고 각 단계의 완료 조건을 명시하는 것**이다.

## 작업 트리 현황

커밋 원본 대비 수정된 tracked 파일은 3개다.

| 파일 | diff 규모 | 역할 |
|---|---:|---|
| `02_build_freemap_esdf.py` | +112 / -19 | GT 없는 신규 scene, ROI bounds, `ref_h_b` 지원 |
| `esdf_utils.py` | +49 / -7 | `voxelize_surface(..., bounds=None)` ROI 지원 |
| `04_render_obs_isaac.py` | +391 / -22 | 신규 scene USDZ, D455 nominal camera, depth 저장 설정 지원 |

그 밖에 여러 `.bak_*`, 실험 스크립트, 비교 결과와 생성 데이터가 untracked 상태다. 특히 `obs/`만 약 7.9GB다. 이 감사에서는 사용자 작업을 보존하기 위해 어떤 파일도 삭제하거나 되돌리지 않았다.

## 확인한 좋은 점

1. 기본 옵션은 대체로 기존 동작을 유지하도록 설계되어 있다.
   - `voxelize_surface`의 `bounds` 기본값은 `None`이다.
   - `--camera` 기본값은 `dataset`이다.
   - `build_renderer`의 near/far 인자는 기존 상수를 기본값으로 가진다.

2. 원본 USDZ를 직접 고치지 않는다.
   - logical MID 영역은 `scene_meta`의 ROI로 표현한다.
   - hidden collision mesh visibility는 runtime stage에서만 override한다.

3. 새 scene에 GT가 없다는 사실을 억지로 PASS로 만들지 않았다.
   - ESDF JSON의 전체 `passed`는 `null`이다.
   - 콘솔/리포트 표현은 `UNVERIFIED`를 사용한다.

4. 노은역 random path의 직접 안전 지표는 양호하다.
   - 20/20 생성 성공
   - hard collision 0
   - refined minimum clearance 약 0.30m (`r_b=0.25m`)
   - trajectory minimum clearance 약 0.224m

## 문제와 위험

### P0: 생성 완료를 증명하는 run-level manifest가 없다

`obs/noeun_station_mid_random_isaac_d455_nominal`에는 `camera_config.json`과 episode 파일들이 있지만, 실행 전체의 episode별 frame 수, RGB/depth/extrinsic 개수 일치, 실패 episode, 최종 PASS/FAIL을 담은 기계 판독 가능한 manifest가 없다.

현재 HTML report는 기본적으로 `log_dir`에 저장되므로 산출물 디렉터리만 전달하면 완료 여부가 사라진다. 데이터셋 제작 관점에서는 각 episode의 세 스트림 개수와 파일 형식을 다시 전수 검사해야 한다.

### P0: 실제 D455 calibration이 아니라 nominal 값이다

현재 설정은 480x270, `fx=fy=240`, horizontal FOV 90도, depth scale 1mm/raw다. 코드 주석도 실제 장비 intrinsics로 교체해야 한다고 명시한다.

이 값으로 생성된 데이터는 “D455 실측 데이터”가 아니라 **D455 nominal synthetic camera 데이터**로만 불러야 한다. 실제 학습/실기 센서 정합이 목표라면 RGB intrinsic, 해상도, distortion 처리, RGB-depth alignment, depth invalid/noise 모델의 출처를 먼저 확정해야 한다.

### P1: 변경량이 최소 수정 원칙에 비해 크다

특히 `04_render_obs_isaac.py` 한 파일에 신규 scene 선택, camera registry 역할, depth encoding, collision USDZ visibility, report 변경이 모두 들어갔다. 기존 프로젝트 지침의 “새 기능이 여러 곳에 흩어지면 새 파일로 분리” 원칙과 맞지 않는다.

권장 구조는 기존 `04_render_obs_isaac.py`의 default 경로를 원본에 가깝게 유지하고, 노은역/D455 생성은 별도 entry point 또는 helper 파일에서 기존 renderer의 좁은 API만 재사용하는 방식이다. 단, 현재 결과를 재현할 수 있는 테스트를 먼저 만든 뒤 분리해야 한다.

### P1: ROI voxel sampling은 큰 source mesh에서 표본 밀도가 불안정하다

현재 구현은 source mesh 전체에서 고정 개수의 surface point를 뽑은 뒤 ROI 밖 점을 제거한다. 노은역 source bounds는 약 248m x 116m x 69m이고 MID ROI는 약 67m x 65m x 1.8m다. ROI가 전체 표면의 일부이면 실제 ROI에 남는 sample 수가 실행 파라미터와 source mesh 구성에 크게 좌우된다.

즉 logical ROI 크기가 같아도 source USDZ에 다른 층이나 먼 geometry가 추가되면 MID occupancy가 더 성기게 될 수 있다. 최소한 ROI 안에 남은 sample 수/비율을 metadata에 기록하고, 가능하면 ROI와 교차하는 face를 먼저 고른 뒤 그 면적에 대해 sampling해야 한다.

### P1: 검증 상태의 의미가 파일마다 다르다

- Stage 02: GT가 없으면 `passed: null`이며 `UNVERIFIED`다.
- Stage 03 random: Matterport 참고 분포가 맞지 않아도 안전 hard gate만 통과하면 `passed: true`다.
- Stage 04: report 생성은 있으나 output directory에 완료 manifest가 없다.

각 값은 코드 문맥 안에서는 설명되지만, 최종 dataset root에서는 하나의 “완료” 의미로 합쳐지지 않는다. `geometry_validated`, `planning_safe`, `distribution_reference_match`, `render_complete`, `sensor_calibrated`처럼 독립 상태로 유지해야 한다.

### P1: 대용량 생성물이 Git에서 무시되지 않는다

`.gitignore`에는 `scrips/dataset_converters/gs_vlnpe/logs/`라는 오타가 있다. 실제 `scripts/...` 경로에는 적용되지 않는다. 일반 `logs/` 규칙 때문에 logs는 우연히 무시되지만, `obs/`, `esdf/`, `paths/`, `scene_meta/`, `verify/`, `compare/`, `gridsearch/`는 현재 untracked로 노출된다.

어떤 JSON/NPZ를 재현성 자료로 커밋할지 먼저 결정한 뒤, RGB/depth 같은 대용량 생성물만 정확히 ignore해야 한다. 디렉터리 전체를 성급히 ignore하면 중요한 작은 metadata까지 숨길 수 있다.

### P2: 실험 잔재가 소스와 같은 위치에 있다

`.bak_*` 파일 다수, 빈 루트 파일 `a`, 전체 소스 덤프로 보이는 `gs_vlnpe_latest_2days.txt`, `04_render_obs_blenderproc.py` 등이 untracked다. 삭제 여부는 작성자 확인 전 결정하면 안 된다. 다만 최종 변경 목록에는 포함하지 않는 것이 안전하다.

### P2: 정적 검사 경고 2건

문법 컴파일은 통과했지만 invalid escape sequence `\*` 경고가 다음 docstring에서 발생한다.

- `03_sample_gt_paths.py` 1행 부근
- `esdf_utils.py`의 `voxelize_surface` docstring

실행 실패는 아니며 우선순위는 낮다. 신규 변경을 정리할 때 `A*`처럼 escape가 필요 없는 표기로 바꾸면 된다.

## 이번 감사에서 실행한 검사

- `git status`, target 파일 목록과 Markdown 기록 확인
- `CLAUDE.md`, `.claude/rules/guideline.md`, 관련 gs_vlnpe memory 검토
- 커밋 원본 대비 `git diff`, `git diff --check`
- `/usr/bin/python3 -m py_compile` 정적 문법 검사
- 노은역 `scene_meta`, ESDF JSON, random path JSON 확인
- 생성물 용량과 Git ignore 적용 여부 확인

Isaac Sim 전체 재렌더는 이번 초기 감사에서 실행하지 않았다. 실행 시간이 길고 기존 3GB 산출물을 덮어쓸 수 있으므로, 먼저 별도 출력 디렉터리를 정한 smoke run으로 검증해야 한다.

## 다음 작업 순서

1. **read-only dataset verifier 추가 또는 기존 verifier 재사용**
   - episode별 RGB/depth/extrinsic frame count 일치
   - intrinsic shape/value, RGB/depth resolution과 dtype
   - depth 유효 비율/거리 분포
   - trajectory JSON frame 예상치와 실제 frame 수
   - 결과를 별도 Codex Markdown과 JSON manifest로 저장

2. **원본 default 경로 회귀 검사**
   - 수정 전 커밋과 현재 코드에 동일한 기존 scene/argument를 넣는다.
   - 출력 metadata와 핵심 수치가 같은지 비교한다.
   - GPU 전체 렌더 전에는 argument parsing과 pure helper 단위 검사를 먼저 한다.

3. **노은역 smoke run을 새 출력 디렉터리에 1 episode만 수행**
   - 기존 산출물을 덮어쓰지 않는다.
   - collision visibility override 전후 depth finite fraction을 기록한다.
   - RGB/depth/extrinsic/K를 시각화와 수치로 함께 검증한다.

4. **검증이 고정된 뒤 코드 분리**
   - `04_render_obs_isaac.py`의 기존 경로를 원본에 최대한 가깝게 복원한다.
   - 노은역 scene/camera/depth 규약은 별도 helper 또는 entry point로 옮긴다.
   - Stage 02 ROI sampling도 별도 함수로 격리하고 sample coverage를 기록한다.

5. **실제 센서 규약 확정**
   - 실제 D455 calibration 파일 또는 장비 출력 확보
   - 목표 학습 데이터 schema의 RGB/depth 해상도와 단위 확정
   - synthetic clean depth만 쓸지, noise/invalid 모델까지 적용할지 결정

## 변경 원칙

- 기존 생성물과 `.bak_*`는 명시적 승인 없이 삭제하지 않는다.
- 기존 default argument 경로의 함수 흐름을 먼저 회귀 검증한다.
- 노은역 전용 기능은 기존 함수 본문 중간에 계속 분기를 추가하지 않는다.
- “PASS”는 한 값으로 뭉개지 않고 geometry/planning/render/calibration 상태를 분리한다.
- 이후 Codex 분석과 결과도 파일명이 `codex_`로 시작하는 `.claude/memory/*.md`에만 기록한다.

## 2026-08-13 구현 결과

사용자가 전체 진행을 승인한 뒤 다음을 구현했다. Conda는 사용하지 않았다.

- `01_prepare_usdz_scene.py` 추가: GT 없는 USDZ logical level을 기존 Matterport Stage 01과 분리했다.
  노은역 `floor_z=-0.05` 주변의 위쪽 수평면 14,618개, 798.19 m²를 근거로 MID XY bounds를
  재구성하며 source USDZ는 수정하지 않는다.
- `camera_profiles.py` 추가: D455 nominal profile을 기존 dataset GT camera에서 분리했다.
  `0.001 m/raw`는 D400 기본 Z16 unit이며 실제 장비에서는 `get_depth_scale()`로 재확인해야 한다.
  10 m는 synthetic 저장 cutoff이고 D455가 10 m에서 정확하다는 뜻이 아니다.
- `usdz_scene_utils.py` 추가: new-scene metadata 로드와 hidden collision mesh runtime visibility만
  담당한다. Matterport asset에는 적용되지 않는다.
- `04_render_obs_isaac.py`는 위 두 모듈에 위임하도록 정리했다. 기본 `--camera dataset` 경로는
  기존 N1의 K, `0.0001 m/raw`, 3 m cutoff를 그대로 쓴다.
- ROI voxelization은 전체 source에서 고정 수를 뽑은 뒤 자르지 않고 ROI와 교차하는 face를 먼저
  선택해 sampling한다. 큰 다층 source 크기에 따라 MID sample density가 바뀌는 문제를 제거했다.

검증 결과:

- 전체 Python 파일 compile PASS, `git diff --check` PASS
- full-scene/ROI voxel helper 검사 PASS
- D455 profile K와 uint16 표현 범위 검사 PASS
- USDZ 준비 단계는 `/tmp/internnav_noeun_verify`에서 실행 PASS
- floor z와 floor XY bounds는 기존 노은역 metadata와 정확히 일치
- 기존 20개 observation 전수 검사 PASS: RGB/depth/extrinsic count 일치, K 일치,
  depth uint16 270x480, 유효 비율 84.4~99.9%, 최대 10.000 m
- 기존 3 GB observation과 기존 scene metadata/ESDF/path는 덮어쓰지 않았다.

실제 D455 장비가 준비되면 반드시 장비별 `get_intrinsics()`와 `get_depth_scale()` 값을 새
calibrated profile로 추가해야 한다. 현재 profile 이름에 `nominal`을 유지한 이유다.
