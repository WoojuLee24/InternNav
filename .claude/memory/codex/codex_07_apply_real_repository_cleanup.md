# 노은역 `apply_real` 소스 정리 기록

작성일: 2026-08-14

## 1. 정리 목적

기존 `scripts/dataset_converters/gs_vlnpe/`의 원본 단계 파일에 노은역 지원 코드와 실행
산출물이 섞여 있었다. 향후 `feature/real_dg` 브랜치에는 다음 경계가 명확해야 한다.

- 원본 단계 스크립트: Git 원본 상태
- 노은역 적용 단계: `apply_real/`
- 공통 helper 개선: 바깥에서도 유지
- 생성 dataset, report, 이미지, NPZ/JSON: Git 제외
- Codex 작성 문서: `.claude/memory/codex/`

## 2. 보존과 복원 절차

수정본 손실을 막기 위해 다음 순서로 작업했다.

1. 수정된 `00~04`와 필수 helper를 `apply_real/`에 복사했다.
2. 원본 위치와 복사본의 SHA-256을 파일별로 비교했다.
3. 16개 대상 파일이 모두 바이트 단위로 동일함을 확인했다.
4. 그 뒤에만 바깥의 추적 대상 `00~04`를 `git checkout -- <명시적 파일 목록>`으로 복원했다.
5. 공통 helper는 사용자 요청에 따라 바깥 수정본도 유지했다.

복원한 단계 파일:

- `00_inspect_vln_n1.py`
- `01_prepare_scene.py`
- `02_build_freemap_esdf.py`
- `03_sample_gt_paths.py`
- `03b_verify_reproduction.py`
- `03c_compare_refine.py`
- `03d_grid_search_params.py`
- `03e_compare_reference_path.py`
- `04_render_obs.py`
- `04_render_obs_isaac.py`

바깥에서도 유지한 공통 변경:

- `esdf_utils.py`
- `camera_profiles.py`
- `usdz_scene_utils.py`

## 3. 새 소스 구조

```text
scripts/dataset_converters/gs_vlnpe/apply_real/
├── README.md
├── 00_inspect_vln_n1.py
├── 01_prepare_scene.py
├── 02_build_freemap_esdf.py
├── 03_sample_gt_paths.py
├── 03b_verify_reproduction.py
├── 03c_compare_refine.py
├── 03d_grid_search_params.py
├── 03e_compare_reference_path.py
├── 04_render_obs.py
├── 04_render_obs_isaac.py
├── camera_profiles.py
├── dataset_utils.py
├── esdf_utils.py
├── geometry_utils.py
├── usdz_scene_utils.py
├── viz_utils.py
├── lidar/
│   ├── 05_render_lidar_isaac.py
│   ├── validate_lidar_dataset.py
│   ├── generate_lidar_report.py
│   └── generate_lidar_multiframe_report.py
└── format_validation/
    ├── validate_noeun_vln_formats.py
    └── generate_format_report.py
```

helper를 함께 복사한 이유는 단계 스크립트가 실행 위치의 Python 모듈을 직접 import하기
때문이다. 바깥 helper에 우연히 의존하면 향후 원본 변경에 따라 `apply_real` 재현성이 깨진다.

## 4. 경로 계약 변경

복사 직후 코드는 과거 `gs_vlnpe/` 또는 `noeun_pipeline_codex/`를 기본 출력으로 가리켰다.
이를 다음처럼 바꿨다.

- canonical output root: `scripts/dataset_converters/gs_vlnpe/apply_real`
- log root: `logs/gs-vlnpe/apply_real`
- 05 input observation: `apply_real/obs/noeun_station_mid_random_isaac_d455_nominal`
- 05 path JSON: `apply_real/paths/noeun_station_mid_random.json`
- format validation output: `apply_real/format_validation/`

폴더가 한 단계 깊어졌기 때문에 다음 repository-root 계산도 함께 수정했다.

- `dataset_utils.py`: `parents[3]` → `parents[4]`
- `lidar/05_render_lidar_isaac.py`: `parents[4]` → `parents[5]`

05는 이제 `apply_real`의 `geometry_utils.py`와 `usdz_scene_utils.py`를 import한다.

## 5. Git 제외 정책

`.gitignore`에 `apply_real`의 다음 생성물을 명시적으로 추가했다.

- `target_schema.json`
- `scene_meta/`, `esdf/`, `paths/`, `obs/`
- `verify/`, `compare/`, `gridsearch/`
- format validation의 JSON, HTML, report asset

기존에 생성된 과거 산출물 경로도 함께 제외했다. Python 소스와 README는 제외하지 않았다.
전역 `*.json` 또는 `*.html` 규칙을 쓰지 않은 이유는 저장소의 정상 소스·설정 파일까지 숨길
수 있기 때문이다.

## 6. 문서 분리

`.claude/memory/` 바로 아래에 있던 `codex*.md` 21개를 다음 위치로 이동했다.

```text
.claude/memory/codex/
```

Claude/박사님 문서와 Codex 실행·분석 기록이 파일명뿐 아니라 디렉터리 수준에서도 분리된다.

## 7. 검증 결과

다음 검사를 실행했다.

```bash
/workspace/isaaclab/_isaac_sim/python.sh -m py_compile \
  scripts/dataset_converters/gs_vlnpe/apply_real/*.py \
  scripts/dataset_converters/gs_vlnpe/apply_real/lidar/*.py \
  scripts/dataset_converters/gs_vlnpe/apply_real/format_validation/*.py

git diff --check
```

결과:

- 모든 Python 파일 문법 검사 통과
- whitespace 오류 없음
- 바깥 `00~04`는 Git 수정 목록에서 제거됨
- 생성된 `__pycache__`는 기존 `.gitignore` 규칙으로 제외됨

## 8. 현재 의도적으로 손대지 않은 파일

처음에는 과거 `.bak_*` 파일을 보존했으나 사용자가 직접 만든 불필요한 임시 백업이며 push할
필요가 없다고 확인했다. 최종 수정본이 `apply_real`에 보존됐음을 다시 확인한 뒤 비추적
`.bak_*` 9개를 삭제했고, 재발 방지를 위해 `.gitignore`에 `*.bak_*`를 추가했다.

후속 내용 검사로 남아 있던 untracked 항목을 다음처럼 판정했다.

- root `a`: 내용이 없는 임시파일
- `gs_vlnpe_latest_2days.txt`: 과거 파일 내용을 모아 둔 조사용 code dump
- `noeun_debug/target_schema.json`: 실행 생성물
- 바깥 `lidar/`: `apply_real/lidar/`에 소스가 보존된 중복본
- 바깥 `format_validation/`: `apply_real/format_validation/`에 소스가 보존됐고 HTML·PNG·JSON
  생성물까지 섞인 중복 폴더
- `04_render_obs_blenderproc.py`: 박사님 `feature/input_v0.1`에는 없고 노은역 00→05에 사용하지
  않은 Matterport3D 1-frame 실험용 smoke-test
- `apply_real/**/__pycache__`: 문법 검사 부산물

위 항목은 최종 push 대상이 아니므로 삭제했다. `04_render_obs_blenderproc.py`는 실행 결과나
노은역 파이프라인의 입력 의존성이 없음을 header와 기본 경로로 확인한 뒤 제거했다.

정리 후 바깥에 남긴 새 공통 소스는 사용자 요청에 따른 `camera_profiles.py`,
`usdz_scene_utils.py`이며, 기존 추적 파일 중 공통 기능 변경은 `esdf_utils.py`뿐이다.

## 9. 실행 안내

노은역 `00→05`의 실제 명령과 각 단계의 역할은 다음 파일을 기준으로 한다.

```text
scripts/dataset_converters/gs_vlnpe/apply_real/README.md
```

05는 현재 OS1-32 smoke test 및 3-frame 검증까지 완료된 상태다. 전체 20 episodes batch가
완료된 것처럼 표시하지 않았다.

## 10. LiDAR 문서 단일화

초기 `Example_Rotary` smoke-test 중심의 완전 재현 가이드와 이후 작성한 OS1-32/robot-frame/
RGB-D calibration 문서가 별도로 존재해 실행법과 수치가 중복됐다. 후자가 더 최신 구현과
16-gate, 3-frame 결과를 반영하므로 다음 방식으로 통합했다.

- canonical 상세 문서:
  `.claude/memory/codex/codex_noeun_rtx_lidar_complete_reproduction_guide.md`
- 제거한 중복 문서:
  `codex_lidar_06_os1_robot_frame_and_calibration_plan.md`
- 최신 OS1 문서를 본문으로 사용
- 기존 문서에만 있던 LiDAR/pointcloud 구분, VLN-CE/VLN-PE 조사 한계, 초기 44,784-point
  smoke 결과, multi-GPU semaphore timeout, hidden collision mesh 문제 해결 이력을 앞부분에 통합
- 모든 실행·산출물 경로를 `apply_real` 기준으로 교정

박사님이 빠르게 재실행할 때는 `apply_real/lidar/README.md`만 보면 된다. README에는 현재
확정 contract, 한 frame 생성→독립 검증→HTML, 3-frame 생성·검증·통합 report의 복사 가능한
명령을 둔다. 설계 근거와 전체 수치를 README에 복제하지 않고 canonical 상세 문서 하나로
연결해 두 문서가 향후 서로 달라지는 것을 방지한다.

## 11. 00~04 README 재실행 안내 강화

LiDAR와 같은 원칙으로 `apply_real/README.md`는 박사님용 빠른 실행서, 다음 파일은 상세 근거
문서로 역할을 나눴다.

```text
.claude/memory/codex/codex_noeun_usdz_mid_00_to_04_complete_reproduction_guide.md
```

README에는 다음을 추가했다.

- 노은역 중간층이 cropped USDZ가 아닌 logical scene이라는 설명
- 실행 전 변수와 USDZ 확인
- 00→01→02→03→04 입력·출력 의존 관계
- 단계별 복사 가능한 실행 명령
- canonical JSON/NPZ/HTML 위치
- 당시 실제 PASS 기준과 핵심 수치
- 03b~03e 진단 결과의 의미
- D435i 0.0001과 D455 nominal 0.001 depth scale 비교
- Open3D 04는 비교용이고 Isaac 04가 기준 dataset이라는 구분
- 05의 중복 명령을 제거하고 `lidar/README.md`로 연결

상세 00~04 문서에 남아 있던 `noeun_pipeline_codex`, 바깥 단계 스크립트 및
`logs/gs-vlnpe/codex_noeun` 경로도 모두 현재 `apply_real` 코드·출력·로그 경로로 교정했다.
