# Codex 포맷 검증 02 — 노은역 04 Isaac vs VLN-PE/VLN-CE 코드 전수 검사

작성일: 2026-08-14 (UTC)

## 요청과 검사 원칙

노은역 중간층에서 Isaac Sim 04를 실행해 저장한 20개 episode가 VLN-PE 및 VLN-CE와 같은
포맷인지 추측이 아니라 코드로 검증했다. `status=PASS`와 `format same`을 분리했다.

- validation PASS: 세 입력을 정상 파싱했고 계약 검사가 완결됨
- exact format same: 파일 layout/dtype/shape/pose/metadata가 실제로 동일한지
- direct loader compatible: 기존 InternNav/Habitat loader가 변환 없이 읽을 수 있는지

## 구현 파일

- `scripts/dataset_converters/gs_vlnpe/format_validation/validate_noeun_vln_formats.py`
- `scripts/dataset_converters/gs_vlnpe/format_validation/generate_format_report.py`
- `scripts/dataset_converters/gs_vlnpe/format_validation/README.md`

## 실제 비교 입력

### 노은역

`scripts/dataset_converters/gs_vlnpe/noeun_pipeline_codex/obs/noeun_station_mid_random_isaac_d455_nominal`

20개 episode의 모든 JPG, PNG, pose NPY를 열어 검사했다.

### VLN-PE

`data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r/s8pcmisQ38h`

실제 episode 0의 parquet, RGB NPY, depth NPY와 `internnav/utils/loader.py`가 요구하는 column을
검사했다.

### VLN-CE

- sensor config: `scripts/eval/configs/vln_r2r_mini.yaml`
- episode JSON: `data/vlnverse_emr/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz`
- 변환 코드 참고: `scripts/dataset_converters/vlnce2lerobot.py`

## 실행 명령

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/format_validation/validate_noeun_vln_formats.py
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/format_validation/generate_format_report.py
```

Conda/Mamba는 사용하지 않았다.

## 전수 무결성 결과

- 노은역 episode: 20
- RGB: 8,378
- depth: 8,378
- pose: 8,378
- 모든 episode에서 stream count 일치
- `frame_0000`부터 연속 번호 확인
- RGB: 480×270 RGB JPG
- depth: 270×480 uint16 PNG
- pose: finite 4×4 float matrix, homogeneous 마지막 행 확인
- intrinsic: episode마다 3×3 float NPY
- 노은역 integrity: PASS
- 실제 VLN-PE sample integrity: PASS
- VLN-CE JSON/config contract: PASS

## 최종 호환성 판정

### 노은역 ↔ VLN-PE

RGB, metric optical depth, per-frame camera pose, 10 m cutoff라는 의미는 대응한다. 하지만 정확한
파일 포맷은 동일하지 않다.

| 항목 | 노은역 | VLN-PE |
|---|---|---|
| RGB | frame JPG, 480×270 uint8 | episode NPY, T×256×256×3 uint8 |
| depth | frame PNG uint16, raw×0.001m | episode NPY float32, value×10m |
| pose | frame 4×4 NPY | parquet position[3]+wxyz quaternion[4] |
| intrinsic | episode 3×3 NPY + config | parquet에 없음, 256²/90° implicit K |
| episode table | 없음 | parquet |
| instruction/action | 없음 | parquet/tasks/meta |
| robot/progress/step | 없음 | loader 필수 column |

판정:

- semantic correspondence: YES
- exact storage format same: NO
- existing VLN-PE loader direct compatibility: NO

### 노은역 ↔ VLN-CE

VLN-CE는 같은 종류의 offline RGB-D trajectory 폴더가 아니다. gzip episode JSON과 MP3D
GLB/navmesh를 Habitat가 읽고 640×480, HFOV 79°, max depth 10 m 관측을 실행 중 생성한다.

판정:

- sensor semantic correspondence: YES
- exact offline storage format same: NO
- Habitat VLN-CE direct dataset compatibility: NO

## 노은역 데이터의 현재 정확한 성격

노은역 결과는 완전하고 정합된 **synthetic sensor trajectory**다. RGB, depth, intrinsic, pose는
있지만 language-conditioned VLN episode를 완성하는 instruction, action, robot state, progress,
step, task/finish metadata는 없다. 따라서 “VLN-PE와 같은 데이터”가 아니라 “VLN-PE식 학습
episode로 변환할 수 있는 핵심 센서 관측과 pose가 준비된 상태”라고 표현해야 정확하다.

## 변환에 필요한 작업

VLN-PE loader 호환 포맷을 만들려면 다음이 필요하다.

1. frame JPG/PNG를 episode-level RGB/depth NPY stack으로 변환
2. 4×4 pose를 camera position과 wxyz quaternion으로 변환
3. depth를 normalized float32 규약으로 변환하거나 loader를 명시적으로 확장
4. parquet과 tasks/episode metadata 생성
5. robot pose, action, progress, step, timestamp, language instruction 정의

VLN-CE에 사용하려면 노은역 USDZ를 Habitat-compatible scene/navmesh로 준비하고 R2RVLN episode
JSON의 scene/start/goal/instruction/action 계약을 만들어야 한다. 또는 Habitat와 별도의 offline
replay adapter를 구현해야 한다.

## 결과 파일

- machine-readable JSON: `scripts/dataset_converters/gs_vlnpe/format_validation/noeun_vln_format_validation.json`
- visual report: `scripts/dataset_converters/gs_vlnpe/format_validation/report.html`
- report image assets: `scripts/dataset_converters/gs_vlnpe/format_validation/report_assets/`

HTML은 노은역/VLN-PE 실제 RGB, metric으로 decode한 depth 비교, 20개 episode frame count,
세 환경 해상도, 계약 비교표, 필요한 변환 목록을 포함한다.

## LiDAR 상태

LiDAR는 노은역 episode 0 frame 0 smoke test와 geometry 검증만 PASS한 상태로 hold한다. 이번 포맷
비교는 LiDAR를 VLN-PE/VLN-CE 기본 필드로 간주하지 않는다. 두 원본 포맷 모두 기본 LiDAR
field가 없으므로 향후 LiDAR는 별도 공통 확장 schema가 필요하다.
