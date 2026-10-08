# 노은역 USDZ 파이프라인 실행 결과 (apply_real → feature/embaug_v0.1)

작성일: 2026-08-20 / 브랜치: `feature/embaug_v0.1`

## 무엇을 했나

`origin/feature/real_dg`의 커밋 `606b7dc "feat: add Noeun real-data generation pipeline"`을
현재 브랜치로 cherry-pick하고, `noeun_station_collision.usdz`로 00→04를 재실행해
**기준선 재현을 확인**했다. 정합(중복 제거)은 하지 않았다 — 별도 작업.

## 이전과 달라진 점

- **도입 전**: 이 브랜치에 노은역 관련 코드가 전무. `gs_vlnpe`는 `vln_n1`/`vln_pe`만 지원.
- **도입 후**: `scripts/dataset_converters/gs_vlnpe/apply_real/` (00~04 + 03b~03e + lidar +
  format_validation) + 공용 파일 3개 (`esdf_utils.voxelize_surface(bounds=)` 추가,
  `camera_profiles.py`·`usdz_scene_utils.py` 신규).
- 기존 경로 무영향 확인 (3가지 전부 통과):
  - `dataset_utils.py` 자체 회귀 → `ALL_CHECKS_PASSED`
  - `esdf_utils.py` self-check 통과 (`voxelize_surface`의 `bounds=None`이 기존 동작)
  - **vln_n1 씬1 `04 --mode gt_replay` → PASS, depth median 0.0025 m / p95 0.0176 m,
    rgb SSIM median 0.876 / min 0.756** — `reports.md` 기록값과 정확히 동일

## 실행 명령 (전부 한 줄)

공통: `ISAAC=/workspace/isaaclab/_isaac_sim/python.sh`, `PIPE=scripts/dataset_converters/gs_vlnpe/apply_real`,
`LOG=logs/gs-vlnpe/apply_real`

```
timeout --signal=KILL 1800 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/00_inspect_vln_n1.py --scene noeun_station_mid --usd_path data/GS_USDZ/Subway/noeun_station_collision.usdz --floor_z -0.05 --out_dir scripts/dataset_converters/gs_vlnpe/apply_real --log_dir logs/gs-vlnpe/apply_real
```
```
timeout --signal=KILL 1800 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/01_prepare_scene.py --scene noeun_station_mid --usd_path data/GS_USDZ/Subway/noeun_station_collision.usdz --floor_z -0.05 --out_dir scripts/dataset_converters/gs_vlnpe/apply_real --log_dir logs/gs-vlnpe/apply_real
```
```
timeout --signal=KILL 3600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/02_build_freemap_esdf.py --scene noeun_station_mid --geometry usd --ref_h_b 0.875 --scene_meta_dir scripts/dataset_converters/gs_vlnpe/apply_real --out_dir scripts/dataset_converters/gs_vlnpe/apply_real --log_dir logs/gs-vlnpe/apply_real
```
```
timeout --signal=KILL 3600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/03_sample_gt_paths.py --scene noeun_station_mid --mode random --num_episodes 20 --esdf_dir scripts/dataset_converters/gs_vlnpe/apply_real/esdf --out_dir scripts/dataset_converters/gs_vlnpe/apply_real --log_dir logs/gs-vlnpe/apply_real
```
```
timeout --signal=KILL 14400 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/04_render_obs_isaac.py --scene noeun_station_mid --mode random --num_episodes 20 --camera d455_nominal --out_dir scripts/dataset_converters/gs_vlnpe/apply_real --log_dir logs/gs-vlnpe/apply_real
```

### 인자 의미 (기준 문서와 달라진 것만)

| 인자 | 값 | 의미 |
|---|---|---|
| `--usd_path` | `data/GS_USDZ/Subway/noeun_station_collision.usdz` | **가이드 문서는 `data/noeun_station/...`** 지만 이 브랜치의 실제 경로는 여기. sha256 `3a4719b1...4fcd1`로 동일 파일 확인. default가 `None`이라 코드 수정 불필요 |
| `--floor_z -0.05` | 중간층 기준 바닥(m) | **수동 인자**. 01은 자동 검출하지 않고 이 높이 주변 상향 수평면 존재만 검증 |
| `--geometry usd` | occupancy 소스 | 노은역은 obj가 없어 usd(=usdz) 필수 |
| `--ref_h_b 0.875` | 참조 로봇 키(m) | GT episode가 없어 `--ref_episode`로 못 얻으므로 직접 지정 |
| `--camera d455_nominal` | 관측 카메라 | default `dataset`은 GT parquet 카메라 → new_scene에선 에러. 반드시 지정 |

## 기준선 대비 결과 — 전 단계 일치

| 단계 | 기준선 | 이번 실행 | 판정 |
|---|---|---|---|
| 00 | 6 gate PASS, triangle 665,812, 후보 face 14,618 | 동일 (`passed:true`) | ✅ |
| 01 | face 14,618, area 798.1859 m², ROI ⊂ source | area **798.1859255685356**, ROI `[-33.220,-25.825,-0.25]~[33.825,39.371,1.55]` | ✅ |
| 02 | grid 1345×1308×40, occ 0.01013, nav 0.8796, maxESDF 27.6688 | occ **0.010131461523595148**, nav **0.8796391664677193**, maxESDF **27.6688003540**, origin `[-33.4,-26.0,-0.4]` | ✅ |
| 03 | 20/20, collision 0, median 14.4067 m, min clr 0.1581 | 20/20, median **14.41 m**, 범위 2.26~31.90 m, min clr **0.158**(ep11), 재추출 **1건** | ✅ |
| 04 | 20/20, 8,378 frame, mesh-anchor 0.00462 m | **PASS** 20/20, **8,378 frame**, mesh-anchor median **0.00462 m** / worst 0.00727 m, step 위반 0, episode별 frame 수 20개 전부 동일 | ✅ |
| 04 용량 | 약 3.1 GB | **0.81 GB** | ⚠️ 미해소 |

## 해소하지 못한 차이 — 저장 용량 0.81 GB vs 문서 3.1 GB

내용 검증은 전부 통과했으나 총 바이트가 다르다. **인코더 설정 차이 가설은 기각**:
이 빌드(OpenCV 4.11.0)의 *기본값*이 on-disk depth PNG를 59,408 B로 바이트 단위 재현한다
(compression 0/1/3/6/9 = 259,945 / 47,634 / 42,133 / 35,200 / 34,098 B 중 어느 것도 아님 —
default가 정확히 일치). 스트림별 실측 rgb 평균 25 KB · depth 평균 69 KB · extrinsic 192 B로
총계와 정합한다. 3.1 GB는 다른 환경 측정값이고 여기서 재현·검증할 수 없었다.
**데이터 유효성에는 영향 없음** — frame 수·스트림 일치·depth dtype/shape/값범위·mesh-anchor·
pose step 전부 기준선 일치, PNG는 무손실이며 디코딩된 depth 값을 직접 확인했다.

## 실측으로 확인한 것 (문서에 없던 것)

- **RGB는 NuRec GS 볼륨에서 나온다.** 렌더 결과에 한글 안내판·노란 유도블록·개찰구·자판기가
  선명하게 나온다. 텍스처 없는 collision mesh(`mesh.ply`)라면 흰색이 됐을 것 —
  Open3D 04가 실제로 그랬다(RGB mean 249~255, FAIL).
- **depth는 collision mesh에서 나온다.** `usdz_scene_utils.expose_collision_meshes_for_rendering()`이
  runtime stage에만 visibility를 켠다(원본 usdz 무수정).
- **depth 원거리는 invalid(0)로 남는다 — mesh 구멍이 아니라 cutoff다.** ep0 frame 0을 행별로
  재보면 invalid는 horizon 행(54~134)에만 62/87/73%로 몰려 있고 그 행들의 유효 depth가 **정확히
  10.00 m에서 포화**한다. 바닥·천장 행은 invalid 0%이고 최대 depth도 1.86~8.79 m로 cutoff에 안 닿는다.
  프레임 전체 invalid 22.2%. 실측 raw 범위 227~10000 (= 0.227~10.000 m), uint16, (270,480).
- **결정론 확인**: episode별 frame 수가 기준선 목록(285, 344, 282, 611, 72, 682, 487, 383, 708,
  407, 159, 845, 65, 420, 181, 417, 72, 463, 583, 912)과 순서까지 전부 동일. 누적값도 중간
  체크포인트(1,594 / 3,854 / 5,750 / 7,466 / 8,378)에서 매번 일치했다.
- **Isaac Sim NuRec 지원은 내장.** `librtx.hydra.so`에 `NuRecVolume::Sync`/
  `createOrUpdateNuRecVolume`, `usdrt.scenegraph`에 `OmniNuRecVolume`/`OmniNuRecFieldAsset`
  토큰. `apps/isaacsim.exp.base.kit:209 useFabricSceneDelegate=true`. 별도 extension 불필요.
- usdz 내부: `default.usda`(834 B) → `gauss.usda`(2.4 KB) → `noeun_station.nurec`(2.15 GB) +
  `mesh.ply`(384,207 verts / 665,812 faces, 색·UV 없음) + `mesh.usd`. **Camera prim 없음** →
  카메라는 `camera_profiles.py`가 정의.
- **stdout이 안 나온다**: 00/01은 Isaac 로그에 묻혀 파이프라인 print가 보이지 않는다.
  성공 판정은 JSON/`report.html` 파일로 해야 한다(02/03은 print가 나옴).
- **`simulation_app.close()`에서 kill됨**: 04는 `main()` 반환 후(=산출물·report 저장 완료 후)
  종료 훅에서 오래 대기한다. 문서에 기록된 동작 그대로이며 실패가 아니다.

## 산출물 위치

```
scripts/dataset_converters/gs_vlnpe/apply_real/
  target_schema.json
  scene_meta/noeun_station_mid.json
  esdf/noeun_station_mid.{npz,json}
  paths/noeun_station_mid_random.json
  obs/noeun_station_mid_random_isaac_d455_nominal/{camera_config.json,episode_0000NN/{rgb,depth,extrinsic,intrinsic.npy}}
logs/gs-vlnpe/apply_real/<script>/<scene>[_random][_d455_nominal]/report.html
```
(전부 `.gitignore` 제외 — 커밋되지 않음)

## 남은 작업

1. **정합**: `apply_real/` 약 7,000줄 중 실제 diff는 833줄(12%)뿐.
   `esdf_utils.py`/`geometry_utils.py`는 diff 0, `viz_utils.py`/`dataset_utils.py`는 2,
   `03`은 6. → 원본에 optional arg로 병합하고 사본 삭제.
2. **저층·상층**: 중간층(-0.05)만 완료. low(-5.30) / high(+6.80)은 probe 이미지만 존재.
   `esdf_utils.detect_floor_levels()`가 파이프라인에 미연결.
3. **카메라 프로파일**: `PROFILES`에 `d455_nominal` 하나뿐 → `d435i_nominal` 등 추가.

## 관련 문서

- `.claude/memory/codex/codex_noeun_usdz_mid_00_to_04_complete_reproduction_guide.md` (상세 근거)
- `scripts/dataset_converters/gs_vlnpe/apply_real/README.md` (빠른 실행)

## 발행한 report

https://claude.ai/code/artifact/e42ce9d3-f8f2-41f3-b4bb-9dd889459abc

## depth 저장범위 30 m 실험 (2026-08-20 추가)

`camera_profiles.py`에 `d455_30m` 추가 (해상도·화각·초점거리는 `d455_nominal`과 동일,
저장 0.1~30 m, **render_far 12 → 35 m**). 두 사본(`gs_vlnpe/`·`apply_real/`) 동일 유지.
`--camera` default는 `dataset`이라 기존 동작 불변. 출력 경로에 카메라명이 들어가 10 m 데이터셋과
분리 저장됨(`obs/..._isaac_d455_30m`).

**render_far도 같이 올려야 한다**: `generate_episode`의 유효 조건이
`depth < camera.render_far_m * 0.99`라서 far가 12 m면 30 m는 애초에 담기지 않는다.

경로 0·1 렌더 결과:

| | d455_nominal (10 m) | d455_30m |
|---|---|---|
| ep0 값없음 평균 / 최대 | 7.6% / 22.6% | **0.1% / 1.3%** |
| ep1 값없음 평균 | 3.2% | 1.1% |
| 최대 저장값 | 10.000 m (포화) | **30.000 m** |

### RGB도 바뀐다 — far plane 때문이다 (혼동 요인을 실측으로 배제)

30 m rgb가 10 m와 달랐다(6프레임 전부, mean|d| 0.80~3.35, max|d| 161~222, px>2 5.8~22.2%).
이것이 far plane 때문인지 **렌더러 비결정성**인지 숫자만으로 구분 불가 → **같은 프로파일
(`d455_nominal`)로 ep0을 다시 렌더해 대조**했다. 결과 rgb·depth 모두 **픽셀 단위 완전 동일**
(mean|d|=0, max|d|=0, invalid flip 0). 즉 **렌더러는 결정론적이고 rgb 차이의 원인은 far plane**이다.

→ **10 m 데이터셋과 30 m 데이터셋의 rgb는 같은 장면의 서로 다른 데이터다. 섞어 쓰면 안 되고,
범위를 바꾸려면 전체를 재렌더해야 한다.**

주의: 파이프라인의 `geometry_utils.colorize_depth`는 프레임별 min~max 정규화라 10 m와 30 m를
나란히 비교할 수 없다(같은 거리가 다른 색). 비교용 시각화는 0~30 m **고정 스케일**로 따로 만들었다.
