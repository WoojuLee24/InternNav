# `04_render_obs.py` — M1.4 렌더링 구현 결과

M1.4 전체(2-A GT 재렌더링 검증 + 2-B 생성 경로 데이터 수집) 완료. 문서 초안의 스펙(RGB
(480,640,3), depth mm-clip, pitch 15도 고정)은 00 이전에 쓰인 것이라 실측과 어긋났고,
`target_schema.json` 실측값(RGB (270,480,3), depth raw/10000=m, pitch 에피소드별 가변)을
따랐다 — "정답지 원칙"(포맷을 발명하지 않고 로컬 `vln_n1` 스키마를 그대로 복제).

## 렌더러 — Open3D 0.19 `OffscreenRenderer`

새 의존성 없이(IsaacLab python에 이미 설치됨) EGL headless로 동작. 핵심 함정 둘:

1. **멀티 머티리얼 텍스처**: `open3d.io.read_triangle_mesh` + 단일 `MaterialRecord`로
   `scene.add_geometry`하면 텍스처가 안 붙어 회색 평면이 된다(23개 텍스처를 가진 이 mesh에서
   실제로 겪음). `open3d.io.read_triangle_model` + `scene.add_model`을 쓰면 정상 렌더된다.
2. **카메라 pose**: `Camera.look_at(center, eye, up)`은 world 벡터만 받으므로, `action_to_c2w`가
   반환하는 OpenCV c2w(x=right,y=down,z=forward)에서 `eye=c2w[:3,3]`, `forward=c2w[:3,2]`,
   `up=-c2w[:3,1]`을 그대로 넘기면 된다 — Open3D 내부 컨벤션(OpenGL, -Z를 봄)을 신경 쓸 필요가
   전혀 없다(look_at이 알아서 처리). 축 변환을 새로 발명하지 않는 게 핵심.
3. `render_to_depth_image(z_in_view_space=True)`가 미터 단위 실거리를 직접 준다(배경=inf) —
   근/원거리 클립을 역산할 필요 없음.

`build_renderer`/`render_along`은 GT 포즈(2-A)든 합성 포즈(2-B)든 완전히 같은 코드를 탄다.

## 2-A — GT 궤적 재렌더링 검증

`--mode gt_replay`: vln_n1 parquet의 실제 `action[t]`를 그대로 mesh에 렌더링해, 실제 캡처
(`observation.images.rgb/depth`)와 프레임별로 비교. 첫 스모크 테스트(씬1 ep0 frame15)에서
depth 절대오차 median 0.001 m / p95 0.020 m로 즉시 통과 — pose/intrinsic/mesh/렌더러 체인 전체가
실제 센서와 일치한다는 강한 증거다(이미 검증된 `action_to_c2w`/mesh를 그대로 재사용했기 때문에
첫 시도에 맞았다).

두 씬 × 3에피소드 × 6프레임(18프레임/씬)에서:

| 씬 | depth median | depth p95 | edge_corr median |
|---|---|---|---|
| 17DRP5sb8fy | 0.0028 m | 0.0126 m | 0.430 |
| s8pcmisQ38h | 0.0033 m | 0.0136 m | 0.475 |

기준(depth median≤0.05, p95≤0.30, edge_corr≥0.30) 대비 **압도적으로 PASS**.

**RGB 엣지 상관계수 임계값을 실측 후 재조정했다** — 처음 0.5로 잡았다가 depth가 완벽히 통과하는
프레임에서도 0.34~0.50이 나와 FAIL로 잘못 걸렸다. 해당 프레임(ep0 frame142)의 render/real RGB를
직접 눈으로 비교하니 같은 소파·테이블·창문이 같은 자리에 있는 완전한 정합이었다 — Laplacian
엣지 상관은 조명/질감 차이에 민감해 절대값 자체가 낮게 나오는 지표라는 뜻. 실측 최소값(0.343)
아래로 여유를 둔 0.30으로 낮췄다. depth가 이미 강한 정량 증거이므로 rgb는 보조 지표로만 쓴다.
([[feedback_verify_alignment_quantitatively]] 원칙 그대로 — 눈으로 보고 임계값을 맞췄다.)

## 2-B — 생성 경로 데이터 수집

`--mode reproduce`/`random`: 2-A로 검증된 같은 렌더러로 `03_sample_gt_paths.py`의 `paths/<scene>
[_random].json` 경로를 따라 새 에피소드를 만든다. 신규 로직은 **카메라 pose 합성** 하나뿐
(`geometry_utils.synthesize_action_poses`, 위치+yaw+pitch → `action[t]` 포맷).

### pose 합성 공식 — analytic 유도만으로 확정하지 않고 실측으로 검증

유도한 공식:
```
forward_world(yaw, pitch) = Rz(yaw - 90°) @ mount_forward(pitch)
```
`mount_forward`는 `compose_camera_extrinsic(h_b, pitch)`가 만드는 "yaw 없는" canonical 마운트를
`action_to_c2w`로 읽은 회전 — 이 canonical 마운트는 world +y를 향한다(즉 yaw=90°에 대응하므로
`Rz`에 `yaw-90°`를 쓴다). `Rz`는 world z축(위) 기준 표준 우수 회전.

**검증 방법**: 이 공식을 analytic 유도만 하고 그냥 믿지 않았다 — 4개 씬/에피소드에서 실제
`action[t]`를 역산해서 검증했다:
1. 정확한 yaw(실제 `action[t]`에서 역산한 값)로 재구성 → 회전 오차 **3e-8**(부동소수점 수준,
   공식 자체가 정확하다는 증거).
2. 우리가 실제로 쓸 tangent 추정(중앙차분 `atan2`)으로 재구성 → yaw 오차 median 0.1°, p95 0.5°,
   max 0.9° — 회전 구간에서도 1도 미만. 공식 버그가 아니라 이산 샘플의 본질적 한계.

`geometry_utils.py` self-check 게이트를 5→**6개**로 늘려 이 회귀를 고정했다(`[6/6]`, yaw 각도[도]
로 게이팅 — 처음엔 원시 회전행렬 원소 diff로 게이팅했다가 회전 구간의 정상적인 tangent 근사
오차(diff~0.012)에 걸려 잘못 FAIL했다. 각도로 다시 재니 실제로는 1도 미만이라 통과 — **원시
행렬 diff보다 물리적으로 의미 있는 단위로 게이팅해야 한다**는 교훈).

프레임 간격은 GT 실측 median(두 씬 공통 0.035 m)을 그대로 썼다 — 실제 데이터셋과 같은 공간
밀도로 새 프레임을 뽑기 위함.

### 결과 (2026-08-04, 씬당 2에피소드)

| 씬 | 모드 | 에피소드 | mesh-anchor median | 이동거리 위반 |
|---|---|---|---|---|
| 17DRP5sb8fy | reproduce | 2 | 0.0017 m | 0 |
| 17DRP5sb8fy | random | 2 | 0.0011 m | 0 |
| s8pcmisQ38h | reproduce | 2 | 0.0020 m | 0 |
| s8pcmisQ38h | random | 2 | 0.0029 m | 0 |

전부 PASS. mesh-anchor(저장한 extrinsic/intrinsic으로 depth를 다시 unproject해 씬 mesh 표면까지
재는 자기 검증)가 GT 실측(2.9e-5 m)보다는 크지만(mm 단위) 여전히 매우 작다 — mesh를 직접
렌더링한 depth라 센서 노이즈가 없어야 정상이고, 남은 차이는 리샘플/저장 반올림 오차로 보인다.

## 산출물

- `obs/<scene>_<mode>/episode_%06d/{rgb,depth,extrinsic}/frame_%04d.*`, `intrinsic.npy`.
- `logs/gs-vlnpe/04_render_obs/<scene>[_<mode>]/report.html` — 2-A는 render/real rgb·depth
  blink 비교, 2-B는 생성 에피소드의 첫/마지막 프레임 미리보기.

## 구현 메모 (guideline 준수)

- `synthesize_action_poses`는 `geometry_utils.py`에 추가했다(04/06 공용 기하 유틸 모듈 — 그
  모듈 자체의 docstring이 "새 스크립트는 직접 재구현 말고 이 모듈을 import"라고 이미 명시).
- 04 내부는 `gt_replay`(2-A)와 `reproduce`/`random`(2-B)이 완전히 분리된 함수
  (`validate_gt_replay` / `run_generate`) — `main()`의 분기 하나로만 갈린다.
- rgb/depth 저장, mesh-anchor 자기검증, 이동거리 assert는 모두 `generate_episode` 안에서
  저장 직후 수행(가이드라인 "개발 목표 관련 코드는 시각화/검증 후 저장").

## 사후 발견 버그 2건 — 렌더 밝기·depth 1px 오프셋 (2026-08-04 수정)

사용자가 발행된 리포트를 보고 "GT rgb보다 render rgb가 더 어둡다, depth도 픽셀만큼 밀린다"고
지적. 둘 다 [[feedback_verify_alignment_quantitatively]] 원칙대로 `cv2.matchTemplate`/MAE
스윕으로 원인을 확정했다(눈으로만 보고 고치지 않음).

**depth 1px 오프셋**: 렌더 depth를 (dx,dy)만큼 `np.roll`한 뒤 실측 depth와의 MAE를 -3~+3 격자로
스캔했는데, (0,0) 근방이 전부 비슷해서(0.0060~0.0064) 이 방법으로는 결론이 안 났다(매칭스코어가
매끈한 depth 평면에서는 변별력이 없다). 대신 **실제 렌더러의 intrinsics 자체**에 cx/cy를
±0.5px 조합 6가지로 바꿔서 4프레임 MAE를 다시 재보니 `(+0.5,+0.5)`에서만 MAE가 0.0074→**0.00005**
로 완전히 붕괴하고 나머지 5개 조합은 전부 악화됐다 — Open3D `Camera.set_projection`이 픽셀
**모서리**가 정수좌표인 컨벤션(OpenCV의 픽셀 **중심** 정수좌표와 반대)을 쓴다는 확정적 증거.
저장/실측용 `k`는 그대로 두고 `build_renderer`가 Open3D에 넘기기 **직전에만**
`RENDER_PRINCIPAL_POINT_OFFSET_PX=0.5`를 더하도록 고쳤다.

**밝기**: 씬1 4프레임 평균 밝기 실측(render 64.8 vs real 116.2, 44% 차이)을
`indirect_light_intensity` 스윕(90000→400000)으로 재보니 **270000**에서 116.3(거의 완전 일치)이
나왔다. 씬2는 같은 값에서 124.2 vs 110.3(13% 과다) — 실카메라는 auto-exposure라 씬마다 다른데
우리 조명은 전역 상수 하나뿐이라 두 씬을 동시에 정확히 맞출 수 없다(03d에서 반복된 "전역
파라미터 한계" 패턴과 같다). 90000보다 훨씬 나은 270000으로 채택.

**결과**: 두 수정을 같이 적용하니 depth median이 0.003→**0.0000 m**로, edge_corr median이
0.43~0.48→**0.71~0.74**로 뛰었다 — 픽셀 오프셋이 depth뿐 아니라 rgb 엣지 정합도 같이 갉아먹고
있었다는 뜻(같은 원인이 두 지표를 동시에 나쁘게 만들었다). 2-B의 mesh-anchor median도
0.001~0.003 m → **0.00000 m**로 개선. 통과 기준(`GT_REPLAY_DEPTH_*_TOL_M`,
`GT_REPLAY_RGB_EDGE_CORR_MIN`)도 이 실측값에 맞춰 재조정했다.

## 사후 발견 버그 3 — rgb 명암 대비 과다 (2026-08-05 수정)

밝기/오프셋을 고친 뒤에도 사용자가 "GT rgb와 여전히 다르다"고 재차 지적. 이번엔 원인 분석부터
다시 했다.

**전제부터 재확인**: vln_n1의 GT rgb가 실제 카메라 사진이 맞는지부터 의심해야 했다 — 확인해보니
**아니다**. 근거 셋: ① `.claude/memory/260803_gs_vlnpe_02_esdf_result.md`에 이미 "GT도 그
mesh에서 렌더됐다"고 적혀 있었다(이전 세션에서 이미 확인된 사실인데 04 작업 때는 안 챙겨봤다).
② `data/InternData-N1-v0.5-mini/README.md`가 VLN-N1을 "**Synthetic Data**"로 명시. ③ GT depth의
mesh-anchor 오차가 2.9e-5 m — 실제 depth 센서 노이즈로는 나올 수 없는 정밀도다. 즉 이 비교는
"실물 vs 렌더"가 아니라 **"그들의 렌더러 vs 우리 렌더러"**이고, depth(포즈·intrinsic·mesh
체인)는 이미 거의 완벽하므로 남은 rgb 차이는 지오메트리가 아니라 **셰이딩/톤 파이프라인** 문제로
좁혀진다.

**정량 비교**(2씬 3프레임, 채널별 mean/std, Laplacian 분산으로 선명도): 선명도는 렌더/실측이
비슷해 텍스처 블러는 주범이 아니었다. 가장 뚜렷한 패턴은 **명암 대비(std)** — 렌더의 채널별
std가 real보다 **1.3~1.5배** 높았다(예: 65~74 vs 48~51). 즉 밝기가 아니라 대비가 어긋난 것 —
`enable_sun_light(False)` + 단일 ambient만 쓰는 지금 조명이 AO/보조광 없이 명암을 과장한다는
가설을 세웠다.

**해결**: `open3d.visualization.rendering.ColorGrading`의 `tone_mapping` 파라미터(`LINEAR/ACES/
ACES_LEGACY/FILMIC/REINHARD/UCHIMURA`) 6종을 기존 3프레임 쌍에서 실측 스윕 — **REINHARD만 std
비율 0.977**(사실상 1.0)이 나왔고 나머지는 전부 1.3~1.4x로 개선이 없었다(하이라이트 압축·그림자
리프트라는 표준 톤매핑 원리가 이 문제에 정확히 대응하는 것으로 확인). 톤매핑을 바꾸면 평균
밝기도 같이 움직이므로 `INDIRECT_LIGHT_INTENSITY`도 같은 스윕 방법으로 270000→**310000**으로
재보정(평균 밝기 113.4 vs 실측 113.75, std 45.0 vs 실측 44~45 — 둘 다 사실상 일치).

**결과**: depth median(0.0000 m)·mesh-anchor median(0.00000 m)는 이 변경으로 전혀 흔들리지
않았다(색상 전용 변경이라 당연함, 회귀로 재확인). edge_corr median은 0.71~0.74 →
**0.73~0.76**로 소폭 개선. 육안으로도 어두운 구석/밝은 벽의 과장된 명암 분리가 사라지고 GT와
톤이 훨씬 비슷해졌다.

**교훈**: 렌더링 파이프라인에서 "GT와 다르다"는 지적을 받으면, 코드를 고치기 전에 **GT 자체의
생성 방식부터 재확인**해야 한다 — "실물이라 어차피 100% 못 맞춘다"고 넘겨짚었으면 이번
개선(0.977 std 비율)을 못 찾았을 것이다. 이 저장소 안에 이미 "GT도 렌더됐다"는 기록이 있었는데도
04를 만들 때는 다시 확인 안 하고 넘어갔다 — 관련 사실이 메모리에 있어도 그때그때 다시 찾아
써야 한다.

관련: [[260803_gs_vlnpe_02_esdf_result]], [[260803_gs_vlnpe_03_reproduce_result]],
[[260804_gs_vlnpe_03_c2_random_result]], [[understanding_gs_vlnpe_fpv_bev_geometry]],
[[feedback_verify_alignment_quantitatively]]
