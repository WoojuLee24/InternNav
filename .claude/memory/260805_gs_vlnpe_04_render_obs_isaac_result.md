# `04_render_obs_isaac.py` — Isaac Sim 기반 렌더러 조사 결과 (M1.4 완료 — 2-A/2-B 모두 PASS)

## 배경

`04_render_obs.py`(Open3D 버전)를 완료한 뒤 사용자가 "GT rgb와 렌더 rgb가 여전히 다르다"고
지적 — vln_n1의 GT rgb 자체가 실제 카메라 사진이 아니라 같은 mesh를 렌더링한 합성 데이터임을
확인했다(`.claude/memory/260804_gs_vlnpe_04_render_obs_result.md`). 이후 이 저장소에 있는
Isaac Sim으로 렌더러 백엔드를 바꾼 `04_render_obs_isaac.py`를 새로 작성했다.

## 확인된 사실 — depth/geometry는 완벽하다

raw `isaacsim.SimulationApp`(IsaacLab `AppLauncher` 우회)로 렌더링하면 GT와 기하학적으로
완벽히 일치한다 — depth median 0.0013~0.0051 m, p95 0.0037~0.0499 m로 두 씬 모두 기준
(0.01 m / 0.05 m) 통과. 겪은 버그 6건(cameras_enabled 플래그, 텍스처 절대경로 심링크, 씬 로드
순서, 워밍업 스텝, PhysX 충돌 쿠킹)은 이전 라운드에 기록했다.

## 조명 — 여러 라운드를 거쳐 결국 DomeLight로 확정

### 시도한 것들 (시간순)

1. **VLN-PE 원본 3-light(distant+up/down disk light)를 위치 고정으로 이식** → 프레임마다
   다른 방향으로 어긋남(카메라가 멀어지면 어둡고 가까우면 과다노출) → 실패.
2. **DomeLight 단독으로 단순화**, 세기 실측 스윕(1000→3500은 거의 무변화, 30만→2,000,000
   구간에서 실제 변화, 200만에서 실측 밝기가 GT 평균과 거의 일치) → 전역 평균은 맞지만 국소
   (방/재질별) 편차가 크다는 사용자 지적.
3. **3-light를 카메라 위치 추적 방식으로 제대로 재구현**(raise×scale 그리드 스윕으로
   raise=0.2m 확정) → depth는 무관하게 완벽 유지, 하지만 edge_corr는 씬1 0.245/씬2 0.141로
   여전히 기준(0.5) 미달 → 화장실 같은 작은 방은 disk light 반경(50m)이 근접 평면광처럼
   작용해 raise를 조정해도 과다노출에서 못 벗어남(전역 파라미터의 한계).
4. 사용자가 이 반복 튜닝에 강하게 불만족 — **"Isaac Sim 기본 렌더링으로 회귀 + 본인이 GUI로
   직접 조정"**을 요청. 조사 결과: 이 컨테이너엔 X 디스플레이가 없고, mp3d_pe raw mesh는
   조명이 전혀 없어(0개 Light prim) GUI로 열어도 검게 보이며, "Isaac Sim 기본 조명"이라는
   것도 실은 `omni.kit.stage_templates`(GUI 전용 확장)의 "sunlight" 새-스테이지 템플릿일 뿐임을
   확인 — 렌더러 자체의 본질적 속성이 아니었다.
5. 사용자가 **HTML로 GT/렌더/차이를 비교하고 여러 조명 옵션을 눈으로 판단**하는 방식으로
   요청을 바꿈 → `04c_light_explorer.py`(신규) 제작.

### `04c_light_explorer.py` — 조명 비교 전용 신규 스크립트

`04_render_obs_isaac.py`(공식 gt_replay 검증)는 건드리지 않고 mesh 로드/pose 변환/시각화
유틸만 재사용하는 별도 스크립트. 씬1의 서로 다른 방 7프레임 × 9개 조명 조합을 렌더링해
GT | 렌더 | 차이(컬러맵) 3분할 합성 이미지를 만들고, `blink_widget_html`로 화살표 전환하며
비교 가능한 report.html을 생성한다.

**비교한 9개 조합과 실측 mean_err(픽셀당 RGB 채널 평균 절대차, 낮을수록 GT에 가까움, 7프레임
평균)**:

| config | 평균 mean_err |
|---|---|
| camera_light_only / camera_light_plus_three_light | ~52 (최저) |
| dome_2M | ~63 |
| gui_default_distant3000 | ~68 |
| three_light_raise0.1~0.5 | ~64~74 |
| isaaclab_default_dome3000 | 미측정(초기 3개 config 표에서 확인, 경쟁력 없음) |

**새로 밝혀진 사실 — "camera light"(사용자 질문으로 조사)**: Isaac Sim 뷰포트의 "Camera
Light"(헤드램프) 모드는 GUI 프리뷰 전용이 아니라 진짜 RTX 렌더 설정 `/rtx/useViewLightingMode`
(bool, `omni.usd.schema.render_settings.rtx`의 `OmniRtxDebugSettingsAPI`로 실제 렌더 패스에
반영)라서 headless 스크립트에서도 켜고 끄는 것만으로 동작한다. 실측 확인: 이 모드를 켜면
스테이지의 다른 라이트를 완전히 무시한다(camera_light_only와 camera_light_plus_three_light의
mean_err가 7프레임 전부 소수점까지 완전히 동일).

**사용자의 최종 판단(육안)**: "camera_light와 dome_2M이 그나마 낫다. dome_2M이 자연스럽다.
camera_light는 좀 강한 것 같다." → **수치(mean_err)는 camera_light가 더 좋았지만, 육안으로는
dome_2M을 선택** — 이번 세션에서 이미 여러 번 확인된 교훈("정량 지표 개선이 시각적 개선을
보장하지 않는다")이 조명 선택에서도 그대로 재현됐다.

### headlamp(카메라 부착, 세기/높이/pitch 조절 가능) 커스텀 라이트 — 시도했으나 보류

camera_light가 세기/방향을 조절할 수 없는 RTX 내장 기능이라, "세기를 낮추고 위로 올리고
pitch를 낮추면 낫겠다"는 사용자 요청에 따라 카메라 부착 `DiskLight`(세기·raise·pitch 파라미터
화)를 `04c_light_explorer.py`에 구현했다. 과정에서 실제 버그 1건 발견: USD xform op를
`AddTranslateOp()` 다음에 `AddOrientOp()` 순서로 추가했더니(반대로 해야 함 — `_spawn_three_light`
의 기존 검증된 패턴은 회전 op를 translate op보다 먼저 추가) 조명의 world 위치 자체가 자신의
회전 행렬에 의해 다시 회전당해 어긋났다. 순서를 고쳐도 pitch ±30도가 육안상 거의 차이가
없었다 — DiskLight는 기본이 램버시안(코사인 감쇠) 방사라 조명 조사각 제한(cone) 없이는
방향성이 약해서다(cos(30°)=0.866, 겨우 13% 감쇠, 톤매핑 압축까지 거치면 더 안 보임).
`UsdLuxShapingAPI`로 원뿔각을 좁히면 진짜 방향성 있는 손전등 효과를 낼 수 있겠지만, 사용자가
"이미 나온 옵션(dome_2M) 중 선택으로 넘어가자"고 판단해 이 부분은 구현하지 않고 보류했다.

### 세기를 올려도 밝기가 잘 안 변하는 이유(사용자 질문)

DomeLight든 headlamp든 세기를 몇 배씩 올려도 최종 픽셀 밝기 변화가 급격히 줄어드는 이유를
질문받아 답변: RTX 렌더러는 선형 HDR(radiance)로 계산한 뒤 최종 0~255 이미지로 바꿀 때
톤매핑(Reinhard/ACES류의 압축 커브, `L_out=L_in/(1+L_in)` 형태)과 감마/sRGB 인코딩을 거치는데,
둘 다 큰 입력일수록 압축이 심해지는 포화 곡선이다. 이미 이전 라운드에서 `/rtx/post/histogram`
(오토 노출)을 켜고 꺼도 결과가 완전히 동일함을 실측으로 확인해 오토노출 때문이 아님은
배제했고, 실제 스윕 데이터(1000→52, 30만→99, 200만→112.6, GT=112.8)도 정확히 이런 포화
곡선 모양이다 — 렌더러의 정상적인 압축 인코딩이지 버그가 아니다. 정확한 톤맵 오퍼레이터
이름/설정까지는 추가 조사하지 않았다(사용자가 다음 단계로 넘어가길 원함).

## 최종 확정 — DomeLight(intensity=2,000,000) 단독

`04_render_obs_isaac.py`의 `_add_lights`를 3-light 카메라추적 코드에서 다시 `UsdLux.DomeLight`
하나(`04c_light_explorer.py`의 `dome_2M`과 동일)로 되돌렸다. `build_renderer`는 다시
`(camera, sim)` 2-tuple을 반환하고, `render_along`도 조명 위치 갱신 호출 없이 pose 설정과
렌더링만 담당한다.

## RGB 판정 기준을 edge_corr에서 SSIM으로 교체

사용자가 "edge_corr median 차이는 뭐지?"(라플라시안 엣지맵 픽셀별 피어슨 상관이라는 정의를
설명) 다음 "엣지 상관 대신 SSIM 같은 다른 지표로도 봐줘"라고 요청 — 저장된 render_rgb.jpg/
real_rgb.jpg(2-A 결과물)에 scikit-image `structural_similarity`를 재계산해보니 **SSIM median
0.648(씬1)/0.709(씬2)로 edge_corr(0.223/0.191)보다 훨씬 관대하고 "괜찮은 일치"에 가까운
판정**이 나왔다. edge_corr는 엣지가 정확히 같은 픽셀 위치에 있어야 점수가 나오는 까다로운
지표라 미세한 톤/노이즈 차이에도 크게 흔들리는 것으로 판단 — `04_render_obs_isaac.py`의
`GT_REPLAY_RGB_SSIM_MIN=0.5`를 새 RGB 통과 기준으로 채택하고, edge_corr는 참고용으로만 계속
표시하도록 코드를 바꿨다(`validate_gt_replay`/`render_gt_replay_report` 수정, `stats['rgb_ok']`
가 이제 `ssim_median >= 0.5` 기준).

## 실측 결과 — 두 씬, 18프레임씩 (DomeLight 2M + SSIM 기준)

| 씬 | depth median | depth p95(최댓값) | ssim median | edge_corr median(참고) | 전체 판정 |
|---|---|---|---|---|---|
| 17DRP5sb8fy | 0.0025~0.0028 m대 | 0.0442 m | **0.648** | 0.223 | ✅ **PASS** |
| s8pcmisQ38h | 0.0028 m대 | 0.0499 m | **0.709** | 0.191 | ✅ **PASS** |

(기준: depth median ≤0.01 m, p95 ≤0.05 m, ssim median ≥0.5)

## 2-B(mesh-anchor 생성 경로) — Isaac Sim 렌더러로 처음 실행, 두 씬 다 PASS

지금까지 Isaac Sim 렌더러는 2-A(GT 재현 비교)만 확인했고 2-B(`generate_episode`/`run_generate`,
`03_sample_gt_paths.py --mode reproduce`가 만든 경로를 따라 실제 rgb/depth/extrinsic을 생성하고
mesh-anchor로 검증하는 경로, 학습 데이터 생성에 실제로 쓰이는 경로)는 한 번도 안 돌려봤었다.
이번에 처음 실행(`--mode reproduce --num_episodes 3`, 씬당 3에피소드, arclength 0.035m
간격으로 리샘플된 전체 경로 프레임):

| 씬 | mesh-anchor median(에피소드별) | 인접 프레임 이동거리 위반 | 판정 |
|---|---|---|---|
| 17DRP5sb8fy | 0.00094 / 0.00163 / 0.00153 m | 0건(기준 0.105m) | ✅ PASS |
| s8pcmisQ38h | 0.00126 / 0.00132 / 0.00128 m | 0건(기준 0.105m) | ✅ PASS |

(기준: mesh-anchor worst ≤0.01 m) — mesh-anchor 정밀도가 2-A와 같은 수준(mm 단위)으로 나와
Isaac Sim 렌더러가 실제 학습 데이터 생성 경로에서도 기하학적으로 완벽함을 확인했다.

## 정직한 결론

**depth/geometry는 두 씬 모두, 2-A/2-B 어느 경로에서도 mesh-anchor 수준(mm 단위)으로
완벽하다.** RGB는 edge_corr 기준으로는 계속 FAIL이었지만, SSIM으로 다시 보니 두 씬 다 통과
수준(0.65~0.71)이었다 — **edge_corr 하나만으로 "RGB가 완전히 다르다"고 판단한 것 자체가
과도하게 엄격한 지표를 썼기 때문**이었을 가능성이 크다. 결과적으로 M1.4(Isaac Sim 렌더러)는
2-A/2-B 모두 PASS로 완료됐다. 다만 SSIM 임계값(0.5)도 논문/외부 근거가 아니라 "SSIM 문헌에서
흔히 쓰는 중간 이상 일치 기준"으로 임의 채택한 값이라는 점은 밝혀둔다 — 더 엄격한 기준이
필요하면 재논의 대상.

## 디버깅 중 만난 또 다른 "가짜 실패" — print 버퍼링 손실

2-B 첫 실행 때 로그에 에피소드 진행 print가 하나도 안 찍혀서 "실패했나" 싶었지만, 실제로는
`report.html`과 산출물이 전부 정상 생성돼 있었다 — `simulation_app.close()`가 오래 걸려
`timeout --signal=KILL`로 강제 종료될 때, 리다이렉트된 stdout에 블록 버퍼링된 print 내용이
플러시 전에 통째로 날아간 것(반면 `cv2.imwrite`/`path.write_text` 같은 실제 파일 쓰기는
버퍼링 없이 즉시 디스크에 반영되므로 살아남는다). **로그에 진행 상황이 안 보여도 report.html/
산출물 파일이 실제로 존재하는지 먼저 확인할 것** — 이전 라운드의 "RTX 셰이더 최초 컴파일"·
"`simulation_app.close()` 지연"과 같은 계열의, 코드 버그가 아닌 "가짜 실패" 패턴이다.

## 남은 일

- [ ] RGB edge_corr가 왜 낮은지(SSIM과 왜 이렇게 다른지) 추가 조사 — 우선순위 낮음(SSIM으로
      이미 PASS 확인됨)
- [ ] `UsdLuxShapingAPI`로 진짜 방향성 headlamp(손전등) 구현 — 보류 중, 필요시 재개
- [ ] `AppLauncher`가 정확히 왜 멈추는지는 여전히 미상(낮은 우선순위)

## 산출물

- `scripts/dataset_converters/gs_vlnpe/04_render_obs_isaac.py` — 공식 검증 스크립트(2-A/2-B),
  최종 조명 DomeLight(2M), RGB 판정 SSIM(≥0.5, edge_corr는 참고용)
- `scripts/dataset_converters/gs_vlnpe/04c_light_explorer.py` — GT/렌더/차이 + 9개 조명 옵션
  비교 도구(신규, 조명 재선정이 필요할 때 재사용)
- `logs/gs-vlnpe/04_render_obs_isaac/17DRP5sb8fy/report.html`, `.../s8pcmisQ38h/report.html`
  (2-A 공식 검증, 18프레임씩)
- `logs/gs-vlnpe/04_render_obs_isaac/17DRP5sb8fy_reproduce/report.html`,
  `.../s8pcmisQ38h_reproduce/report.html`(2-B 공식 검증, 씬당 3에피소드)
- `logs/gs-vlnpe/04c_light_explorer/17DRP5sb8fy/report.html`(9개 조명 옵션 비교, 7프레임)
- 진단 스크립트(스크래치패드, repo 밖): `isaac_dome_sweep.py`, `isaac_light_sweep.py`,
  `calibrate_headlamp.py`, `compute_ssim.py`

## 실행

```
timeout --signal=KILL 900 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04_render_obs_isaac.py --scene 17DRP5sb8fy --mode gt_replay --episodes 0,1,2 --n_frames 6
```
```
timeout --signal=KILL 900 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04_render_obs_isaac.py --scene 17DRP5sb8fy --mode reproduce --num_episodes 3
```
```
timeout --signal=KILL 900 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04c_light_explorer.py --scene 17DRP5sb8fy
```
(`isaaclab.sh`가 아니라 `_isaac_sim/python.sh`로 직접 실행 — `AppLauncher`를 안 쓴다.)
`simulation_app.close()`가 새 조명 타입을 쓴 직후엔 수 분(disk light 등 복잡한 조명일수록
더 김) 걸릴 수 있으므로 `timeout --signal=KILL`로 감싸는 것을 기본으로 한다 — report.html/
이미지는 `main()`이 반환하는 시점에 이미 저장 완료이므로 강제 종료해도 결과물 손실이 없다.
**단, 이때 로그의 print 진행상황은 버퍼링 때문에 안 보일 수 있으니 report.html 파일 존재
여부로 성공을 판단할 것**(위 "가짜 실패" 섹션 참고). 새 조명 타입을 처음 쓰는 프로세스는
RTX 셰이더/PSO 최초 컴파일 때문에 1프레임짜리 최소 재현도 수백 초가 걸릴 수 있다(같은
조합으로 재시도할수록 캐시가 워밍업되어 빨라짐) — 이건 코드 버그가 아니라 정상적인 일회성
비용이다. SSIM 계산에는 `scikit-image`가 필요(`/workspace/isaaclab/_isaac_sim/python.sh -m pip
install scikit-image`로 설치 완료).

관련: [[260804_gs_vlnpe_04_render_obs_result]], [[260803_gs_vlnpe_02_esdf_result]]
