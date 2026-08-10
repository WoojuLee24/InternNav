# 00~04 파이프라인 VLN-PE 데이터셋 지원 (`--dataset {vln_n1,vln_pe}`) — 완료

## 요청

vln_n1 전용이던 00~04 파이프라인을 VLN-PE 데이터셋에도 적용해 reproduce. ① `--dataset` arg로
전환, ② 두 데이터셋 모두 reproduce + logs 분리 + 시각화 + report, ③ 스텝별 보고, ④ 중복 공유.

## 신규 공유 모듈 — `dataset_utils.py`

4곳에 중복돼 있던 `load_gt_episode`(02/03/geometry_utils/04x2)와 이미지 로더를 registry 패턴
하나로 통합. `DATASETS` dict(data_root/render_wh/tag), `load_gt_episode()`(poses_c2w/cam_xyz/
body_xyz/h_b/pitch_deg/floor_z/k), `load_rgb_frame`/`load_depth_frame_m`(dataset 분기),
`dataset_tag()`(출력 경로 접미사 — vln_n1은 ''로 기존 경로 유지). `python dataset_utils.py`가
자체 회귀(vln_n1 bit-identical + vln_pe 실측 관계 검증).

## VLN-PE 포맷 (실측 확정 — meta/info.json은 vln_n1 스키마를 베낀 stale이라 신뢰 금지)

| | vln_n1 | vln_pe |
|---|---|---|
| GT pose | `action`(4x4 OpenGL c2w) | `camera_position`+`camera_orientation`(**wxyz** 쿼터니언, 15° 마운트+토르소 피치 포함) |
| intrinsic | 컬럼 | 없음 — USD 실측 90° hfov 256² → fx=fy=cx=cy=128 |
| RGB/depth | 프레임별 jpg/png | 에피소드당 npy 스택; depth m=raw×10, 1.0=클립(invalid) |
| 해상도 | 480×270 | 256×256 |

**쿼터니언→OpenCV**: R(q) 컬럼=[forward,left,up] → `R_cv = R[:,[1,2,0]] * [-1,-1,+1]`.
00의 `--dataset vln_pe` 모드가 mesh-anchor(대조군 2개 포함)로 실측 확정 — 씬1 정지 frame0
0.006m vs 대조군 0.55/0.76m(16배). **h_b/floor_z 유도**: extrinsic이 없어
`floor_z=median(rob_z)−0.95`(H1 torso 기립고 실측 상수), `h_b=median(cam_z)−floor_z`(≈1.66m).
**body_xyz**: 카메라가 몸통보다 0.2m 앞(마운트 [0.2,0,0.72], 잔차<8.2mm 실측)이라 제자리
회전만 해도 카메라 xy가 가짜 경로를 그림 → 02/03은 `body_xyz`(robot_position)를 쓴다.

## vln_pe GT 데이터의 속성 (파이프라인 버그 아님 — 전부 실측으로 원인 확정)

1. **보행 pose-depth 지터**: 물리 시뮬 캡처라 pose 기록과 depth 렌더 시점이 어긋남. 정지
   frame0은 6mm, 보행 프레임은 2~9cm(±2프레임 오프셋 보정 불가 — 고정 아님). → 04 2-A depth
   허용치를 vln_pe만 median≤0.10m/p95≤3.0m로 분리(p95는 회전 경계 시프트가 지배, 참고용).
2. **가구 관통**: GT가 가구 높이(0.24~0.79m) 점유 셀을 통과(충돌 단순화/카메라 돌출) → 03
   A*가 그 길을 못 씀(씬1 5/20 astar_failed, 컴포넌트 단절까지 확인). 로봇 키와 무관.
3. **13cm 문턱 밟기 + 다락 낮은 천장**: H1(h_b 1.66m) 밴드에서 clearance 게이트의 vln_n1 범위
   이전 불가 → 02 게이트를 "좌표/맵 파손 검출" 목적 기준(median>0.1m & zero_frac<30%)으로
   vln_pe만 분기. h_nav는 `--h_nav_ratio 0.12`(≈0.2m) 권장.
4. **씬2 롤아웃 배회**: GT 회전 500~1000deg/m(A* 17~45) — traj_data가 물리 롤아웃이라 정책
   진동/우회가 그대로 있음(body_xyz로도 동일 → 카메라 오프셋 아님). A*와 DIFFERENT-ROUTE가
   정상. 진짜 계획 GT와 비교하려면 raw_data/r2r의 reference_path 매칭 필요(후속 과제).
5. 퇴화 에피소드(씬2에 2프레임짜리 1개) → 03에서 <5프레임 skip 가드.

## 스크립트별 변경

- **00**: `--dataset` + `run_inspect_vln_pe()`(guard clause) — 스키마 덤프 + mesh-anchor 게이트.
- **01**: vln_n1 전용 유지 — mesh 전용이 아니라 vln_n1 GT(ply/extrinsic)에 결합돼 있고, 목적은
  기존 vln_n1 실행(같은 mesh) + 00 vln_pe 모드가 커버(중복 구현 회피).
- **02/03/04/04_isaac**: 로더를 dataset_utils로 교체(vln_n1 경로는 self-check로 bit-identical
  확인), 출력에 `_vlnpe` 태그, RENDER_WH per-dataset(vln_pe 256²), per-dataset 게이트.
- **04(Open3D)에 SSIM 추가**: isaac 버전과 RGB 판정 기준 통일(ssim≥0.5, edge_corr 참고용).

## 결과 (두 씬 모두, --out_dir scripts/dataset_converters/gs_vlnpe/logs)

| 스텝 | 씬1(17DRP5sb8fy) | 씬2(s8pcmisQ38h) |
|---|---|---|
| 00 pose 검증 | PASS(frame0 6mm, 16배) | 승자 일관(4/4 eps)·frame0 1.1~1.9cm — 씬 mesh 품질 차이 |
| 02 ESDF | PASS (zero_frac≤5.1%) | PASS (≤19%, 다락 낮은 천장) |
| 03 reproduce | 15/20, 같은루트 14/15, chamfer 0.138 | 11/16, 같은루트 1/11(배회 롤아웃 탓) |
| 04 Open3D 2-A | PASS ssim 0.683 | PASS ssim 0.746 |
| 04 Isaac 2-A | PASS ssim 0.553(정지 depth 1~4mm) | PASS ssim 0.788(정지 16~20mm) |
| 04 Isaac 2-B | **PASS mesh-anchor 0.0033~0.0054m** | **PASS 0.0063~0.0079m** |

vln_pe에서 edge_corr(0.3~0.8)가 vln_n1(0.19~0.22)보다 좋음 — GT가 같은 Isaac 렌더러 출신.

## vln_n1 회귀 (default 경로 무결성)

- 02: PASS(h_b 0.700 등 기존값 일치) / 03: 20/20 같은루트, chamfer 0.103(기존 0.107 — esdf
  재생성의 표면 샘플링 차이 수준) / 04 Open3D: depth 0.0000m·edge_corr 0.713(기존 일치),
  ssim 0.851 / 04 Isaac: (결과는 리포트 참고 — 기존 0.0025m/ssim 0.648 수준 기대)

## 실행 예 (out_dir은 현재 레이아웃상 logs 하위)

```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/00_inspect_vln_n1.py --dataset vln_pe --scene 17DRP5sb8fy
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/02_build_freemap_esdf.py --dataset vln_pe --scene 17DRP5sb8fy --out_dir scripts/dataset_converters/gs_vlnpe/logs --scene_meta_dir scripts/dataset_converters/gs_vlnpe/logs
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/03_sample_gt_paths.py --dataset vln_pe --scene 17DRP5sb8fy --h_nav_ratio 0.12 --out_dir scripts/dataset_converters/gs_vlnpe/logs
timeout --signal=KILL 900 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04_render_obs_isaac.py --dataset vln_pe --scene 17DRP5sb8fy --mode gt_replay --out_dir scripts/dataset_converters/gs_vlnpe/logs
```
(pkill을 다음 실행과 같은 백그라운드 복합 커맨드에 넣지 말 것 — `pkill -f` 패턴이 자기 셸
커맨드라인과 매치돼 자기를 죽인다. 실측으로 확인.)

관련: [[260805_gs_vlnpe_04_render_obs_isaac_result]], [[260803_gs_vlnpe_03_reproduce_result]]

## 남은 문제 해결 (2026-08-08)

**잔여1 — 씬2 정지프레임 1~2cm 오프셋**: mesh 버전 가설 **기각**(raw obj/isaacsim obj/fixed.usd
전부 소수점까지 동일 거리). 진짜 원인: frame0도 완전 정지가 아님 — 두 씬 모두 첫 프레임 구간에
13~14° 회전 중(실측). 즉 같은 캡처 타이밍 지터의 **회전 성분**이고, 씬2는 배회 롤아웃이라
느린 프레임조차 제자리 회전 중이어서(느린 프레임 anchor 7~11cm) 깨끗한 프레임이 없다. 우리
쪽에서 고칠 수 없는 GT 속성으로 확정, 씬1의 6mm가 변환식 검증을 담보.

**잔여2 — 계획GT(reference_path) 재평가**: 신규 `03e_compare_reference_path.py` —
instruction 텍스트로 traj↔raw 매칭(10/10), Habitat y-up→z-up `(x,-z,y)` 변환(start 오차
0.01~0.17m 검증). 결과: **씬1 같은루트 13/15**(chamfer median 0.159m) — 03의 재현 품질이 계획
GT 기준으로 회복. **씬2는 롤아웃 9/11이 목표 미도달**(끝점이 ref 목표에서 2~16m 이탈 = 실패
에피소드)이고, ref의 진짜 끝점으로 직접 A* 계획해도 **8/11 불가** — 의도된 경로가 H1 밴드
맵에서 막힌 다락 통로를 지나며, **VLN-PE 로봇 자신도 그 경로들을 실패**했다(정합적).

**잔여3 — vln_pe 렌더 톤 정합**: `04_render_obs_isaac.py`에 `--light {dome,three_light}` 추가
(기본 dome, vln_n1 경로 무변경). three_light = VLN-PE 실제 프로덕션 조명(distant+카메라추적
disk, raise 0.2m). vln_pe 씬1 2-A 실측: **ssim median 0.553(dome) → ~0.72(three_light)**,
edge_corr도 0.61~0.75로 균일하게 상승, 육안으로도 GT와 거의 동일한 톤. **vln_pe 렌더에는
`--light three_light` 권장**(GT가 같은 레시피로 렌더된 것이므로 당연한 결과지만 실측으로 확정).
구현 시 함정 재확인: str.replace 패치가 조용히 no-op 되어 build_renderer 반환만 2-tuple로
남는 버그(unpack ValueError → atexit segfault로 위장됨)를 겪음 — 컴파일 체크만으론 못 잡고
실행으로 잡았다.

관련 리포트: `logs/gs-vlnpe/03e_compare_reference_path/<scene>_vlnpe/report.html`,
`logs/gs-vlnpe/three_light/04_render_obs_isaac/17DRP5sb8fy_vlnpe/report.html`

## 사용자 후속 이슈 3건 (2026-08-08)

**이슈1 — "롤아웃이라도 rgb/depth 비정렬은 이상하다" → 사실로 확인, 데이터 수집 아티팩트로 확정**
- 분수 프레임 보간(slerp, dt −1.0~+1.0 스윕): 프레임별 최적 dt가 +0.2~−1.0으로 제각각, 잔차
  1~3cm — **일정한 지연이 아님**(보정 불가).
- rgb↔depth 상호 정렬 검사(depth 불연속 픽셀 위의 RGB 그라디언트): frame40에서 rgb가 **f+1의
  depth와 더 잘 정렬** — rgb와 depth조차 서로 다른 시점에 캡처될 수 있음.
- 결론: VLN-PE 수집 루프에서 pose/rgb/depth 세 스트림이 **각자 독립적인 가변 지연**으로
  샘플링됨. 사후 보정 불가한 GT 아티팩트. (사용자 직관이 옳았다 — "지터" 한 단어로 뭉뚱그릴
  문제가 아니라 스트림별 비동기가 실체.)

**이슈2 — three_light 천장 어두움: 원인 규명 + dome_three 하이브리드 + 조정 인자**
- 실측(씬1 5프레임, 상단 1/3 밝기): GT 207.6 vs three_light(카메라 근처) 62.8~65.5 —
  **disk 세기를 3배(5k→15k) 올려도 거의 불변**(위치 토폴로지가 지배, 세기 아님).
- 원본 공식 문자 그대로(z=−cam−1: 바닥 아래에서 위로, 뒷면 투과 가설)는 **기각** — 전체가
  오히려 어두워짐(56, 메쉬가 차단). 세기 4배도 무반응(59.6→59.7).
- 해결: `--light dome_three`(three_light + 약한 DomeLight ambient 병행, 기본 500k) 추가 —
  ssim ~0.72→~0.73, 천장 육안 개선(GT 수준까지는 아직).
- **조정 인자(신규 CLI)**: `--tl_raise`(disk 오프셋 m), `--tl_up_intensity`/`--tl_down_intensity`
  (disk 세기), `--tl_distant_intensity`, `--tl_dome_intensity`(dome_three 전용, 천장 밝기의
  실질 레버 — disk 세기는 천장에 안 먹힌다는 게 실측 결론).

**이슈3 — 03e legend + 경로 차이 원인**: overlay 범례(노랑=reference 계획GT, 하늘=planned A*,
빨강=rollout 실주행, 초록=navigable)를 리포트에 명시, DIFFERENT 에피소드마다 원인 컬럼 추가
(① 롤아웃 목표 미도달(Xm) → planned가 잘못된 끝점 기준, ② A*가 다른 corridor 선택(최단경로/
차단 우회)). 같은 URL로 재발행.

## 측정 오류 정정 + 진짜 원인 (2026-08-08, 사용자 재질문으로 발견)

사용자가 "GT인데 왜 장애물을 뚫고 가지?", "GT가 롤아웃인데 왜 rgb/depth가 안 맞지?"라고
되물어 재검증한 결과, **위에 적은 진단 3건이 틀렸다.** 정정한다.

| 이전 주장 | 실제 |
|---|---|
| GT가 가구를 관통(0.24~0.79m 점유셀 통과) | **측정 오류** — `cam_xyz`(몸통보다 0.2m 앞)로 쟀다. `body_xyz`로는 관통 **0건**. 카메라가 가구 위(1.66m)로 튀어나온 것뿐 |
| 13cm 문턱 밟기 + zero clearance 5~19% | **측정 오류** — 같은 원인. body_xyz 기준 zero_frac **0%**, min clearance 0.15~0.40m |
| A* 실패는 GT가 못 지나는 길 | **r_b=0.25(vln_n1 값)가 과대**. H1 실제 통과 clearance는 0.141~0.158m → `--r_b 0.20`으로 씬1 15/20 → **20/20** |
| 씬2 루트 불일치는 배회 롤아웃 탓 | **다층 버그** — 03이 씬 대표 floor_z 하나만 써서 다락(floor≈2.9m) 에피소드를 1층 맵으로 계획. 에피소드별 floor_z 자동 감지(`--floor_tol_m`, 기본 0.3) 추가 → 씬2 같은루트 **1/11 → 9/14** |
| 천장이 어두운 건 조명 위치/세기 문제 | **`/rtx/sceneDb/ambientLightIntensity`가 0이었다** — raw SimulationApp 부팅이 IsaacLab kit(이 값 1.0)을 우회해서. 조명 형태/방향/세기/오토익스포저/톤맵/감마 전부 top≈90에서 포화했지만 이 값 하나로 해결 |

### 정정된 실측치

- **03 reproduce**(`--r_b 0.20 --h_nav_ratio 0.12`): 씬1 **20/20 성공, 같은루트 17/20**, chamfer 0.147 /
  씬2 **14/16, 같은루트 9/14**, chamfer 0.195 (이전 0.403)
- **03e 계획GT(reference_path) 대비**: 씬1 **19/20**(chamfer 0.173), 씬2 3/14 — 씬2는 롤아웃 자체가
  목표 미도달이라 planned 끝점이 다른 게 정상
- **롤아웃 성공률**(R2R 기준 목표 3m 이내): 씬1 **25/25**, 씬2 **7/15**
- **02 clearance**: 두 씬 PASS, zero_frac 0%
- **04 2-A 조명**(씬1 5프레임 SSIM): ambient 0/4/6/8/10/13 → 0.715/0.818/**0.824**/0.819/0.810/0.794.
  **`--light three_light --rtx_ambient 6.0` 권장**(전체 밝기 182.0 vs GT 178.4)

### rgb/depth 비정렬 — 대조군으로 확정(사용자 지적이 옳았다)

엣지맵 픽셀시프트 상호상관(±4px 스윕):
- **vln_n1(대조군)**: 5프레임 중 4개가 정확히 (0,0)에서 peak, 비율 1.00x → 완전 정렬
- **vln_pe**: **한 번도 (0,0)이 peak가 아님**, 최적 시프트가 (0,−3)/(−4,−4)/(+4,+1)/(+4,+3)/(+1,−3)로
  매 프레임 다름, peak/(0,0) 비율 1.18~4.01x
- pose 쪽도 분수 프레임 slerp 보간(dt −1~+1)으로 스윕했으나 최적 dt가 +0.2~−1.0로 제각각(잔차 1~3cm)
- **VLN-PE 코드 자체가 인지**: `vln_eval_task.py:206` `# without this, possible issues: delay by get_rgb`
- 결론: pose/rgb/depth 세 스트림이 각자 다른 가변 지연으로 샘플링됨. 사후 보정 불가.
  근본 해결은 VLN-PE 수집 루프에서 세 스트림을 같은 렌더 틱에 동기화하는 것.

### 재발한 함정

`pkill -f <패턴>`을 다른 명령과 같은 백그라운드 복합 커맨드에 넣으면 **감싸는 셸의 커맨드라인이
패턴에 매치돼 자기 자신을 죽인다**. 이번 세션에서 두 번 당했다(스윕 로그가 통째로 사라짐).
반드시 별도 foreground 호출로, 자기 자신과 매치되지 않는 패턴으로 실행할 것.

### ambient 적용 위치 함정 (추가)

`/rtx/sceneDb/ambientLightIntensity`는 **씬/카메라가 올라온 뒤**(`build_renderer`의 `sim.reset()`
직후)에 걸어야 한다. `main()` 초반(씬 로드 전)에 걸었더니 RTX 초기화가 기본값으로 덮어써서
전혀 반영되지 않았다 — 스윕 스크립트에선 먹히는데 본 스크립트에선 SSIM이 0.72로 무변화라
같은 설정이 다르게 동작하는 것처럼 보였다. 최종 18프레임 검증: **SSIM median 0.804**
(dome 0.553 → three_light 0.72 → three_light+ambient6 **0.804**).

## 사용자 후속 4건 (2026-08-09)

**① `delay by get_rgb` 주석 — 샘플링에 반영 안 됐다(코드 근거)**
`vln_eval_task.py`의 `get_rgb_depth()`를 읽으면:
- `rep.orchestrator.step(rt_subframes=2, ...)`가 **`env_id == 0`일 때만** 호출된다 — 병렬 env는
  명시적 렌더 스텝 없이 버퍼에 있는 걸 그대로 읽는다.
- `cur_obs = camera.get_data()`로 이미지를 받은 **뒤에** `camera.get_world_pose()`로 pose를
  읽는다. 이미지가 렌더된 시점의 pose가 아니라 **현재 sim 상태의 pose**다. 타임스탬프 매칭도
  보정도 없다.
- `rt_subframes=2`는 우리가 실측한 애노테이터 캐치업 소요(카메라 이동 후 ~10스텝)보다 훨씬 적다.
- `warm_up_step = 50` 주석은 지연을 **인지했지만** 워밍업 구간에서 제자리걸음을 늘리는 우회였을
  뿐, pose↔이미지 동기화는 하지 않았다.
→ 우리가 측정한 pose/rgb/depth 3중 어긋남과 정확히 일치. 사후 보정 불가.

**② three_light가 아래를 더 밝게 만든다 — 사용자 지적이 맞고, `ambient_only`로 해결**
씬1 6프레임 상단1/3 vs 하단1/3 실측:

| 설정 | 상단 | 하단 | 상−하 | 전체 | SSIM |
|---|---|---|---|---|---|
| **GT** | 209.6 | 158.5 | **+51.1** | 180.1 | — |
| three_light + ambient 6 | 168.9 | 184.6 | **−15.7**(반대) | 183.4 | 0.833 |
| **ambient_only + ambient 10** | 195.1 | 153.9 | **+41.2** | 177.9 | 0.818 |
| ambient_only + ambient 13 | 206.5 | 167.7 | +38.8 | 190.0 | 0.812 |

down disk light가 바닥을 과하게 밝혀 상하 명암이 GT와 **반대로 뒤집힌다**. 조명 prim을 아예
없애고 전역 간접광만 쓰는 `--light ambient_only --rtx_ambient 10.0`이 GT의 명암 방향(+41 vs
+51)과 전체 밝기(177.9 vs 180.1)를 재현한다. SSIM은 0.833→0.818로 근소 하락하지만 **밝기 분포가
GT와 같은 방향**이라 이쪽을 권장. (distant light는 기여가 사실상 0 — 켜고 끈 값이 소수점까지 동일.)

**③ 경로가 벽에 붙는다 — `--refine_radius 0.30`(vln_pe 한정)**
씬1 12에피소드 스윕(chamfer는 GT 대비, clearance는 중앙값):

| 설정 | chamfer | our clearance | GT clearance |
|---|---|---|---|
| rr 0.10 (기본) | 0.154 | 0.451 | 0.436 |
| **rr 0.30** | **0.147** | **0.529** | 0.436 |
| rr 0.50 | 0.159 | 0.538 | 0.436 |
| A* clearance_weight 0.3~1.0 | 0.151~0.161 | 0.474~0.532 | — |

**refine_radius 0.30이 chamfer와 clearance를 동시에 개선**한다(벽에서 더 멀어지면서 GT에도 더
가까워짐). A*의 `clearance_weight`는 도움이 안 된다. **단 vln_n1은 반대** — rr 0.30이 chamfer를
0.1285→0.1400으로 악화시키고 clearance가 GT(0.479)를 넘어 0.574로 과하다. **기본 0.10 유지,
vln_pe에서만 0.30을 준다.**

**④ navigable area는 정상이었다**
시각화로 확인(씬1): navigable 44.3% / obstacle 30.9% / r_b 미달 24.8%로 평면도가 정상이다.
GT와 경로가 갈리는 건 맵 결함이 아니라 **A*가 최단경로라 코너를 안쪽으로 지르는 것**이며,
③의 refine_radius 0.30으로 코너가 넓어지면서 GT에 더 붙는다.

## 사용자 후속 2건 (2026-08-09 오후)

**① "GT와 경로가 다른 게 장애물 회피(refine)가 잘 적용 안 돼서인가?" → 정반대다**
- refine은 **정상 작동한다**: clearance 중앙값이 A* 직후 0.382 → refine 후 0.495(rr 0.1)/
  0.646(rr 0.3)으로 확실히 올라간다(vln_n1 기준).
- **GT 자체가 clearance 최대점이 아니다** — GT 점 주변 0.3 m 안의 최대 clearance보다 GT가
  **0.20 m 낮은 곳**에 있다(vln_n1 0.217, vln_pe 0.196). 즉 "장애물에서 멀어지기"를 강화할수록
  GT에서 **멀어진다**.
- 결정적 실험(기준선 ⓒ): **GT 경로 자체를 우리 후처리에 통과**시켰을 때 GT와의 chamfer —
  refine 없음 0.019(vln_n1)/0.046(vln_pe), refine 0.1 → 0.077/0.088, refine 0.3 → 0.161/0.140.
  **refine 단계가 이탈의 주원인**이고 A* 루트 선택이 아니다.
- 편차는 코너에 몰려 있지도 않다(곡률 상위 25% 구간이 오히려 0.8~0.9배로 더 정확).
  전 구간에 걸친 **균일한 측면 오프셋**이다.

**refine_radius 트레이드오프 곡선**(실제 03 실행 기준, 씬1):

| | vln_n1 | vln_pe |
|---|---|---|
| rr 0.10 | chamfer **0.103**, 같은루트 20/20 | chamfer 0.147, 같은루트 17/20 |
| rr 0.20 | chamfer 0.106, 같은루트 20/20 | chamfer **0.130**, 같은루트 **19/20** |

→ **vln_n1은 기본 0.10 유지, vln_pe는 0.20**. (이전에 권장했던 vln_pe 0.30은 격리
재구현 스윕만 보고 정한 것이라 **0.20으로 정정** — 실제 파이프라인 실행이 판단 기준이다.
격리 스윕과 실제 실행이 어긋날 수 있다는 교훈.)

**② vln_n1에도 ambient light 적용 — 개선폭이 vln_pe보다 크다**
씬1 6프레임 실측: GT 전체 112.8·상하차 +44.5인데 기존 `dome 2M`은 전체 77.1·상하차 +10.4로
어둡고 평평했다. ambient 단독 스윕 3/6/10/14 → SSIM 0.810/**0.854**/0.806/0.758.
18프레임 실제 검증에서 **SSIM 0.648(dome) → 0.876(ambient_only 6.0, min 0.761)**.
`--light ambient_only --rtx_ambient 6.0`을 vln_n1 권장으로 한다(vln_pe는 10.0).
