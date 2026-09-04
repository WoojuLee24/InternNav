# gs_vlnpe 파이프라인 리포트 링크 모음 (00~04, vln_n1 + vln_pe + 노은역 usdz)

각 스크립트가 만든 `report.html`의 Artifact 링크 + 로컬 경로. 로컬 경로는 repo 루트 기준.
최종 갱신 2026-09-03 (노은역 usdz 절 추가). 상세 경위·수치는
`.claude/memory/260804_gs_vlnpe_04_render_obs_result.md`(Open3D),
`.claude/memory/260805_gs_vlnpe_04_render_obs_isaac_result.md`(Isaac),
`.claude/memory/260807_gs_vlnpe_vlnpe_dataset_result.md`(VLN-PE 지원) 참고.

## 권장 실행 커맨드

**최종 결정(2026-08-10): `--light ambient_only --film_iso 70`.** vln_n1은 `--rtx_ambient 6.0`,
vln_pe는 `--rtx_ambient 10.0`. 씬1(17DRP5sb8fy)·씬2(s8pcmisQ38h) 모두 이 설정으로 재렌더해 PASS 확인함
(아래 표의 "현재 권장" 행 참고).

```
# vln_n1 (03은 기본값, 04는 ambient 조명 권장 — SSIM 0.648 -> 0.876, film_iso 70으로 밝기편차 최소화)
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/03_sample_gt_paths.py --scene 17DRP5sb8fy --out_dir scripts/dataset_converters/gs_vlnpe/logs
timeout --signal=KILL 900 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04_render_obs_isaac.py --scene 17DRP5sb8fy --mode gt_replay --light ambient_only --rtx_ambient 6.0 --film_iso 70 --out_dir scripts/dataset_converters/gs_vlnpe/logs
```
```
# vln_pe (실측으로 확정한 파라미터 — 근거는 260807 메모리 문서, film_iso 70으로 밝기편차 최소화)
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/02_build_freemap_esdf.py --dataset vln_pe --scene 17DRP5sb8fy --r_b 0.20 --h_nav_ratio 0.12 --out_dir scripts/dataset_converters/gs_vlnpe/logs --scene_meta_dir scripts/dataset_converters/gs_vlnpe/logs
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/03_sample_gt_paths.py --dataset vln_pe --scene 17DRP5sb8fy --r_b 0.20 --h_nav_ratio 0.12 --refine_radius 0.20 --out_dir scripts/dataset_converters/gs_vlnpe/logs
timeout --signal=KILL 900 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04_render_obs_isaac.py --dataset vln_pe --scene 17DRP5sb8fy --mode gt_replay --light ambient_only --rtx_ambient 10.0 --film_iso 70 --out_dir scripts/dataset_converters/gs_vlnpe/logs
```
`timeout --signal=KILL`로 감싸는 이유: `simulation_app.close()`가 수 분 걸릴 수 있는데
report.html/이미지는 `main()` 반환 시점에 이미 저장 완료다. 로그의 진행 print는 SIGKILL 시
버퍼째 날아가므로 **성공 여부는 report.html 파일 존재로 판단**할 것.

---

# vln_n1 (기본 데이터셋)

| 단계 | 씬1 17DRP5sb8fy | 씬2 s8pcmisQ38h |
|---|---|---|
| 00 데이터 검사 | [링크](https://claude.ai/code/artifact/de729add-5777-444e-8491-d4ae1dc966f4) | — |
| 01 씬 준비 | [링크](https://claude.ai/code/artifact/e7f57d22-bfe9-449f-bd82-ce416c245013) | [링크](https://claude.ai/code/artifact/7cb35166-ff60-4530-a540-16ab31a3d316) |
| 02 ESDF 맵 | [링크](https://claude.ai/code/artifact/7c1bc45b-7fd0-4813-b666-6438fce6b9d7) | [링크](https://claude.ai/code/artifact/d84e872f-dacf-40f5-a93c-43e304faed70) |
| 03 경로 reproduce | [링크](https://claude.ai/code/artifact/1bea4e9a-997c-4f43-9fdf-2920f5596a8f) | [링크](https://claude.ai/code/artifact/5c986fb2-9a23-4287-84be-aed4b03ee38f) |
| 03 경로 random | [링크](https://claude.ai/code/artifact/4739e7ea-7a72-4c8f-9278-567a97e936f4) | [링크](https://claude.ai/code/artifact/94d32ebb-7d7c-4f97-9f04-12666b38c2ab) |
| 03b 재현 검증 | [링크](https://claude.ai/code/artifact/22087e26-59f1-4b69-be00-911e921d98ff) | [링크](https://claude.ai/code/artifact/4db6b895-57c7-4cf2-8abd-1062e321092d) |
| 03c refine 비교 | [링크](https://claude.ai/code/artifact/e69d44bf-f3a1-4fdc-b8d8-afdc3440dd2c) | [링크](https://claude.ai/code/artifact/77f179d1-2378-4e60-894a-f6b12f90d1df) |
| 03d 8축 그리드서치 | [링크](https://claude.ai/code/artifact/3a07b6b1-51a2-45fe-aae3-f559535a85ee) | — |
| 04 Open3D — GT 그대로 렌더(`gt_replay`) | [링크](https://claude.ai/code/artifact/0621a082-76fa-423f-a48d-8ae3e0309a15) | [링크](https://claude.ai/code/artifact/a5a5ba52-fc5e-4aa7-abc1-0d31620dc019) |
| 04 Open3D — 재현 경로 렌더(`reproduce`) | [링크](https://claude.ai/code/artifact/d681642b-bc3b-420f-84dd-8796131a69b7) | [링크](https://claude.ai/code/artifact/c3663deb-1aab-48e2-9c74-69ec7f12dc72) |
| 04 Open3D — 임의 경로 렌더(`random`) | [링크](https://claude.ai/code/artifact/c586d7ae-23a8-41e8-ba89-9b08db8b0931) | [링크](https://claude.ai/code/artifact/ef656f54-2be6-4e24-b0cd-08d3279fb8de) |
| 04 Isaac — GT 그대로 렌더(`gt_replay`, dome 조명·구버전) | [링크](https://claude.ai/code/artifact/847c3297-c562-4ba3-8821-669dfcf53b28) | [링크](https://claude.ai/code/artifact/a0eeab7c-b5ba-45fa-819a-3191ec5b0149) |
| **04 Isaac — GT 그대로 렌더(`gt_replay`, ambient_only+iso70·현재 권장)** | [링크](https://claude.ai/code/artifact/08a0f63f-0549-48fb-8931-5afb5b81dec5) | [링크](https://claude.ai/code/artifact/9efcccd0-38dc-4956-b50a-1eca33a7fd94) |
| 04 Isaac — 재현 경로 렌더(`reproduce`) | [링크](https://claude.ai/code/artifact/f490e7f6-e65b-4fdd-8052-1a407e2b3e50) | [링크](https://claude.ai/code/artifact/8ec9d496-befd-4325-a548-a57d52c7f034) |

**표 용어**: "GT 그대로 렌더"(`--mode gt_replay`)는 GT parquet의 실제 pose 시퀀스를 그대로 렌더링해
우리 렌더러가 실제 캡처와 같은 이미지를 만드는지 검증 — 경로 생성 로직은 개입하지 않는다.
"재현 경로 렌더"(`--mode reproduce`)/"임의 경로 렌더"(`--mode random`)는 그 검증을 통과한 같은
렌더러로 `03_sample_gt_paths.py`가 만든 경로(각각 GT의 start/goal 재현, 임의 start/goal)를
렌더링한 것.

로컬 경로 규칙: `logs/gs-vlnpe/<script_name>/<scene>[_random|_reproduce]/report.html`
(00만 `logs/gs-vlnpe/00_inspect_vln_n1/<scene>/episode_000000/report.html`).

**주요 수치**: 03 같은루트 20/20(chamfer 0.103) · 04 Open3D GT 그대로 렌더 depth 0.0000 m·SSIM 0.851 ·
04 Isaac GT 그대로 렌더 depth median 0.0025 m·SSIM 0.648(dome) / **0.876(ambient_only 6.0, film_iso 70·현재 권장)** ·
재현 경로 렌더 mesh-anchor 0.0009~0.0016 m.

---

# vln_pe (`--dataset vln_pe`)

| 단계 | 씬1 17DRP5sb8fy | 씬2 s8pcmisQ38h |
|---|---|---|
| 00 pose 변환식 검증 | [링크](https://claude.ai/code/artifact/295ddf18-9847-447b-b3ba-6bd46058c5f7) | [링크](https://claude.ai/code/artifact/cf383e9e-3fe6-4610-af4e-9072e0cf692c) |
| 02 ESDF 맵 | [링크](https://claude.ai/code/artifact/20679f72-7af9-407a-9e74-94d8efa806c2) | [링크](https://claude.ai/code/artifact/73ad3066-f496-4853-acb8-c64b300baf82) |
| 03 경로 reproduce | [링크](https://claude.ai/code/artifact/faca4ce0-cce4-4b4a-8373-d5b07da4e04f) | [링크](https://claude.ai/code/artifact/81935cbd-3955-4cf9-a713-51ec25f07177) |
| 03e 계획GT(reference_path) 대비 | [링크](https://claude.ai/code/artifact/1cd74782-37ed-4b6c-aede-a62af9dc0025) | [링크](https://claude.ai/code/artifact/e89ecf77-06bc-4ae7-840c-9ff0903174de) |
| 04 Open3D — GT 그대로 렌더(`gt_replay`) | [링크](https://claude.ai/code/artifact/c4156f51-d5ca-4fb3-9ad8-aaffc80f11bb) | [링크](https://claude.ai/code/artifact/2aa96051-86ea-4d6d-81ec-0fec95cacaa5) |
| 04 Isaac — GT 그대로 렌더(`gt_replay`, dome 조명·구버전) | [링크](https://claude.ai/code/artifact/d5b61592-37f3-4c49-958e-b880cbf0839f) | [링크](https://claude.ai/code/artifact/6a3b2e3d-0207-4ee4-a2d9-92459379953e) |
| **04 Isaac — GT 그대로 렌더(`gt_replay`, ambient_only+iso70·현재 권장)** | [링크](https://claude.ai/code/artifact/1c9330d2-a7d4-438e-ab3d-efcb97e5ffc7) | [링크](https://claude.ai/code/artifact/dddc5214-9c2d-471d-8e01-f58fe172e30b) |
| 04 Isaac — 재현 경로 렌더(`reproduce`) | [링크](https://claude.ai/code/artifact/6fa6dcf8-f9f5-4cfa-a865-cde1be70d7a8) | [링크](https://claude.ai/code/artifact/8b4cfedc-ecb5-475c-8ded-7d04230fce1c) |

표 용어는 위 vln_n1 절과 동일("GT 그대로 렌더"=`gt_replay`, "재현 경로 렌더"=`reproduce`).
로컬 경로는 위와 같되 씬 이름에 `_vlnpe` 접미사가 붙는다
(예: `logs/gs-vlnpe/03_sample_gt_paths/17DRP5sb8fy_vlnpe/report.html`).

**주요 수치**(측정 오류 정정 후 — 260807 메모리 문서의 "측정 오류 정정" 절 참고):
03 씬1 20/20 성공·**같은루트 19/20**(chamfer 0.130, `--refine_radius 0.20`) / 씬2 14/16·9/14 ·
03e 계획GT 대비 씬1 19/20 · 04 Isaac 재현 경로 렌더 mesh-anchor 0.0033~0.0079 m ·
롤아웃 자체 성공률(R2R 3 m 기준) 씬1 25/25, 씬2 7/15.

---

# 노은역 usdz (`apply_real/`, GT 없는 GS 씬)

`data/GS_USDZ/Subway/noeun_station_collision.usdz`(NuRec Gaussian-splat + 충돌 mesh, 2.2 GB)의
**중간층**(`floor_z = -0.05 m`)에서 만든 데이터셋. 위 두 절과 달리 **기존 GT 궤적이 없어**
02가 geometry만으로 맵을 만들고 03이 경로 GT를 생성한다(`--mode random` 전용, `gt_replay` 불가).
`noeun_station_mid`는 잘라낸 새 usdz가 아니라 원본 안의 범위를 적어둔 **논리 scene ID**다.

개별 단계 report는 Artifact로 발행하지 않았고, 아래 통합 리포트 하나에 5단계 대조 결과가 모여 있다:
**[노은역 00→04 실행 결과](https://claude.ai/code/artifact/e42ce9d3-f8f2-41f3-b4bb-9dd889459abc)**
(카메라·depth 범위 3종 다이어그램, rgb↔depth 전환 위젯, 10 m vs 30 m 비교 포함)

| 단계 | 로컬 경로 (`logs/gs-vlnpe/apply_real/` 이하) | 판정 | 주요 수치 |
|---|---|---|---|
| 00 자산 계약 검사 | `00_inspect_vln_n1/noeun_station_mid/` | PASS | 6 gate, triangle 665,812, 중간층 후보 면 14,618 |
| 01 중간층 논리 ROI | `01_prepare_scene/noeun_station_mid/` | PASS | 면 14,618, 면적 798.1859255685 m² |
| 02 occupancy·ESDF | `02_build_freemap_esdf/noeun_station_mid/` | UNVERIFIED | grid 1345×1308×40, occ 0.010131462, navigable 0.879639166, maxESDF 27.6688 m |
| 03 경로 GT 20개 | `03_sample_gt_paths/noeun_station_mid_random/` | PASS | 20/20, 충돌 0, 길이 median 14.41 m (2.26~31.90), min clearance 0.158 m |
| 04 렌더 (`d455_nominal`) | `04_render_obs_isaac/noeun_station_mid_random_d455_nominal/` | PASS | 20/20, **8,378 frame**, mesh-anchor median 0.00462 m / worst 0.00727 m |
| 04 렌더 (`d455_30m`) | `04_render_obs_isaac/noeun_station_mid_random_d455_30m/` | PASS | 2 episode(비교용), 값없음 7.6 %→0.1 % |

02가 **UNVERIFIED**인 이유는 실패가 아니다 — GT 궤적이 없어 독립 clearance 검증을 건너뛴 것이고
geometry sanity 게이트는 통과했다.

## 실행 커맨드 (전부 한 줄, repo 루트에서)

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

**노은역 전용 인자**: `--usd_path`(기본 `None`=기존 Matterport 경로) · `--floor_z -0.05`(수동 지정,
자동 검출 아님) · `--geometry usd`(obj가 없어 필수) · `--ref_h_b 0.875`(GT가 없어 `--ref_episode`로
못 얻음) · `--camera`(new_scene에서 기본값 `dataset`은 exit 2, 반드시 프로파일 지정).

## 새 카메라 프로파일 (`camera_profiles.py`)

| | 해상도 | hfov | depth 한 칸 | 저장 범위 | 렌더 near/far | uint16 상한 |
|---|---|---|---|---|---|---|
| `dataset` (vln_n1 GT) | 480×270 | 68.0° | 0.0001 m | 0.1~3 m | 0.05/10 m | 6.5534 m |
| `d455_nominal` | 480×270 | 90.0° | 0.001 m | 0.1~10 m | 0.05/12 m | 65.534 m |
| `d455_30m` | 480×270 | 90.0° | 0.001 m | 0.1~30 m | 0.05/35 m | 65.534 m |

**불변식**: `render_far_m > depth_max_m / 0.99` — `generate_episode`의 유효 조건이
`depth < render_far × 0.99`라서 far를 안 올리면 저장 상한을 올려도 담기지 않는다.

⚠️ **far plane 변경은 rgb도 바꾼다**(실측). 같은 프로파일로 재렌더하면 rgb·depth가 픽셀 단위
완전 동일(렌더러 결정론적)임을 확인해 원인을 far plane으로 확정했다. 따라서 **10 m와 30 m
데이터셋의 rgb는 같은 장면의 다른 데이터이므로 섞어 쓸 수 없다.**

상세 경위·수치는 `.claude/memory/260820_gs_vlnpe_noeun_run_result.md`,
실행 순서·기대 수치는 `apply_real/README.md`, 설계 근거는
`.claude/memory/codex/codex_noeun_usdz_mid_00_to_04_complete_reproduction_guide.md` 참고.

---

# 조명 실험 (vln_pe 씬1, GT 그대로 렌더 기준, 시간순)

천장이 GT보다 어두운 문제를 추적한 기록. **근본 원인은 조명 배치가 아니라
`/rtx/sceneDb/ambientLightIntensity`가 0**이었던 것(raw `SimulationApp` 부팅이 IsaacLab kit을
우회) — 조명 형태/방향/세기, 오토익스포저, 톤맵, 감마 전부 상단 밝기 ~90에서 포화했다.

| 설정 | SSIM | 상단−하단(GT +51.1) | 링크 |
|---|---|---|---|
| 9개 옵션 비교 도구(04c) | — | — | [링크](https://claude.ai/code/artifact/b73b8323-304f-4564-b9b1-cbde33a25b9a) |
| 조명없음 / headlamp 비교(04c) | — | — | [링크](https://claude.ai/code/artifact/439e907c-6e46-4a7f-92e8-e483add3f667) |
| `--light three_light` | 0.72 | — | [링크](https://claude.ai/code/artifact/20ea584a-171e-47f3-b63f-5bc8a02c7971) |
| `--light dome_three` | 0.73 | — | [링크](https://claude.ai/code/artifact/85e7c97f-353c-40ae-9d90-6ef16577c012) |
| `--light three_light --rtx_ambient 6` | **0.804** | −15.7 (GT와 반대) | [링크](https://claude.ai/code/artifact/ea8dd4d5-1984-4bc2-b7c1-8667942ceacd) |
| **`--light ambient_only --rtx_ambient 10`** (권장) | 0.779 | **+41.2** (GT와 같은 방향) | [링크](https://claude.ai/code/artifact/0f52fe77-9afa-455c-acad-684f5c59cfa5) |

마지막 두 줄이 핵심 트레이드오프다: SSIM 절대값은 three_light가 근소 우위지만 **down disk가
바닥을 과하게 밝혀 상하 명암이 GT와 뒤집힌다.** 밝기 분포를 GT와 맞추려면 `ambient_only`.

**vln_n1에도 같은 기능이 훨씬 크게 먹힌다**(2026-08-09 확인): 기존 `dome 2M`은 GT보다 어둡고
(전체 77 vs GT 113) 상하 명암도 평평했는데(+10.4 vs GT +44.5), `--light ambient_only
--rtx_ambient 6.0`으로 **SSIM 0.648 → 0.876**(min 0.761)로 올랐다.

| vln_n1 씬1 (GT 그대로 렌더) | SSIM | 링크 |
|---|---|---|
| dome 2M (기존 기본) | 0.648 | [링크](https://claude.ai/code/artifact/847c3297-c562-4ba3-8821-669dfcf53b28) |
| **ambient_only + ambient 6** | **0.876** | [링크](https://claude.ai/code/artifact/d8fed212-f553-49d1-9603-a29141de3c97) |

### 남은 차이는 조명으로 못 고친다 (2026-08-09 실측)

ambient를 켠 뒤에도 GT와 차이가 남아 **상한부터 측정**했다 — 아래 상한 측정과 기각 후보 검증은
**전부 vln_n1 씬1 18프레임 기준**이다(vln_pe는 별도로 반복하지 않음 — 렌더 파이프라인이 같아
결론이 이전된다고 보고 vln_pe는 filmIso 곡선만 따로 재확인했다, 121행 표 참고):

| 처리 (vln_n1) | SSIM |
|---|---|
| 현재 그대로 | 0.860 |
| + 프레임별 최적 gain/bias (조명·톤 보정의 이론적 상한) | 0.894 |
| + ±3px 최적 시프트 (pose 잔차) | 0.870 |
| + 둘 다 | 0.904 |

**어떤 렌더 설정을 써도 최대 +0.034**이고 남는 `1-SSIM = 0.096`은 메쉬/텍스처 충실도다.
기각한 후보(전부 vln_n1 실측): Iray `crushBlacks`/`burnHighlights`는 RTX Real-Time에서 **무반응**,
단일 감마 보정은 0.860 -> 0.839로 악화, 슈퍼샘플링 x2는 0.856 -> 0.840으로 악화,
DomeLight+ambient 혼합은 **밝기를 거의 완벽히 맞추지만**(대역편차 -3.1/-1.7/-0.5) GT에 없는
방향성 음영이 생겨 SSIM이 0.851 -> 0.761로 하락(스텝 10->60으로 올려도 동일 = 수렴 문제 아님).

유일하게 듣는 노브는 `--film_iso`(노출)다. SSIM은 그대로지만 **눈에 보이는 밝기 편차는 줄었다**:

| | SSIM | 휘도대별 편차(어두움/중간/밝음) | 링크 |
|---|---|---|---|
| vln_n1 ambient 6.0, iso 100 | 0.876 | -4.3 / +19.4 / +37.1 | [링크](https://claude.ai/code/artifact/d8fed212-f553-49d1-9603-a29141de3c97) |
| vln_n1 ambient 6.0, iso 85 | 0.874 | -9.6 / +9.0 / +29.0 | [링크](https://claude.ai/code/artifact/c2378975-8df4-4f03-aec1-7f1e84467ada) |
| **vln_n1 ambient 6.0, iso 70**(현재 권장, 씬1, 재렌더 2026-08-10) | 0.876 (min 0.756) | 별도 미측정 | [링크](https://claude.ai/code/artifact/08a0f63f-0549-48fb-8931-5afb5b81dec5) |
| **vln_n1 ambient 6.0, iso 70**(현재 권장, 씬2 s8pcmisQ38h, 2026-08-10) | 0.800 (min 0.746) | 별도 미측정 | [링크](https://claude.ai/code/artifact/9efcccd0-38dc-4956-b50a-1eca33a7fd94) |
| vln_pe ambient 10.0, iso 100 | 0.779 | +28.1 / +44.4 / +7.4 | [링크](https://claude.ai/code/artifact/0f52fe77-9afa-455c-acad-684f5c59cfa5) |
| vln_pe ambient 10.0, iso 75 | 0.781 | +16.4 / +29.4 / -5.5 | [링크](https://claude.ai/code/artifact/06e970e3-c3fd-4abc-bc51-1a3e278be216) |
| **vln_pe ambient 10.0, iso 70**(현재 권장, 씬1, 재렌더 2026-08-10) | 0.775 (min 0.654) | 별도 미측정 | [링크](https://claude.ai/code/artifact/1c9330d2-a7d4-438e-ab3d-efcb97e5ffc7) |
| **vln_pe ambient 10.0, iso 70**(현재 권장, 씬2 s8pcmisQ38h, 2026-08-10) | 0.865 (min 0.745) | 별도 미측정 | [링크](https://claude.ai/code/artifact/dddc5214-9c2d-471d-8e01-f58fe172e30b) |

iso 70 행들은 이번에 `gt_replay`로 재렌더해 얻은 값이다(SSIM은 `04_render_obs_isaac.py`가 report.html에
직접 계산해 넣은 값 — 휘도대별 편차는 `04d_tonemap_sweep.py --mode iso` 스윕에서만 산출되므로
이번 재렌더에는 없다). SSIM 절대값은 iso 85/75 대와 거의 동일(±0.002~0.006, 노이즈 수준) — 노브가
밝기 편차만 줄이고 SSIM엔 영향이 없다는 기존 결론과 일치한다. 씬2는 이번에 처음으로 ambient+iso70
조합을 확인했고 두 데이터셋 모두 PASS.

근거 스윕 리포트(신규 `04d_tonemap_sweep.py` — Isaac 한 번만 띄우고 설정 조합을 순회):

| 스윕 | 데이터셋 | 결론 | 링크 |
|---|---|---|---|
| `--mode tonemap` | vln_n1 | Iray 하위 노브 무반응, filmIso만 반응 | [링크](https://claude.ai/code/artifact/e9167e2b-1d53-45f1-9a3b-ce55ef07bfc0) |
| `--mode flatten` | vln_n1 | 밝기를 맞출수록 SSIM 하락(대조군 포함) | [링크](https://claude.ai/code/artifact/434681ca-9565-4cb3-b4ca-74855a4dfa13) |
| `--mode iso` | vln_pe | 노출별 SSIM·휘도대 편차 곡선 | [링크](https://claude.ai/code/artifact/2dd470b3-2a55-4fa9-86a6-66cd4eea31b3) |

자세한 경위는 `.claude/memory/260809_gs_vlnpe_render_residual_result.md`.

`--light` 4종(dome / three_light / dome_three / ambient_only)과 조정 인자
`--rtx_ambient`, `--tl_raise`, `--tl_up_intensity`, `--tl_down_intensity`,
`--tl_distant_intensity`, `--tl_dome_intensity`, `--film_iso` 전부 CLI로 노출돼 있다. `--rtx_ambient`는
**씬 로드 후**(`build_renderer` 내부)에 적용해야 먹는다 — `main()` 초반에 걸면 RTX 초기화가
덮어써서 무효(실측).

---

# 그 외 조사용 임시 리포트

| 내용 | 링크 | 로컬 경로 |
|---|---|---|
| pose 규약 변경 전/후 비교 | [링크](https://claude.ai/code/artifact/6446f9e3-a51d-479b-a80e-bc67abffb08b) | `logs/gs-vlnpe/_convention_compare/17DRP5sb8fy/report.html` |
| h_nav 0.15→0.10 경로 비교 | [링크](https://claude.ai/code/artifact/f7541d43-2020-4362-a50b-77313573485f) | `logs/gs-vlnpe/_hnav_compare/17DRP5sb8fy/report.html` |
| 00 초기판(2026-08-02) | [링크](https://claude.ai/code/artifact/5880bb05-70fe-4f39-8836-33b25f4678f4) | (구버전, 위 00 링크로 대체) |
| 03 bezier 스무딩(구버전) | [링크](https://claude.ai/code/artifact/f69a22ab-a400-400f-8d66-c7f5f9f9ad4d) | (cubic으로 대체, 로컬 파일 없음) |

---

이 파일이 **리포트 링크의 단일 소스**다. 다음 두 파일은 Isaac 렌더러만 다루던 부분집합이라
여기 내용에 포함돼 있다(참고용으로만 남겨둠):
`04_render_obs_isaac_reports.md`, `.claude/memory/260805_gs_vlnpe_04_render_obs_isaac_links.md`.
