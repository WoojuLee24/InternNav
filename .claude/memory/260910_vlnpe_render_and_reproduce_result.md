# vln_pe 전수 재렌더 + GT 없이 경로 재현 (2026-09-08 ~ 09-10)

## 목적

두 가지다. 섞이기 쉬우니 먼저 구분한다.

| | 무엇 | GT를 쓰나 |
|---|---|---|
| **A. 전수 재렌더** | GT의 카메라 pose를 그대로 재생해 우리 렌더러로 이미지만 다시 만든다 | pose를 **쓴다** (검증이 목적) |
| **B. 경로 재현** | start/goal만 받아 **GT 궤적을 안 보고** 같은 길을 만든다 | 궤적은 **안 쓴다** |

B의 허용선: start/goal·`h_b`·pitch·`floor_z`는 GT에서 가져와도 된다(문제 정의).
**GT 궤적(`reference_path`, parquet의 pose 시퀀스)을 쓰면 안 된다** — 그래서 GT 경유점 투입과
GT 유사도 비용은 자격 미달로 기각했다.

---

## A. 전수 재렌더 — 완료

```
data/InternData-N1-v0.5-mini/vln_pe_render/traj_data/r2r/
61씬 · 2,813 에피소드 · 271,252 프레임 · 116 GB · 렌더 5.2시간
```

GT와 씬·에피소드·프레임 수가 정확히 일치. 릴리스 `vln_pe/`는 읽기만 했다.

**핵심 단순화**: GT pose를 쓰므로 parquet·`meta/`를 GT에서 **바이트 복사**한다. 궤적·action·
지시문이 동일하고 이미지만 우리 렌더 → 구조 동일성이 복사로 보장되고 프레임 단위 대조가 된다.
action 이산화·통계 재계산이 전부 불필요.

### 검증 (`05b_verify_vlnpe.py`)

```
구조 게이트 V1~V5   61/61씬 PASS   (--deep: parquet·meta sha256 동일)
negative 5종        5/5 검출
GT 대비 SSIM        중앙값 0.8295 (최악 씬 0.6669, 0.75 이상 56/61)
```

SSIM 0.8295는 톤맵 전수 측정의 0.8232와 일치(프레임 표본 차이) → 저장 경로가 이미지를 변형하지
않았다는 뜻.

### 신규 파일

| 파일 | 역할 |
|---|---|
| `05_export_vlnpe.py` | GT pose 재생 → vln_pe 구조 저장. 씬당 프로세스 1개, 용량 가드, sentinel 재시작 |
| `05b_verify_vlnpe.py` | V1~V5 게이트 + GT 대조(SSIM) + negative 5종 |
| `vlnpe_writer.py` | vln_pe 레이아웃 라이터(Phase C용). GT 왕복 자기검사 통과 |

`vlnpe_writer.py`가 메운 `eval_gt_collect.save_episode`와 릴리스의 차이 5개:
`timestamp = fi/6.0`(30이 아님) · `observation.step = fi*50` · `episodes_stats.stats` 9키 ·
에피소드당 지시문 **3개** · parquet `huggingface` 스키마 메타(1,073 B).
마지막 것은 `pa.table({...})`만으로는 안 붙는다 → **릴리스 parquet에서 스키마를 읽어 그대로 쓴다.**

---

## B. 경로 재현 — 73.1% → 78.8%

`03 --mode reproduce`를 45씬 2,274 에피소드에 돌려 "같은 루트 비율"(Fréchet < 0.8 m)을 쟀다.
(61씬 중 15씬은 `01`이 vln_n1 GT의 `camera_extrinsic`을 읽으므로 제외, 1씬은 GT가 1프레임)

### 어긋남의 정체 (측정)

```
다른 루트 611 = 위상 다름 103 + 같은 회랑 드리프트 508
```

| | chamfer | Fréchet | 기준선 Fréchet | 길이비(우리/GT) |
|---|---|---|---|---|
| 같은 루트 | 0.165 | 0.465 | 0.322 | 0.906 |
| 드리프트 508 | 0.343 | 1.077 | 0.364 | 0.868 |
| 위상 다름 103 | 0.858 | 3.130 | — | 0.734 |

세 가지가 확정됐다.

1. **격자·스무딩 탓이 아니다.** 다른 루트 611개 중 기준선이 0.8 m를 넘는 건 **0개**.
2. **드리프트 508 중 342(67%)는 chamfer < 0.4 m** — 경로가 집합으로는 겹치는데 Fréchet만
   3.3배 크다 = **어긋남이 몇 지점에 몰려 있다**(코너 잘라먹기).
3. **우리가 GT보다 짧다.** 길이비 < 0.95가 501개, **우리가 더 긴 경우는 0개.**
   회전량은 GT의 1/4 (34 vs 130 °/m).

→ **A\*는 최단경로, 사람은 아니다.** 이게 근본 원인이다.

### 파라미터 탐색 (30여 조합)

`refine_radius`를 키우면 루트가 GT에 가까워진다 — 사람은 벽을 스치지 않고 복도 가운데로 걷는데
A*는 장애물 코너를 스친다. **내 첫 가설(키우면 나빠진다)이 반대였다.**

```
0.05 63.6%  0.15 71.2%  0.30 76.4%  0.50 78.2%  0.70 69.3%  1.00 38.5%
```

**그런데 충돌을 안 봤다.** `refine 0.50`은 하드 충돌이 18 → 247 에피소드로 폭증하고
`no_collision` 게이트 통과 씬이 36/45 → 7/45로 무너진다. 추천을 철회했다.

**충돌은 refine이 아니라 스무딩이 만든다** — refine된 웨이포인트는 최소 여유 0.071 m인데
스무딩된 궤적은 0.000 m다. 스플라인이 웨이포인트 사이에서 벽을 뚫는다.
→ `waypoint_spacing_m`을 0.8 → 0.2로 좁히면 곡선이 짧게 끊겨 **충돌 18 → 0**.

다른 축: `bezier` > `cubic`(같은 refine에서 기준선 0.438 → 0.399) · `downsample_mode majority`가
r_b 침범을 209 → 113으로 반감 · **A\* 격자를 굵게 하면 안 된다**(0.25에서 충돌 159ep, 0.30에서 405ep).

### 확정 baseline = `reproduce_v1`

| | legacy | reproduce_v1 |
|---|---|---|
| chamfer 중앙값 | 0.191 | **0.167** |
| Fréchet 중앙값 (p90) | 0.553 (1.446) | **0.480 (1.295)** |
| 같은 루트 | 73.1 % | **78.8 %** |
| 하드 충돌 ep | 18 | **0** |
| r_b 침범 ep | 427 | **113** |
| 게이트 통과 씬 | 36/45 | **45/45** |
| 쓸 수 있는 ep | 1,649 | 1,700 |

바뀐 값 4개: `refine_radius` 0.2→0.3 · `waypoint_spacing_m` 0.8→0.2 ·
`smooth` cubic→bezier · `downsample_mode` any→majority.

**정직한 단서**: 같은 루트 비율은 +5.7점인데 **절대 거리는 −13%뿐**이다. 비율이 크게 오른 건
0.8 m 경계에 몰려 있던 에피소드가 넘어온 것이다. 그리고 `refine 0.30`은 기준선을
0.334 → 0.399로 **나쁘게** 만든다 — "기준선 대비 배수 1.66 → 1.20" 개선의 일부는 분모가
나빠진 것이므로 그 숫자로 성과를 말하면 안 된다.

### 파라미터 천장

**시험한 30여 설정이 전부 73~79%에 있다.** 남는 21%는
- 4.5% 위상 다름 — GT가 36% 더 긴 길로 돌아갔다. 최단경로 기반으로는 원리적으로 불가.
- 나머지 — A*/사람 차이.

더 올리려면 **비용 함수를 바꿔야 한다**(회전 벌점, 문 통과 선호, 벽 이격 비용). 새 코드다.

---

## 설정 관리 — `path_profiles.py` 신규

**문제**: 경로 계획 설정이 세 곳에 흩어져 있었고 **서로 달랐다.** 실제로 쓰던 값은 코드
기본값이 아니라 `reports.md`에 적힌 커맨드 문자열이었고, 그래서 부를 이름이 없었다.

| | `esdf_utils.py` 상수 | `03` argparse | `reports.md` 커맨드 |
|---|---|---|---|
| `r_b` | 0.25 | 0.25 | **0.20** |
| `refine_radius` | — | 0.10 | **0.20** |

`camera_profiles.py`와 같은 규약으로 `path_profiles.py`를 만들어 `legacy` / `reproduce_v1`에
이름을 줬다. `03`·`03f`에 `--path_profile`을 붙였고, **개별 인자를 명시하면 그 값이 이긴다**
(04의 `--preset`과 같은 방식). 프로파일을 안 주면 종전 동작 유지.

검증: `--path_profile reproduce_v1`로 45씬을 돌려 실험 `c3`와 **모든 수치와 기록된 `params`
11개가 완전히 일치**함을 확인했다.

참고로 실행 결과 json에는 원래부터 `params` 11개가 남는다 — 개별 실행 provenance는 있었고,
없던 것은 **이름**이었다.

---

## 고친 기존 버그

**`esdf_utils.py:742` `heading_alignment()`** — 조기 반환이 `median_deg`만 담고
`p90_deg`·`max_deg`를 빼먹었다. 경로가 퇴화한 에피소드가 하나라도 섞이면 호출부
(`03:875`)가 `KeyError`로 죽고, 통계 계산이 json 저장보다 앞이라 **씬 전체 결과가 날아갔다.**
vln_pe 61씬 중 4씬(258 에피소드)이 이걸로 유실됐다. 세 키를 항상 채우게 고쳐 복구했다.

---

## 이번에 반복한 실수 — 전부 "한 축만 보고 판단"

1. **텍스처 없이 톤맵을 측정했다** (앞선 문서 참고). 파이프라인이 쓰는 loader를 우회했다.
2. **격자 경계를 세 번 넘겼다** — 톤맵 iso, `crush`, 그리고 `refine_radius`(0.30이 상한이었다).
   최적값이 경계에 있으면 그 측정은 끝난 게 아니다.
3. **`refine_radius`를 충돌 없이 추천했다** — "계획 성공률 96.7% 유지"를 부작용 없음의 근거로
   댔는데, 그 값은 "경로가 나왔는가"만 보고 "벽을 뚫는가"는 다른 컬럼(`hard_collision_eps`)이었다.
4. **sentinel 사고 2연발** — 부분 실행(`--max_episodes 3`)을 완료로 착각해 한 씬이 3/25만
   남았고, 그걸 고치려 넣은 판정이 과해 완성된 60씬을 재렌더하기 시작했다.
   **기록된 의도가 아니라 실제 산출물 개수를 세야 한다.**

일반 원리: **스윕 결과는 목표 지표 하나로 판단하지 말고, 그 지표를 올리면 나빠질 수 있는
축(충돌·게이트·기준선)을 같이 표에 놓는다.**

---

## 커맨드

전부 `scripts/dataset_converters/gs_vlnpe/reports.md`의 「05~06 · 설정 프로파일」 절에 있다.
이 문서는 **왜 그렇게 정했는지**를, `reports.md`는 **어떻게 돌리는지**를 담는다.

핵심 네 줄만 옮겨 둔다.

```
# 설정 확인
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/path_profiles.py --diff legacy reproduce_v1

# A. 전수 재렌더 (61씬 · 5.2시간 · 116 GB)
timeout --signal=KILL 86400 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/05_export_vlnpe.py --dataset vln_pe --n_scenes 0 --out_root data/InternData-N1-v0.5-mini/vln_pe_render --log_dir logs/gs-vlnpe

# A 검증 (구조 게이트 + GT 대조)
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/05b_verify_vlnpe.py --render_root data/InternData-N1-v0.5-mini/vln_pe_render/traj_data/r2r --gt_root data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r --scenes all --require_complete --negative --deep --log_dir logs/gs-vlnpe

# B. 경로 재현 + GT 대조 (CPU · 45씬)
timeout --signal=KILL 21600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/03f_reproduce_batch.py --dataset vln_pe --n_scenes 0 --path_profile reproduce_v1 --work_dir scripts/dataset_converters/gs_vlnpe/logs --log_dir logs/gs-vlnpe/03f_v1
```

## 남은 것

| | |
|---|---|
| 비용 함수 개조 | 회전 벌점 등. 78.8%를 더 올리려면 이것뿐 |
| `r_b`·`h_nav_ratio` 스윕 | 미측정. ESDF를 다시 만들어야 해서 값당 15분 |
| 충돌 기준 육안 확인 | ESDF는 0.05 m 격자 근사 — "여유거리 0"이 정말 벽 안인지 안 봤다 |
| 15씬 | `01`이 vln_n1 GT를 읽어서 제외됨. vln_pe만으로 `scene_meta`를 쓰면 편입 가능 |
| Phase C 렌더 | `reproduce_v1` 경로로 렌더 → `vlnpe_writer.py`로 저장 |
