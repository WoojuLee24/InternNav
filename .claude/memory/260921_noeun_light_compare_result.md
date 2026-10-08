# 노은역 GS 씬 조명 옵션 비교

작성 2026-09-21 · 브랜치 `feature/embaug_v0.1`
Artifact: https://claude.ai/artifact/7mraFYXQDy8hfsxXcK2bLt (§05)
로컬 리포트: `logs/gs-vlnpe/apply_real/04c_light_explorer/noeun_station_mid/report.html`

## 결론

**노은역(NuRec Gaussian splat)에서 추종 조명은 효과가 나쁜 게 아니라 "존재하지 않는다".**
카메라 추종 3-light도 RTX 헤드램프도 **조명을 끈 것과 구분되지 않는다**(잡음 바닥 이내).
유일하게 화면을 바꾸는 건 정적 `dome_2M`이고, 그것도 GS가 아니라
`expose_collision_meshes_for_rendering`이 visible로 만든 **충돌 mesh 1개**가 반응한 것이다.

## 새 파일

`scripts/dataset_converters/gs_vlnpe/apply_real/04c_light_explorer.py`
- 기존 `04_render_obs_isaac.py`는 미수정. 본가 `gs_vlnpe/04c_light_explorer.py`와 같은 구조
  (Isaac 1회 부팅 → 같은 프레임을 조명만 바꿔 재렌더)
- **노은역은 GT rgb가 없다** → "정답과의 거리" 불가, 설정끼리 상대 비교만
- `--reference LABEL` : 차이를 재는 기준 (기본 `dome_2M`)
- `--self_check` : 기준 설정을 **2회** 더 렌더해 잡음 바닥 확보

## 실측 (noeun_station_mid · 2ep × 4frame = 8frame · film_iso 100 고정 · `--reference no_light`)

| config | 추종 | 평균 휘도 | 포화 ≥250 | no_light 대비 평균차 | max |
|---|---|---|---|---|---|
| no_light (기준) | — | 191.03 | 0.95 % | 0 | 0 |
| **no_light 재렌더 ①** | 대조군 | 191.04 | 0.96 % | **0.231** | 85 |
| **no_light 재렌더 ②** | 대조군 | 191.03 | 0.94 % | **0.699** | 143 |
| ambient_only 10 | — | 191.14 | 0.94 % | 0.783 | 139 |
| three_light raise 0.2 | 카메라 | 191.08 | 0.94 % | 0.738 | 155 |
| three_light raise 0.5 | 카메라 | 191.05 | 0.95 % | 0.235 | 86 |
| camera_light (헤드램프) | 카메라 | 191.12 | 0.94 % | 0.758 | 143 |
| **dome_2M** (현재 기본값) | 정적 | **192.70** | **1.67 %** | **1.953** | 246 |

추종 조명 3종(0.235 / 0.738 / 0.758)이 대조군 재렌더 두 값(0.231 / 0.699)에 **정확히 포개진다**.
휘도도 191.03~191.14(0.11 DN 안), 포화도 0.94~0.96 %(0.02 %p 안).
`dome_2M`만 휘도 +1.67 DN, 포화 0.95→1.67 %(1.8배).

## 방법론 교훈 — 대조군 없이는 해석 불가였다

첫 실행은 `--reference dome_2M`으로 돌려서 **모든 설정이 mean 1.94~2.28 / max 244~246**로 나왔고,
나는 이를 "전부 렌더 잡음"이라고 **잘못 추정**했다. `--self_check`(동일 설정 재렌더)를 넣어보니
잡음 바닥은 mean 0.216 / max 53 — 조명 차이가 그보다 9~10배 커서 **진짜 차이가 맞았다**.
기준을 `no_light`으로 바꾸고 나서야 "추종 조명 = 무조명"이 드러났다.
→ **기준 선택이 결론을 가린다. 잡음 바닥(대조군)을 먼저 재고, 기준은 답하려는 질문에 맞춰 고른다.**

## 새로 발견 — 렌더 파이프라인의 2프레임 주기

동일 설정 재렌더 잡음이 **0.231과 0.699 두 값으로 갈린다**(두 번의 독립 실행에서 재현).
설정 실행 순서의 홀짝과 일치 → 디노이저/TAA류 시간 누적으로 추정, 원인 미확인.
**함의**: 재렌더 자체 검증은 반드시 **2회 이상** 붙여야 한다(1회면 기준과 같은 패리티에만 떨어져
잡음 바닥을 3배 과소평가).

## 내 버그 1건 (수정함)

새 파일의 진행 출력이 `vs {REFERENCE_LABEL}`(모듈 상수)을 찍어서 `--reference no_light`로
돌려도 "vs dome_2M"으로 표시됐다. `{reference}`(인자 값)로 수정. 수치 자체는 정상.
본가 `04c_light_explorer.py`의 `/rtx/useViewLightingMode`를 켜기만 하고 끄지 않는 문제는
새 파일에서 설정마다 명시적으로 다시 거는 방식으로 처음부터 회피했다.

## 미측정

- **8프레임 · 2에피소드 · 중간층 한 층**만 쟀다.
- 노은역엔 GT rgb가 없어 "어느 설정이 실물과 가장 닮았나"는 답 못 함. 설정 간 동일/상이만 답했다.
- **DomeLight 세기 스윕 안 함**(2M만). 3-light를 2M급으로 올리면 달라질 수 있으나
  그건 더 이상 "VLN-PE 원본 레시피"가 아니다.
- 충돌 mesh가 RGB 화면의 어느 영역인지 diff 이미지로 특정하지 않았다(report.html에 이미지는 있음).

## 커맨드

```
timeout --signal=KILL 3000 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/04c_light_explorer.py --scene noeun_station_mid --num_episodes 2 --n_frames 4 --reference no_light --self_check
```
