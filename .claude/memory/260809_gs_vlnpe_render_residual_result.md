# ambient light 이후 남은 GT-렌더 차이 — 원인 분해 결과 (2026-08-09)

사용자 질문: "ambient light가 효과가 좋긴 한데 여전히 gt와 차이가 난다. 원인 확인하고 해결해라"

결론부터: **남은 차이의 대부분은 조명/톤매핑으로 못 고친다.** 렌더 설정을 아무리 바꿔도 얻을 수
있는 최대치가 SSIM +0.034이고, Isaac이 실제로 노출하는 노브로는 +0.003이 한계다. 나머지는
메쉬/텍스처 충실도다. 아래는 그 결론에 도달한 실측 근거 전부.

## 1. 잔차 분해 — 상한을 먼저 측정 (vln_n1 씬1, 18프레임, ambient 6.0)

| 처리 | SSIM | 해석 |
|---|---|---|
| 현재 그대로 | 0.860 | — |
| + 프레임별 최적 gain/bias | 0.894 | **조명·톤 보정으로 가능한 이론적 상한 (+0.034)** |
| + ±3px 최적 시프트 | 0.870 | pose 잔차 기여 (+0.010) |
| + 둘 다 | 0.904 | 남는 `1-SSIM = 0.096`은 메쉬/텍스처 충실도 |

최적 시프트는 전 프레임 (0,0)~(1,1) — **pose는 이미 1픽셀 이내**라 더 짜낼 게 없다.

이 표가 이 문서의 핵심이다. 상한을 먼저 재지 않고 조명 스윕부터 돌았다면 +0.03짜리 여지를 놓고
몇 시간을 더 태웠을 것이다(`feedback_measure_metric_floor` 교훈의 반복 적용).

## 2. 증상: 응답곡선 기울기가 GT보다 가파르다

GT 휘도대별 (렌더 - GT) 평균 밝기차:

| GT 휘도대 | vln_n1 (amb 6.0) | vln_pe (amb 10.0) |
|---|---|---|
| 0~60 (어두움) | **-4.3** | +21.8 |
| 60~150 (중간) | **+19.4** | +36.7 |
| 150~255 (밝음) | **+37.1** | +2.7 |

vln_n1은 "어두운 곳은 더 어둡고 밝은 곳은 더 밝다"(기울기 과다), vln_pe는 반대로 어두운 쪽이
들려 있다. **두 데이터셋의 GT 렌더러가 서로 다르다**는 것과 일관된다(vln_pe GT는 VLN-PE
프로덕션 3-light, vln_n1 GT는 NavDP 파이프라인).

부수 측정:
- 단일 감마 피팅 `GT = 255*(render/255)^0.742` → 적용하면 SSIM 0.860 **-> 0.839로 악화**.
  단일 감마로는 "darks 들기 + brights 누르기"를 동시에 못 한다.
- 렌더만 검고 GT는 안 검은 픽셀(메쉬 구멍/머티리얼): 전체의 **0.36%**, SSIM 기여 **+0.003**.
  눈에 크게 띄지만(1행 바닥의 검은 구멍) 지표상 무시 가능.

## 3. 시도하고 기각한 것 — 전부 실측

### (a) Iray 톤매퍼 하위 노브 — **RTX Real-Time에서 무반응**

Isaac 기본 톤매퍼는 `/rtx/post/tonemap/op = 6`(Iray)이고 증상에 정확히 대응하는 노브가 있다:

- `/rtx/post/tonemap/irayReinhard/crushBlacks` (기본 0.5)
- `/rtx/post/tonemap/irayReinhard/burnHighlights` (기본 0.7)

0.0/0.25/0.75, 0.1/0.3/1.0으로 흔들어도 **SSIM·대역편차가 소수 셋째 자리까지 baseline과 동일**.
RTX Real-Time 경로는 이 하위 파라미터를 읽지 않는다. (Iray/path-traced 렌더러 전용으로 보인다.)

### (b) `filmIso` — 유일하게 듣는 노브지만 순수 게인

vln_n1 6프레임: iso 70/85/100/130 → SSIM 0.852 / **0.859** / 0.856 / 0.836
vln_pe 6프레임: iso 65/75/85/100/110 → 0.806 / 0.809 / **0.810** / 0.807 / 0.804

전 대역을 같이 움직이므로 기울기는 못 고친다. 그래도 편차 합(|어두움|+|중간|+|밝음|)은
vln_n1 58 → 45(iso 85), vln_pe 61 → 42(iso 75)로 줄어 **채택**했다.

### (c) DomeLight + ambient 혼합 — 밝기는 완벽히 맞는데 SSIM이 떨어진다

가설: ambient 단독이면 광원이 없어 GI 바운스가 0이다. 실제 광원을 넣으면 밝은 벽에서 튄 빛이
어두운 구석을 들어올려 기울기가 완만해질 것이다. → **밝기 정합은 가설대로 됐는데 SSIM은 급락.**

| 설정 | 어두움 | 중간 | 밝음 | SSIM |
|---|---|---|---|---|
| ambient 6.0 (대조군) | -5.2 | +17.5 | +36.3 | **0.851** |
| dome 0.15M + amb 3.0, iso 70 | **-3.1** | **-1.7** | **-0.5** | 0.761 |
| dome 0.30M + amb 3.0, iso 70 | +9.5 | +7.0 | +1.5 | 0.738 |

대역편차가 사실상 0인 조합이 SSIM은 0.09 낮다. 원인은 **dome이 GT에 없는 방향성 음영을 만들기
때문**이다 — 대역편차는 평균이라 음영의 분산이 상쇄돼 0으로 보이지만 픽셀 단위로는 어긋난다.

렌더 노이즈 때문이 아니라는 것도 확인했다: 프레임당 스텝을 10 → 60으로 올려도 0.749 → 0.738로
동일(수렴 문제였다면 올라갔어야 한다). 대조군 `ambient_only`가 0.856/0.851로 재현된 것도 확인.

### (d) 슈퍼샘플링 x2 — 오히려 악화

GT가 우리 렌더보다 미세하게 흐리다(라플라시안 분산 GT 231.4 vs 렌더 243.3). 렌더에 가우시안
sigma 0.8을 먹이면 SSIM 0.860 → **0.876**으로 오른다. 그래서 2배 해상도로 렌더 후 INTER_AREA로
축소해 봤더니:

| | iso 85 | iso 100 |
|---|---|---|
| 1x (대조군) | 0.859 | 0.856 |
| 2x SSAA | 0.843 | **0.840** |

오프라인 블러가 얻던 이득이 렌더 단계에서는 재현되지 않는다(오히려 -0.016). 후처리 블러로
지표를 올리는 건 GT에 맞춘 화장이지 렌더 충실도 개선이 아니라 채택하지 않았다.

## 4. 채택한 설정

`04_render_obs_isaac.py`에 `--film_iso` 추가(기본 100 = Isaac 기본값이라 **기존 경로 무변경** —
값이 기본값과 같으면 carb 설정을 아예 건드리지 않는다).

```
# vln_n1
timeout --signal=KILL 2400 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04_render_obs_isaac.py --scene 17DRP5sb8fy --mode gt_replay --light ambient_only --rtx_ambient 6.0 --film_iso 85 --log_dir logs/gs-vlnpe/n1_iso85
# vln_pe
timeout --signal=KILL 2400 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04_render_obs_isaac.py --dataset vln_pe --scene 17DRP5sb8fy --mode gt_replay --light ambient_only --rtx_ambient 10.0 --film_iso 75 --log_dir logs/gs-vlnpe/pe_iso75
```

- `--film_iso`: RTX 톤매퍼 노출(`/rtx/post/tonemap/filmIso`). 낮추면 전체가 어두워진다.
- 나머지 인자 의미는 `reports.md` 참고.

## 5. 신규 파일

`scripts/dataset_converters/gs_vlnpe/04d_tonemap_sweep.py` — Isaac을 한 번만 띄우고 설정 조합을
순회하며 같은 프레임을 반복 렌더하는 탐색 스크립트(조합당 재기동 1~2분 회피). 4개 모드:

| `--mode` | 내용 |
|---|---|
| `tonemap` | Iray 노브(crushBlacks/burnHighlights/filmIso) — 앞의 둘은 무반응 확인용 |
| `light_mix` | DomeLight 세기 x ambient 격자 |
| `flatten` | light_mix에서 찾은 "기울기는 맞고 전체만 밝은" 조합을 iso로 내림 (대조군 포함) |
| `iso` | ambient 고정, filmIso만 정밀 스윕 |

`--supersample N`(N배 렌더 후 축소), `--steps`(프레임당 sim.step, dome 수렴 확인용)도 있다.

## 6. 다음에 이걸 더 밀어붙인다면

남은 0.096은 메쉬/텍스처 쪽이다. 조명을 더 만지는 건 헛수고이고, 손댈 데가 있다면:

- mp3d_pe USD의 텍스처 해상도/MIP 설정 (GT 생성 파이프라인과 동일한지 미확인)
- 메쉬 구멍(전체 0.36%) — 지표엔 안 잡히지만 육안으로는 가장 먼저 보인다
- GT 자체가 jpg 압축된 것이라 고주파가 이미 손상돼 있을 가능성 (원본 png 유무 미확인)
