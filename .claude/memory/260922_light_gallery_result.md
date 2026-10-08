# 조명 모드별 렌더 갤러리 (vln_ce / vln_pe / 노은역)

작성 2026-09-22 · 브랜치 `feature/embaug_v0.1`
Artifact: https://claude.ai/artifact/RWYPkWwgKhJhBw1y9fhKZG
로컬: `logs/gs-vlnpe/04f_light_gallery/{vln_ce,vln_pe,noeun}/report.html`

## 새 파일

`scripts/dataset_converters/gs_vlnpe/04f_light_gallery.py` (기존 04 계열 미수정)
- 컨텍스트마다 해상도가 달라 `Camera` 재생성이 안 되므로 **프로세스를 분리**(`--context` 3회 실행)
- 조명 6모드 공통: `no_light` / `ambient_only_10` / `dome_2M` / `three_light_raise0.2`(추종) /
  `three_light_raise0.5`(추종) / `camera_light`(헤드램프·완전추종). `film_iso` 100 고정(조명만 변수)

## 실측 (17DRP5sb8fy · 4시점 · film_iso 100)

**vln_pe (256², GT=Isaac 프로덕션 3-light)** — 정합 눈으로 확인 완료
| mode | lum | std | sat |
|---|---|---|---|
| GT | 133.36 | 42.58 | — |
| no_light | **0.00** | 0.00 | 0 % |
| ambient_only_10 | 184.63 | 43.35 | 0 % |
| dome_2M | 72.40 | 44.38 | 0.14 % |
| three_light_raise0.2 (추종) | 100.91 | 88.34 | 0 % |
| three_light_raise0.5 (추종) | 86.31 | 85.98 | 0 % |
| camera_light (추종) | 126.50 | 67.05 | 0 % |

- **PBR mesh는 `no_light`에서 완전히 검다(lum 0)** — GS(노은역 191)와 정반대. 자산 종류가 답을 바꾼다는 직접 증거.
- 추종 조명은 **std가 2배**(88 vs GT 42.6) — 천장이 검게 죽고 바닥만 밝은 "사진 위 손전등" 패턴이 눈에 보인다.
- 헤드램프는 좌측이 통째로 그림자로 죽는다.

**노은역 (480×270, GT 없음)** — 260921 결과와 일치(추종 조명 = 무조명, dome만 반응)

## ⚠️ vln_ce는 Isaac 재렌더를 싣지 못했다

`pose.125cm_0deg` 4×4를 mesh world c2w로 옮기는 변환을 **확정 못 함**. 렌더해 보니 같은 씬의
**다른 방향**이 나왔고 일부는 검은 화면(카메라가 지오메트리 안).

**내 실수 2건**:
1. vln_ce와 vln_pe의 xyz 범위가 겹치는 것만 보고 "같은 world 프레임"이라 단정했다.
   범위 겹침은 축 대응의 증거가 아니다. c2w냐 w2c냐, Y-up이냐 Z-up이냐 둘 다 미확정이었고
   **"높이 1.25 고정"·"전진 0.25 m"는 두 해석 모두를 만족해서 구분에 못 쓴다**.
2. 후보 12종(c2w/w2c × cv/gl × world축 3종)을 SSIM으로 가리려 했는데 **지표에 판별력이 없었다** —
   **공백(백색) 이미지가 0.578로 1위**. 평탄한 이미지가 부드러운 GT 대비 높은 SSIM을 받는다.
   → **정합 판별에 SSIM 단독 금지.** 특징점 매칭이나 depth→world 재투영 같은 기하 기반 판정을 쓸 것.

VLN-CE는 조명 모드가 **하나뿐**(`habitat-sim` 기본값 `NO_LIGHT_KEY`)이라 스윕 대상이 아니므로,
GT 프레임만 비교 기준으로 실었다. 틀린 그림을 싣느니 빼는 쪽을 택했다.

## 미측정 / 한계

- 4시점 · 1씬(17DRP5sb8fy) · 노은역 중간층만.
- 세 컨텍스트의 **해상도·화각이 다르다**(256² / 480×270 / 640×480) → 컨텍스트 간 밝기 절대값 비교 금지.
- `film_iso` 100은 프로덕션 권장값(70)과 다르다 → 프로덕션 산출물과 밝기 직접 비교 금지.

## 커맨드

```
for C in vln_ce vln_pe noeun; do timeout --signal=KILL 3000 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04f_light_gallery.py --context $C --n_frames 2; done
```
`exit=137`은 `simulation_app.close()` 중 SIGKILL로 정상 — 산출물은 `main()` 반환 시점에 저장 완료.
