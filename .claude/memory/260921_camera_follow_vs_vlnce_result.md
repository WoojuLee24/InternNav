# 카메라 추종 렌더링 vs VLN-CE 고정 리그 — 실측 비교

작성 2026-09-21 · 브랜치 `feature/embaug_v0.1`
Artifact: https://claude.ai/artifact/7mraFYXQDy8hfsxXcK2bLt

## 질문

"렌더링할 때 카메라가 궤적/로봇을 따라가게 하면 더 현실감 있다"는 주장을
기존 VLN-CE 렌더 방식과 비교해라.

## 이전과 달라진 점

지금까지 `gs_vlnpe` 리포트는 **조명·톤맵**(ambient, film_iso, tonemap op)만 비교해 왔다.
이 문서는 처음으로 **카메라 포즈 생성 방식** 축을 분리해 실측했다.

## 레포 안의 세 방식

| | A 고정 리그 | B 궤적 추종 | C 로봇 추종 |
|---|---|---|---|
| 어디 | `vln_ce/traj_data/r2r` `pose.*` · `vlnce_discretize.py` | `geometry_utils.py:253 synthesize_action_poses` · `04_render_obs_isaac.py:596` | `vln_pe/traj_data/r2r` `camera_position`+`camera_orientation` |
| 04 모드 | — | `--mode reproduce` | `--mode gt_replay` |

## 실측값 (오늘 parquet 직접 읽음)

A: `vln_ce` r2r 120 ep / 5,946 frames · C: `vln_pe` r2r 200 ep / 17,004 frames

| 자유도 | A | B | C |
|---|---|---|---|
| 높이 z | 에피소드 내 p2p **0.000 m** (p95 0.60 m는 층이동 계단식 점프) | `floor_z+h_b` 상수 | σ 0.026 m, p1–p99 −0.073…+0.053 m |
| 하향 pitch | p2p **0.00000°** (리그 0/15/30/45°) | GT 상수 1개 | 평균 18.26° σ1.76°, p1–p99 14.43…23.67° |
| roll | 절대 최대 **0.00000°** | 식에 항 없음 | σ 2.26°, p1–p99 −4.21…+4.39° |
| yaw | 15° 격자 (잔차 max 0.0007°) | 경로 접선 중앙차분 | 연속, 프레임간 평균 6.57° max 47.3° |
| 프레임간 이동 | 0.2501 m 고정, **38.5 %가 0 m**(제자리 회전) | 기본 0.035 m (GT 맞춤 0.14) | 평균 0.135 m, p1–p99 0.012…0.238, 0 m 프레임 없음 |
| 카메라 원점 | 회전축 위 | 회전축 위 | 몸통보다 0.2 m 앞 |

## 결론

1. **차이는 부드러움이 아니라 자유도다.** A의 높이·pitch·roll은 근사적 고정이 아니라 **정확히 0 변화**다.
2. **"궤적 추종"(B)은 절반만 푼다.** yaw·위치만 연속이고 높이·pitch는 상수, roll은 항 자체가 없다.
   평면도에서는 B와 C가 같아 보이고, **측면도에서 갈린다.**
3. **train ≠ eval 갭이 지금 존재한다.** 학습은 `vln_ce` 리그 2종(`r2r_125cm_0_30%30, r2r_60cm_15_15%30`)만
   읽는데 Isaac 평가는 C 분포다 → pitch 9.2°, 높이 12.6 cm, roll ±4.4°가 학습 데이터에 한 프레임도 없다.
   BEV가 depth unprojection 하류라 pitch/높이에 특히 민감. CLAUDE.md "train==eval" 규칙이 이 축에서 깨져 있음.

## 측정하지 않은 것 (중요)

- **B vs C 렌더 이미지 화질 차이(SSIM/LPIPS)는 안 쟀다.** 오늘 잰 건 포즈 분포뿐.
- "캡처 궤적에서 멀어지면 재구성 품질이 떨어진다"는 기전은 일반론이며 이 레포에서 확인된 바 없다.
  `reports.md`의 SSIM 0.876/0.823은 조명·톤맵 스윕 결과이지 카메라 경로 비교가 아니다.
- 모션 블러·롤링 셔터는 별개 문제 — 04는 정지 상태 렌더라 추종 여부와 무관하게 블러가 없다.

## 다음 실험 (04에 이미 두 모드가 있음)

C(기준선):
```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04_render_obs_isaac.py --dataset vln_pe --scene 17DRP5sb8fy --mode gt_replay --light ambient_only --rtx_ambient 10.0 --film_iso 70 --out_dir scripts/dataset_converters/gs_vlnpe/logs
```
B(비교):
```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04_render_obs_isaac.py --dataset vln_pe --scene 17DRP5sb8fy --mode reproduce --frame_step_m 0.14 --light ambient_only --rtx_ambient 10.0 --film_iso 70 --out_dir scripts/dataset_converters/gs_vlnpe/logs
```
- `--mode gt_replay` GT 포즈 시퀀스 그대로 렌더 / `--mode reproduce` 03 재현 경로를 합성 포즈로 렌더
- `--frame_step_m 0.14` GT 프레임 밀도(0.141 m)에 맞춤. 기본 0.035는 4배 촘촘해져 `max_step=200`에 90%가 잘림
- 출력 `logs/gs-vlnpe/04_render_obs_isaac/<scene>_vlnpe[_reproduce]/report.html`, 성공 판정은 report.html 존재로

그 다음: B에 높이/pitch/roll 변조를 넣어 C 분포를 합성 경로에서 재현.
`synthesize_action_poses` 본문은 건드리지 말고 **새 파일에 별도 함수 + config 기본값 off**
(CLAUDE.md §1 guard clause+위임, §2 기본동작 보존). 현 브랜치 embaug 작업과 동일 문제 —
지금 embodiment는 리그 5종 중 고르는 **데이터셋 선택 축**인데, 이를 **프레임 단위 연속 축**으로 바꾸는 것.
