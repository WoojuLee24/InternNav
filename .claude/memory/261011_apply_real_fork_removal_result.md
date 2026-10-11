# apply_real 포크 해소 (A/B/C) — 2026-10-11

## 목적
`gs_vlnpe/apply_real/`은 `gs_vlnpe/`의 **포크**였다. 라이브러리 5개가 통째로 복사돼 있어
한쪽만 고치면 다른 쪽이 조용히 뒤처진다. 실제로 이미 그렇게 됐다.

## 무엇이 달라졌나

**−2,861줄 / +41줄.** `apply_real/` 파일 18개 → 13개.

| 삭제한 사본 | 줄 | 원본 대비 차이 |
|---|---|---|
| `geometry_utils.py` | 689 | **0줄** |
| `usdz_scene_utils.py` | 43 | **0줄** |
| `dataset_utils.py` | 283 | 1줄 — `parents[3]` vs `[4]`, `__main__` self-check 전용 |
| `viz_utils.py` | 370 | 1줄 — `LOG_ROOT`, **어디서도 참조 안 되는 죽은 상수** |
| `esdf_utils.py` | 1,476 | 4줄 — 아래 버그 수정분 |

대신 `apply_real` 스크립트 15개(하위 `lidar/`·`format_validation/` 4개 포함)의 `sys.path`에
상위 `gs_vlnpe/`를 추가했다. `insert(0, ...)`는 **나중에 넣은 것이 앞**이므로
`apply_real` 전용 모듈(`camera_profiles` 등)이 계속 우선한다.

## 포크가 이미 낳은 버그

커밋 `ad6a2d9`가 고친 `esdf_utils` KeyError — 조기 반환이 `median_deg`만 채워서 호출부
(`03_sample_gt_paths.py:875`의 `['p90_deg']`)가 죽고 **씬 전체 결과가 저장되지 않던**
문제(vln_pe 61씬 중 4씬 실측) — 가 `gs_vlnpe/` 쪽에만 반영돼 있었다. 노은역 경로는
여전히 `return {'median_deg': float('nan')}`를 쓰고 있었다. 이번에 사본을 지우면서 닫혔다.

## 검증 — `git worktree`로 변경 직전 커밋을 꺼내 같은 명령 대조

```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/03_sample_gt_paths.py --scene noeun_station_mid --mode random --num_episodes 20 --esdf_dir scripts/dataset_converters/gs_vlnpe/apply_real/esdf --out_dir <OUT> --log_dir <LOG>
```
```
timeout 5400 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/04_render_obs_isaac.py --scene noeun_station_mid --mode random --num_episodes 1 --camera d455_nominal --out_dir <OUT> --log_dir <LOG>
```
⚠️ `04`의 `--out_dir`는 **입력도 겸한다** — `paths/`를 거기서 읽는다. 별도 디렉토리에 돌리려면
`esdf`·`paths`·`scene_meta`를 심볼릭 링크해 둬야 한다.

| 게이트 | 결과 |
|---|---|
| `03` paths JSON | **바이트 동일** |
| `03` report.html + 산출물 4개 | **4/4 바이트 동일** |
| `04` rgb 285장 | **285/285 바이트 동일** |
| `04` depth 285장 | **285/285 바이트 동일** |
| `04` `intrinsic.npy` · `camera_config.json` | **바이트 동일** |
| `04` report.html | 출력 경로 문자열 한 줄만 다름 |
| import 해결 (AST, 17파일 41심볼) | 전부 `gs_vlnpe`, `camera_profiles`만 `apply_real` |

커버리지: `03`이 `geometry_utils`·`esdf_utils`·`dataset_utils`·`viz_utils`를,
`04`가 거기에 `usdz_scene_utils`를 더해 다섯 개를 모두 탄다.

## 알아둘 것 (겪은 것들)

- **`simulation_app.close()`가 20분 넘게 멈춘다.** 285/285 프레임을 다 저장한 뒤였다.
  reports.md가 `timeout --signal=KILL`을 쓰라고 한 이유가 이것 — **성공 판정은 종료코드가
  아니라 산출물로** 한다(0바이트 파일 0개 + 마지막 프레임 존재).
- ⚠️ `pkill -f "04_render_obs_isaac"`는 **그 문자열을 명령줄에 담은 대기 셸까지 같이 죽인다.**
  실제로 대기 중이던 다음 렌더 작업이 함께 사라졌다.
- 8월 산출물과 비교하면 **rgb만 다르고 depth는 같다**(평균차 0.4~0.8/255). 이건 변경 전
  코드에서도 나므로 이번 변경과 무관하다 — `ad6a2d9`의 톤맵 프로파일 도입 때문으로 보인다.
  **오늘 두 번 돌린 결과끼리는 rgb도 바이트 동일**하므로 렌더러는 결정론적이다.

---

# D단계 — 스크립트 사본 4개 제거 (−2,382줄)

## 또 드리프트가 있었다
`apply_real/03_sample_gt_paths.py`는 `ad6a2d9`가 추가한 **`--path_profile`을 못 받고 있었다**.
A/B/C의 `esdf_utils` 버그에 이어 두 번째다.

## 차이는 셋뿐이었고 전부 optional argument 로 흡수했다

| 차이 | 어느 파일 | 흡수 방법 | 생략하면 |
|---|---|---|---|
| `DEFAULT_OUT_DIR`/`LOG_DIR` | 03·03b·03c·03d | 호출부가 이미 `--out_dir`/`--log_dir` 전달 | — |
| GT 대신 생성 경로를 기준으로 | 03b·03c·03d | `--mode {reproduce,random}` | `reproduce` = 종전 |
| 점유격자를 스캔 관측 영역으로 제한 | 03b·03c | `--scan_coverage` | off = 종전 |

분기는 exhaustive `if/elif/else` + `assert False` 로 적었다(guideline 6).
`--scan_coverage`는 3DGS 스캔의 구멍 대응이라 노은역에서만 켠다 —
**예전 `apply_real` 사본은 이걸 무조건 적용**했으므로 노은역 커맨드에 플래그를 추가해야 한다(README 갱신함).

## 검증 — 두 경로 모두 전후 대조

```
# mp3d 기본 경로 (회귀)
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/03b_verify_reproduction.py --scene 17DRP5sb8fy --num_episodes 3 --out_dir scripts/dataset_converters/gs_vlnpe/logs --log_dir <LOG>
# 노은역 (신규 인자)
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/03b_verify_reproduction.py --scene noeun_station_mid --mode random --scan_coverage --num_episodes 5 --out_dir scripts/dataset_converters/gs_vlnpe/apply_real --log_dir <LOG>
```

| 경로 | 03b | 03c | 03d | stdout |
|---|---|---|---|---|
| mp3d 기본 (패치 전/후) | 20/20 동일 | 15/15 동일 | 4/4 동일 | 출력 경로 문자열만 다름 |
| 노은역 (구 사본 vs 병합본) | 20/20 동일 | 15/15 동일 | 4/4 동일 | **완전 동일** |

`gs_vlnpe/03`이 `apply_real/03`의 노은역 paths JSON을 **바이트 동일**하게 재현하는 것도 확인했다.

## 최종
`apply_real/*.py` **18개 → 9개**. A/B/C/D 합계 **−5,243줄**.
남은 9개는 진짜 다른 것들이다 — `00`/`01`/`02`(usdz ROI·`--geometry usd`),
`04_render_obs*`(NuRec GS), `04c`, `05_export_vlnce`(새 경로 저장), `camera_profiles`(vln_ce rig), `03e`.

## 남은 것 (제안했으나 미실행)
- `03b`~`03g` 이름 정리(`study_` 접두사) — 파이프라인과 일회성 실험 구분.
- `gs_vlnpe/README.md` 신규 — 지금 `reports.md` 400줄이 커맨드까지 겸한다.
