# `esdf_utils.py` + `02_build_freemap_esdf.py` (M1.2) — 완료

씬 mesh를 occupancy로 만들어 저장하고, 그 맵이 정상인지 검증한다. NavDP 논문 §3 Trajectory
Generation의 A\* 직전까지를 구현했다.

> 좌표 규약·과거 오진은 `understanding_gs_vlnpe_fpv_bev_geometry.md` 참고.

## 처리 단계

```
mesh -> 표면 점군(3M) -> (1) occupancy (Nx,Ny,Nz) bool     ** npz로 저장, h_b 무관 **
                      -> (2) obstacle_2d (Ny,Nx) bool     ┐ 밴드 (floor+h_nav, floor+h_b]
                      -> (3) esdf (Ny,Nx) float32 [m]     │ h_b 의존 →
                      -> (4) navigable (Ny,Nx) bool       │ 03이 매번 도출 (수 ms)
                      -> (5) nav_coarse (Ny/4,Nx/4) bool  ┘ A* 용 0.2 m
```

## `h_nav` / `h_obs`

로봇을 높이 `h_b`·반경 `r_b` 원통으로 보면 충돌 구간은 바닥 위 `(h_nav, h_obs]` 밴드다.
`h_nav` 아래는 밟고 넘는 지면(문턱·러그), `h_obs`(= `h_b`) 위는 밑으로 통과(천장·선반).
`h_b`에 의존하는 이유가 여기 있다 — 키 0.25 m 로봇은 의자 밑을 지나가지만 1.25 m 로봇은 못 지나간다.

## 파라미터 (논문 §3 + 실측)

| | 값 | 근거 |
|---|---|---|
| voxel | 0.05 m | 논문 |
| A\* 격자 | 0.2 m (factor 4) | 논문 |
| `h_nav` | 0.15 m | 0.05~0.20에서 결과 둔감 |
| `r_b` | 0.25 m | 논문 `r_b`, InternNav `agent_radius` 기본값 |
| 표면 샘플 | 3M | 씬당 0.7초 |

**`truncate`는 값 clipping이 아니라 navigable 집합에서 제외**다(논문 원문 오독 주의).
`d_safe = 0.5`는 M2 critic 라벨용이지 경로 필터가 아니다.

## 실행

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/02_build_freemap_esdf.py --scene 17DRP5sb8fy
```

`--geometry {obj,usd,both}` 기본 **obj**. 논문이 raw scene mesh를 썼고 GT도 그 mesh에서 렌더됐다.
USD는 표면이 동일하고 collision Plane 2장만 더 있는데 최종 navigable 차이가 0.4%(103셀)뿐이다.
Stage 2(Isaac)로 넘어갈 때 `usd`로 바꾼다.
`--scene_meta_dir`은 읽기 루트를 출력과 분리(negative test용).

## 출력

| 경로 | 내용 |
|---|---|
| `esdf/<scene>.npz` | `occupancy`(bool 3D), `origin`, `voxel_size`/`h_nav`/`r_b`/`floor_z`, `ref_h_b`, `nav_mask_ref`/`esdf_ref`(참조 2D), (both일 때) `occupancy_alt` |
| `esdf/<scene>.json` | 같은 내용의 사람이 읽는 메타 + 검증 결과 |
| `logs/gs-vlnpe/02_build_freemap_esdf/<scene>/report.html` | 맵 단계별 blink 4-state |

## 검증

1. **GT 궤적 clearance** (유일한 독립 검증) — 에피소드마다 그 로봇 키로 맵을 만들어 궤적 clearance를 잰다
2. sanity — 장애물 셀 `esdf==0`, 자유공간 `esdf>0`, occupancy가 bounds 안
3. `.ply` obstacle 겹침 — **판정에 쓰지 않음**. 그 점들은 mesh 표면 위 0.0000 m라 순환논증이고,
   바닥 슬라이스라 우리 밴드와 높이가 달라 겹침률이 낮은 게 정상

### 실행 결과

| 씬 | grid | 점유 | clearance median / worst min | npz | 판정 |
|---|---|---|---|---|---|
| 17DRP5sb8fy | 328×166×57 | 7.4% | 0.495 / 0.224 | 0.1 MB | PASS |
| s8pcmisQ38h | 477×200×245 | 2.9% | 0.680 / 0.269 | 0.5 MB | PASS |

**Negative test** (게이트가 항상 통과만 내지 않음을 보장):

| 조건 | 결과 |
|---|---|
| `--h_nav -0.1` (바닥을 장애물에 포함) | clearance 0.000으로 붕괴 → FAIL, exit 1 |
| `--h_nav 1.0` (로봇 키보다 큼) | 설정 불가로 사전 차단 → exit 1 |

## 구현 중 고친 것

**1. clearance 기준선을 단일 값으로 잡았다가 2번째 씬을 오판했다.** `17DRP5sb8fy`의 0.500(논문
`d_safe`와 일치)을 기준으로 ±0.05를 걸었더니 `s8pcmisQ38h`(0.680)가 FAIL로 나왔다. **6개 씬 실측**:

```
median : 0.461 / 0.495 / 0.501 / 0.602 / 0.650 / 0.680   (넓은 씬일수록 큼)
min    : 0.206 ~ 0.320
```

→ 범위 판정으로 변경: median ∈ (0.30, 1.00), min > 0.15. 실측 범위(0.40~0.80) 밖이면 경고만.
**상한이 필요한 이유**: 밴드가 비면 장애물이 사라져 clearance가 무한대가 되므로, 상한이 없으면
negative test가 통과해 버린다.

이건 이미 기록해둔 교훈("씬 1개에서 잡은 임계값을 일반화하지 말 것")을 **또 어긴 것**이다.

**2. `detect_floor_levels`가 쓸 수 없는 상태였다.** 점유 개수만 세니 `s8pcmisQ38h`에서 13개가 나오고
1순위가 2.64 m로 틀렸다. **점유 개수 × 위쪽 자유 비율**("위가 비어 있는 넓은 수평면")로 바꿔
두 씬 모두 1순위가 GT 바닥과 일치하게 했다. 단 Stage 1에서는 쓰지 않는다 — 01이 `floor_z`를
정확히 주고, GT 전 에피소드가 바닥 하나를 공유한다(std ~1e−8). Stage 3(GS-map)용 fallback이다.

**3. USD/obj 비교 게이트를 개수 비교 → 고립 voxel 판정으로 바꿨다.** 표면 샘플링이 무작위라 두 격자는
항상 얇은 껍질만큼 어긋난다(XOR 6,765칸 = 0.218%). 진짜 신호는 **obj에만 있고 USD 표면에서 떨어진**
voxel이다. seed를 바꿔 검증: obj-only 고립은 0~5로 흔들리고(노이즈), usd-only 고립은 831~836으로
일정하다(= collision Plane, 실제 차이).

## self-check

`python esdf_utils.py` — 5개 게이트: 순수 로직 / 좌표 왕복 / **GT 궤적 clearance** /
바닥 탐지 fallback / USD vs obj 지오메트리 누락.

## 참고

- Artifact: (아래 링크)
- 설계: `docs/execution-staged.md` M1.2 절
