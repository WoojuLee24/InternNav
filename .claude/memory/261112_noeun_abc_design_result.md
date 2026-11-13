# 노은역 S2 목표 근처(ABC) 데이터 설계 — 토론 + 1 차 실험 결과 (2026-11-12)

6000 Pro(apptainer) 서버. **전체 ABC 생성·학습은 아직 시작 안 함.**
읽은 것: `261112_noeun_train_deploy_gap_result.md` · `261111_noeun_stop_radius_eval_result.md` ·
`261110_noeun_stop_radius_result.md` · `261110_noeun_dagger_result.md` · 84 · 85 · 81 · 82.

## 0. 과녁 — ABC 가 고칠 수 있는 몫 (r1.0 VLN-PE v6_b1 val_seen 40 ep)

| 분류 | 수 | ABC 사정권 |
|---|---|---|
| 성공 | 13 | — |
| 지나침 | 8 | ✅ 정면 과녁 |
| 3~5 m 앞 조기 STOP (ep 17 22 32 36) | 4 | ✅ 부분 |
| 같은 층 못 감 (나머지) | ~7 | ❌ DAgger |
| 걸음 상한 600 | 6 | ❌ 평가 예산 (100 m 경로는 전진만 400 step) |
| 다른 층 | 2 | ❌ |

- **천장 12/40.** 판정은 SR 하나가 아니라 `지나침` · `걸음상한` 내역을 **같이** 본다
  (질의를 늘리면 지나침 ↓ 와 걸음상한 ↑ 이 반대로 움직여 SR 이 안 변할 수 있다).

## 1. (실험 완료 · 가설 기각) 질의 간격 2 m → 1 m

### 왜 했나 — 학습과 평가의 질의 간격이 2 배 달랐다 (코드 확인)

| | 학습 | 평가 |
|---|---|---|
| 질의 간격 | **1 m** (`internvla_n1_argument.py:52` `sample_step=4` × 0.25 m) | **최대 2 m** (`internvla_n1_s2_agent.py:34` `MAX_STEPS=8` × 0.25 m) |
| 목표점 거리 | 최대 3.25 m 앞 (`vlnce_pixel_goal.py:68` `MAX_FORWARD_STEPS=13`) 중 **보이는 가장 먼 점** | 모델 출력 → ~2~2.6 m 앞 |

- `MAX_STEPS = 8` 은 노은역용이 아니라 **habitat 평가기에서 물려받은 상수**. 코드 수정 없이
  **환경변수 `NOEUN_MAX_FOLLOW`** 로 바꾼다 (config 에 원래 있던 손잡이).
- 이동 중에는 **안 묻는다** — 픽셀 목표에 도착(follower STOP)하거나 8 걸음을 다 써야 다시 묻는다 (`:306`).
- 함정: `forward_action` 은 이름과 달리 **회전도 1 로 센다** (`:305`).
- r1.0 모델의 첫 STOP 중앙 **1.56 m** → 멈추려는 지점을 질의가 건너뛸 수 있다고 봤다.

### 명령 (이 서버, 1 시간 걸림)

`SG=1 EG=0 INST=internnav NOEUN_FOLLOWER=navmesh NOEUN_NAVMESH_URL=http://127.0.0.1:8040 NOEUN_BASE_HEIGHT=1.05 NOEUN_MAX_FOLLOW=4 NOEUN_DATA=/ws/src/InternNav/data/data/noeun_eval/labels/vln_pe_noeun_nav3d_rbq_v6_b1/raw_data/r2r bash scripts/apptainer/run_noeun_vlnpe_eval_apptainer.sh noeun_r1.0_v6b1_navmesh_follow4 /ws/src/InternNav/checkpoints/relabel_half_noeun-v01/s1.noeun-both-b1_wait-wait_stopr1.0_internvla-n1-system2_20261110_112953`

| 인자 | 뜻 | 왜 이 값 |
|---|---|---|
| `NOEUN_MAX_FOLLOW=4` | 픽셀 목표 하나를 따라가는 최대 step | 4 × 0.25 m = **1 m** = 학습 `sample_step` 과 같게 |
| `SG=1` / `EG=0` | 모델 서버 GPU / Isaac 렌더·물리 GPU | 이 서버는 GPU 0·1 만 사용 가능 (2·3 은 타인 작업) |
| `NOEUN_BASE_HEIGHT=1.05` | base z − 바닥 | H1 (sim RBQ10 은 0.55) |

### 결과 — **가설 기각. 지나침이 줄지 않았다.**

| | 기준 (2 m 질의) | 이번 (1 m 질의) |
|---|---|---|
| SR | 0.325 | **0.300** |
| OS | 0.525 | 0.550 |
| SPL | 0.312 | 0.295 |
| NE | 12.9 | 12.46 |
| **지나침** | **8** | **9** |
| 걸음상한 | 6 | 7 |
| 3 m 안에 못 감 | 11 | 13 |

- SR 차이는 "실행 흔들림 1~3 ep" 범위 — **악화라 단정 못 하지만 좋아지지도 않았다.**
- 결정적 증거 — 지나친 9 개 중 4 개는 **목적지 0.2 m 안까지 갔다**:
  `ep27 0.12 m → 끝 4.91` · `ep1 0.12 → 7.06` · `ep21 0.13 → 36.57` · `ep24 0.21 → 27.72`
  1 m 마다 물었으니 그 자리에서 **분명히 물었고**, 그래도 "앞으로" 라고 답했다.
  → 문제는 **"물어볼 기회가 없었다" 가 아니라 "물어봐도 못 멈춘다"**.
- 부작용은 예상대로: 걸음상한 6 → 7 (질의마다 제자리 1 걸음).
- 분석 결과: `logs/noeun_r1.0_v6b1_navmesh_follow4/breakdown.json` (ep 별 최근접·최종 거리·결과).

## 2. (정정) 계단 — **목적지가 계단 위인 경로는 사실상 없다**

### 내가 틀렸던 것
처음에 "387 경로 중 목적지가 계단 위 = 219 (57 %), 쓸 수 있는 건 168 (43 %)" 라고 보고했다. **근거 없는 수치다.**
기준을 "`reference_path` 마지막 구간의 `dz/dxy > 0.2`" 로 잡았는데, 마지막 구간은 수평 이동이 0.06~0.22 m 밖에 안 돼
**높이 6~11 cm 차이(바닥 요철·리샘플링 오차)만으로 경사로 잡혔다.**

### 맞는 기준 — navmesh 연결부 bbox
`verify.json` (12_build_navmesh 결과): 노은역은 **3 층 · 층 연결부 3 개, 전부 `kind: stairs`**
(`stepness_rms_m` 0.137/0.145/0.149, 기준 `STEP_RMS_M=0.05`; 기울기 25.8~27.6°). 층간 통행 검증
`v5: n_pairs 200, found 200, on_navmesh_frac_min 1.0, passed: true` — 걸어서 이동 가능. **엘리베이터는 없다.**

| 계단 | 잇는 층 | bbox x | bbox y | z |
|---|---|---|---|---|
| id0 | 0↔1 | 10.2 ~ 23.0 | 3.7 ~ 8.5 | −5.2 ~ −0.4 |
| id1 | 0↔1 | 11.5 ~ 22.2 | −9.3 ~ −7.4 | −5.1 ~ −0.5 |
| id2 | 1↔2 | −26.9 ~ −5.6 | −21.9 ~ −19.9 | −0.2 ~ 8.3 |

| 기준 | train 387 중 "목적지가 계단 위" |
|---|---|
| 계단 bbox 안 (맞는 기준) | **0 개** |
| 마지막 구간 경사 (틀린 기준) | 219 개 |
| 둘 다 | **0 개** (전혀 안 겹친다) |

### "계단 경로를 빼자" 를 실제로 해 보면 — 성적이 **나빠진다**
val_seen 40 ep 에서 계단 bbox 기준: 계단 출발 2 · 계단 목적지 3 → 제외 대상 **5 개**.

| | n | SR | OS | 지나침 | 걸음상한 |
|---|---|---|---|---|---|
| 전체 | 40 | 0.275 | 0.550 | 11 | 7 |
| 계단 제외 후 | 35 | **0.229** | 0.486 | 9 | 6 |
| (제외된 5 개) | 5 | **0.600** | 1.000 | 2 | 1 |

빠지는 5 개가 **오히려 쉬운 경로**(다섯 다 3 m 안까지 감)라 제외하면 숫자가 떨어진다.
→ **실패의 주원인은 계단이 아니다.** "계단 제외" 작업 항목은 할 일이 없어졌다.

## 3. (완료) STOP 라벨에 방향 조건 — 81 수정

`81_dagger_navmesh_server.py` 의 STOP 판정이 **거리만** 봤다 (xy ≤ `stop_radius_m`, |dz| < 1.0).
→ 84 smoke 에서 **목표가 등 뒤(90°+)인데 STOP 라벨 13 개**.

고친 것: 반경 안이어도 **목표가 `stop_yaw_deg`(기본 45°) 안에 보일 때만 STOP**, 밖이면 목표 쪽으로 회전.
아주 가까우면(`min_goal_dir_m` 0.3 m, 84 `view()` 와 같은 값) 방향이 정의되지 않으므로 그대로 STOP.
`--stop_yaw_deg 180` 으로 예전 동작 복원 가능. 사용자 지시로 **82(DAgger)도 같은 기본값**을 쓴다 (예전 DAgger 데이터는 폐기 예정).

확인 (목표까지 0.6 m, 반경 1.0 m, 보는 방향만 바꿈):

| 보는 방향 | 새 코드 | 예전 코드 |
|---|---|---|
| 정면 0° / 비스듬 30° | STOP | STOP |
| 옆 90° | 우회전 ×3 → STOP | STOP ❌ |
| 등 뒤 180° | 우회전 계속 | STOP ❌ |

- r1.0 학습은 각도를 직접 안 썼지만 안전했다: `internvla_n1_lerobot_dataset_stop_radius.py:8-11` 의 조건 (b)
  "f 부터 끝까지 남은 action 이 전진뿐" 이 **정답 경로 위에서는** "목표가 앞에 있다" 와 같은 뜻이기 때문.
  82 · 84 는 일부러 비켜서고 지나치므로 (b) 가 그 역할을 못 한다.

## 4. 라벨 없는 프레임 — 원래 데이터에도 있고, 처리 규칙이 이미 있다

`internvla_n1_lerobot_dataset.py:917-944`:

| 상황 | 기존 처리 |
|---|---|
| 목표점 안 보임 + 정답이 **회전** | **회전(turn) 샘플로 만든다** (r=0 에서 3,438 개) |
| 목표점 안 보임 + 정답이 **전진** | 버린다 |
| `goal_len < 3` (너무 가까움) | 버린다 |

- `check_visible` (`vlnce_rig.py:77`) 가 거르는 것: `behind_camera` · `too_close`(<0.3 m) · `out_of_image` · `occluded`.
  `occluded` = 그 픽셀 깊이가 목표 거리보다 0.35 m 이상 **가깝다** = 사이에 뭔가 있다 (주로 모퉁이).
- **가려진 곳에 점이 찍히는 일은 없다** — `select_goal` 은 보이는 점 중 가장 먼 것만 고른다.
- **"follower 가 피해 가니 괜찮다" 는 성립 안 함**: `internvla_n1_s2_agent.py:199` `_pixel_to_world` 가
  **그 픽셀의 깊이**로 3D 목표를 만든다 → 벽을 가리키면 목표가 **벽 표면**에 생기고 navmesh snap 은 벽 **앞** 바닥으로 보낸다.
  벽을 돌아가지 않는다. **가림 검사는 유지.**
- ABC 에서는 비율이 다를 것 (82 시험 240+47 프레임 중 18 개 ≈ 6 %). 정확한 수치는 traj_data 가 있는 A5000/H200 에서만 셀 수 있다.

## 5. 그 밖에 정해진 것

- **반경 r**: 84 에서 고정하지 않고 `abc.goal_dist_m` 만 저장, **학습 로더가 결정** (현재 설계 유지) → r2.0 · r3.0 결과 반영 가능.
- **history**: prefix 가 정답대로 걸은 깨끗한 과거라 실제 주행(이탈 · 제자리 회전 11 % · 같은 그림 반복)과 다르다.
  → **추후 DAgger 적용 시 해결**. 지금은 손대지 않는다.
- **양**: 387 × 3 가지 × 40~60 프레임 ≈ 5만~7만 장 = 기존 학습(74 k)과 거의 1:1 → 과하다.
  **100 경로 먼저**, 섞는 비율은 **학습 config 로 조절 가능하게**.
- **학습 로더** (미확인): `internnav/dataset/internvla_n1_lerobot_dataset_frame_label.py` 를 새로 만들고 **ABC · DAgger 공용**.
  `STOP_RADIUS_DATASETS` 처럼 이름 표 등록, trainer 는 `internvla_n1_stop_radius_trainer.py` 패턴.

## 6. 지금 열려 있는 질문

**모델이 목적지 0.12 m 앞에서도 못 멈추는 이유를 모른다.** 계단도 아니고, 비켜선 것도 아니고, 바로 그 자리였다.
ABC 가 만드는 "옆으로 1 m 비켜선 장면" 이 이걸 고칠지 불확실하다.
→ **다음 후보: 지나친 9 개의 agent 로그 분석** (`logs/<task>/agent/ep*/steps.jsonl` 에 모델 출력·픽셀·위치가 다 있다, 약 1 h).

필요한 것 (A5000):
- `logs/stopr_eval/vlnpe_fail.json` — ABC 의 옆 거리 · 접근 각도 · 지나침 거리 분포
- r2.0 · r3.0 결과 — 반경 r 확정

## 7. 이 서버 상태 (2026-11-12)

- 코드: InternNav `main` 8e6a45b (+ 81 수정, **커밋 안 함**) · InternUtopia `main` 92b9353. 둘 다 origin 과 0/0.
- 데이터 · 체크포인트: gdrive 원본과 **MD5 전량 일치**. 평가 셋 `vln_pe_noeun_nav3d_rbq_{v6,hunav_v1}_b1` 도 받음.
- **isaac**: 기존 sandbox `~/workspace/sandbox/internnav` + overlay `internnav_260723.img` — torch 2.7.0+cu128, arch 에 **sm_120**.
  `bash docker/start-apptainer-instance.sh internnav` 로 인스턴스 기동 → `apptainer exec instance://internnav` 가 `docker exec` 자리.
- **habitat**: 새로 빌드 `~/workspace/sif/internnav-habitat-6000pro.sif` (20 GB, `docker/habitat-6000pro.def`).
  a5000 Dockerfile 그대로 + `TRITON_OVERRIDE_ARCH=sm120`. **numba 0.65.0 + llvmlite 0.47.0** 짝이 맞아 `habitat_sim 0.3.3` import 성공.
  (기존 `internnav_habitat_260820.img` 는 Isaac Sim 베이스 위에 habitat 을 얹어 numba 0.59 ↔ llvmlite 0.49 충돌로 깨져 있다 — habitat 이미지의 베이스는 Isaac 이 아니라 ROS 이미지다.)
- 래퍼: `scripts/apptainer/noeun_apptainer.sh (isaac|habitat) <명령>` (`DRY=1` 지원) ·
  `scripts/apptainer/run_noeun_vlnpe_eval_apptainer.sh` (평가, docker 판 포팅) · `scripts/apptainer/noeun_manifest.sh` (두 서버 지문 대조).
- 서버: 81 DAgger navmesh :8041 · 평가용 navmesh :8040 (둘 다 habitat SIF, CPU).
- 함정: apptainer 빌드 중 호스트 `/tmp` 가 컨테이너 `/tmp` 를 덮는다 (`apptainer.conf` `mount tmp = yes`) → `%files` 는 `/opt` 로.
- 함정: `pkill -f '<스크립트명>'` 은 **자기 명령줄에도 걸려** 셸이 같이 죽는다. `pgrep` 후 자기 PID 제외할 것.
