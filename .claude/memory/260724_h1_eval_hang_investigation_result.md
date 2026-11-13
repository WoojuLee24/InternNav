# h1(Isaac Sim) eval hang 원인 조사 결과

_작성: 2026-07-24 / 대상: `--machine h1` eval 도중 프로세스가 얼어붙는 현상_

## 증상

`scripts/train_eval/qwenvl_train/runner.py --machine h1 --no-train` 로 돌린
InternVLA-N1 eval(PID 14204, `--flash-collision none`)이 에피소드 ~550/1311 에서
멈춤. 프로세스는 살아있으나 진행 0. 확인 시점 기준 3~4시간 방치됨.

## hang 지점 (증명됨)

`env.step()` **내부**에서 정지. 매 스텝 로그 순서는
`agent step time`(vln_distributed_evaluator.py:149) → `Env Step time`(:183) 인데,
마지막 사이클은 `agent step time: 0.0s` 만 찍히고 `Env Step time` 이 안 나옴.
즉 에이전트(S1/S2 모델)는 정상 리턴했고 **Isaac Sim 시뮬레이션 advance 에서 멈춤**.
→ 모델/S2 버그 아님.

## 스레드 상태 (정황)

- 메인 스레드(14204): `wchan=futex_do_wait` (락 대기)
- Omniverse 워커풀: 대부분 `futex_do_wait`, **`carb.tasking3`(14694) 하나만
  `wchan=0`(userspace)에서 스핀** (~33% CPU)
- GPU util 0%, D-state(GPU 드라이버 대기) 스레드 없음
→ **Isaac Sim/Omniverse 내부 스레드 동기화 stall** 과 일치. 단, py-spy/gdb 부재로
  정확한 C++ 프레임은 미확보. "carb.tasking 스케줄러 데드락"은 정황 추론이지 증명 아님.

## system memory 초과 여부 → **원인 아님 (배제됨)**

| 항목 | 값 | 해석 |
|---|---|---|
| cgroup `memory.max` | `max` | 컨테이너 메모리 제한 없음 (251 GiB 다 가용) |
| cgroup `memory.events` | `oom 0`, `oom_kill 0`, `high 0`, `max 0` | OOM·throttle **0회** |
| 프로세스 `VmHWM` (평생 최대 RSS) | 9.7 GiB | 251 GiB 중 최대 9.7 GiB만 사용 |
| 프로세스 `VmSwap` | 0 | 스왑으로 밀린 적 없음 |
| 호스트 `free` available | 216 GiB | 여유 충분 |
| Swap 사용 | 428 KiB / 63 GiB | thrashing 없음 |
| kernel OOM-killer 로그 | 없음 | — |

논리: OOM이면 프로세스를 **kill** 하지 hang 시키지 않음(살아있음). 메모리 압박이면
스왑 thrashing/`memory.events.high>0` 가 보여야 하는데 VmSwap=0·high=0.
`memory.current`(185 GiB)가 커 보이는 건 `free` 의 buff/cache(208 GiB, 회수가능한
페이지캐시)와 일치 — 압박 아님.

## 결론

hang은 RAM 부족과 무관. **Isaac Sim/Omniverse 내부 stall** 이 유력.
정확한 프레임 확정에는 py-spy/gdb attach 필요(미설치).

## 대응 (진행 중)

- 원인 확정과 별개로 **외부 watchdog** 으로 복구: 로그 mtime >10분 정지 → 프로세스
  그룹 kill → GPU 메모리 해제 대기 → 동일 명령 재실행. eval은 이미 자동 resume
  (lmdb skip-done + `wandb_run_id.txt` 공유)이라 이어서 진행됨. 무한루프 방지 위해
  연속 3회 무진행 시 중단. → `scripts/train_eval/qwenvl_train/run_watchdog.sh`
- none/stop 결과 분리: `run_eval` 의 `log_dir` 을 h1에서 `logs_<flash_collision>`
  으로 분리(같은 wandb run 유지). runner.py 반영 완료.
