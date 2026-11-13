# tmux 소켓 격리 (apptainer mount namespace) — 결과

_2026-07-23_

## 문제

컨테이너 안에서 `ls data/` 는 정상인데, tmux 세션 안에서는 `data/` 가 비어 보임.

## 원인

1. `/ws/src/InternNav/data` 는 `APPTAINER_BIND` 로 주입되는 bind mount → 해당 apptainer 인스턴스의 mount namespace 안에서만 보인다. (`/ws` 를 bind해도 그 안의 중첩 mount는 따라오지 않아 `data` 를 별도 bind한 구조)
2. tmux 기본 소켓 `/tmp/tmux-<uid>/default` 는 호스트 `/tmp`(공유 ext4)에 있어 컨테이너 종료 후에도 살아남는다.
3. 새 컨테이너에서 `tmux` / `tmux a` 를 치면 새 서버가 뜨는 게 아니라 **예전 실행의 서버**에 붙고, 그 서버가 fork하는 셸은 **예전 namespace를 상속** → `data/` bind 없음.

실제 확인:

```
tmux 서버 PID 1101323  start=Tue Jul 21 10:18:21   ← 옛 인스턴스
현재 셸 mnt ns          mnt:[4026534349]            ← 현재 인스턴스
/proc/1101323/ns/mnt   Permission denied           ← 다른 namespace
```

## 변경 (`docker/run-...-6000pro-apptainer.sh`)

- 실행마다 `TMUX_RUN_ID=$(date +%y%m%d_%H%M%S)` 로 소켓 디렉토리 분리, `--env TMUX_TMPDIR=/ws/.tmux/<run_id>` 전달
  → 붙을 서버가 없으므로 tmux가 **현재 namespace에서 새 서버를 fork**, bind가 그대로 보인다
- 실행 시 예전 실행 디렉토리를 스캔해 **소켓이 남아있는 것(=서버 생존)만 안내**하고 나머지는 정리
- 안내 출력:
  ```
  [tmux] 이번 실행 소켓: /ws/.tmux/260723_225846  → 컨테이너 안에서 그냥 'tmux new -s train'
  [tmux] 이전 실행의 살아있는 세션에 붙으려면 (그 실행의 mount를 그대로 물려받음):
           TMUX_TMPDIR=/ws/.tmux/260721_101800 tmux a
  ```

## 검증

- `bash -n` 통과
- 실제 tmux 서버 기동: 새 소켓 서버는 `default` 소켓(옛 서버)과 완전히 분리되고, 그 안에서 `ls /ws/src/InternNav/data` 가 `Embodiments / InternData-N1-v0.5-mini / vlnverse_emr` 정상 출력
- prune 로직: 소켓 없는 디렉토리 삭제 / 소켓 살아있는 디렉토리 유지+안내 확인
- 기존 `train` 세션(옛 서버)은 건드리지 않음

## 함정 (구현 중 발견)

- tmux는 `TMUX_TMPDIR` 아래에 **`tmux-<uid>/` 를 한 단계 더 만든다** → 실제 소켓은 `${TMUX_TMPDIR}/tmux-<uid>/default`. 따라서 run 디렉토리는 서버가 죽어도 `-empty` 가 아니다 (prune을 `find -type s` 기준으로 해야 함).
- unix socket 경로 길이 제한(~107자) → `TMUX_TMPDIR` 를 긴 경로로 잡으면 `File name too long` 으로 서버 기동 실패. run id는 짧게 유지.

## 남은 정리 (수동)

옛 서버 안에 돌던 job이 없으면:

```
tmux -L default ls && tmux -L default kill-server
rm /tmp/tmux-1002/-s      # 'tmux -L -s train' 오타로 생긴 소켓
```
