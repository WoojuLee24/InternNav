# InternNav를 dstart/dexec로 실행 — 결과

_2026-07-23_

## 배경

`dstart`/`dexec`는 VLN-Challenge의 apptainer 래퍼(`~/.bashrc:143` → `VLN-Challenge/docker/apptainer-docker-aliases.sh`)인데 sandbox/overlay가 vln-challenge로 하드코딩돼 있었다. 그래서 `dstart internnav`를 하면 **VLN 컨테이너를 "internnav"라는 이름으로 띄우려 시도**한다.

실제로 프로젝트별로 갈리는 건 `dstart` 하나뿐이다. `dexec`/`dps`/`drm`은 `instance://<name>`만 보므로 원래 프로젝트 무관.

## 왜 instance 방식인가

- `apptainer exec` (`run-...-apptainer.sh`) = 호출마다 별도 mount namespace → 터미널마다 다른 namespace
- `apptainer instance start` = namespace 하나, 모든 exec 세션이 합류 → tmux/프로세스가 자연스럽게 공유됨

단, **apptainer는 호스트 `/tmp`를 그대로 bind**하므로 인스턴스 안에서 그냥 `tmux`를 치면 여전히 다른 인스턴스/예전 실행의 서버(`/tmp/tmux-<uid>/default`)에 붙는다.
→ 인스턴스 전용 `TMUX_TMPDIR=/ws/.tmux/instance-<name>`으로 해결. 관련: [[tmux_namespace_isolation_result]]

### 근본 원인 확정

처음 겪은 "tmux 안에서 data/가 안 보임"의 정체:

```
1096829  appinit            ← vln 인스턴스 (dps에 보이는 그것)
1101325  tmux new -s train  (PPID=1096829)
```

`train` 세션은 **VLN-Challenge 컨테이너 안에서** 돌고 있었다. 거기엔 InternNav의 `data` bind가 없다.

## 변경

| 파일 | 변경 |
|---|---|
| `InternNav/docker/start-apptainer-instance.sh` | **신규.** `apptainer instance start`. sandbox=`~/workspace/sandbox/internnav`, overlay=`internnav_260723.img`, 인스턴스명 기본 `internnav`. bind 구성은 `run-...-6000pro-apptainer.sh`와 동일. GPU는 `CUDA_VISIBLE_DEVICES`로 제한. 중복 기동 거부. 기동 실패 시 `\|\| exit 1` (VLN 버전은 이게 없어서 실패해도 "시작됨"을 출력함) |
| `VLN-Challenge/docker/apptainer-docker-aliases.sh` | **수정 3곳.** ① `NAV_DOCKER_DIR` 추가 ② `dstart`에 인스턴스 이름 기반 dispatch (`internnav\|nav*` → InternNav, `vln*` → VLN, 그 외 exit 2) ③ `dexec`에 `--env TMUX_TMPDIR=/ws/.tmux/instance-<name>` + 호스트 쪽 mkdir |

명령어는 한 벌(`dstart`/`dexec`/`dps`/`drm`)로 유지. `~/.bashrc`는 원상복구(추가했던 source 줄 제거).

## 사용법

```bash
dstart internnav 0    # 인스턴스 기동, 물리 GPU 0번만
dexec internnav       # 셸 열기, 몇 개든
dps / drm internnav   # 목록 / 중지
```

컨테이너 안에서는 그냥 `tmux new -s train`. 다른 터미널에서 `dexec internnav` 후 `tmux a` 하면 같은 서버에 붙고 `data/`도 정상.

## 주의: overlay 동시 사용

`internnav_260723.img` 하나를 둘이 동시에 물면 손상된다. apptainer가 락으로 막아주긴 하지만(실제로 `dstart internnav`가 이 락 때문에 FATAL로 거부됐음) 의존하지 말 것.

- 인스턴스 1개 + `dexec` 여러 개 → 같은 namespace라 안전
- **`run-...-6000pro-apptainer.sh` 셸이 떠 있으면 `dstart internnav` 전에 종료할 것**
- `vln` 인스턴스는 overlay가 달라 같이 떠 있어도 무방

## 검증

- 두 파일 `bash -n` 통과
- `apptainer`를 stub으로 대체해 실제 조립되는 커맨드 확인:
  - `dstart internnav 0` → internnav sandbox + `internnav_260723.img` + `CUDA_VISIBLE_DEVICES=0` + `TMUX_TMPDIR=/ws/.tmux/instance-internnav` + `.../data:/ws/src/InternNav/data:rw`
  - `dstart vln` → vln-challenge overlay (기존 동작 그대로)
  - `dstart foo` → `unknown instance name`, exit 2
  - `dexec internnav` / `dexec vln` → 각각 `instance-internnav` / `instance-vln` 소켓
- 테스트 흔적 정리 완료
- **미검증**: 실제 `apptainer instance start`. `apptainer` 바이너리가 컨테이너 안에 없어(호스트 전용) 여기서 실행 불가

## 남은 사항

- **`--env HOME=/home/dnwn24`** → apptainer가 `WARNING: Overriding HOME environment variable with APPTAINERENV_HOME is not permitted` 을 내는 no-op. 기본값이 이미 `/home/dnwn24`라 지워도 동작은 같다. 사용자가 제거를 거부해서 그대로 둠 (경고만 뜸)
- **MP3D bind**: `~/workspace/data/InternData-N1-v0.5-mini/scene_data/mp3d_pe`가 호스트에 없어서 스킵됨. 기존 run 스크립트도 같은 조건이라 동작 차이는 없음
- **`dcommit`은 여전히 vln 전용.** InternNav 컨테이너를 commit하려면 override:
  `VLN_SANDBOX=~/workspace/sandbox/internnav VLN_OVERLAY=~/workspace/sandbox/internnav_260723.img dcommit out.sif --merge-overlay`
