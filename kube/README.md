# kube

Kubernetes pod ↔ 로컬 간 체크포인트 전송 스크립트 모음. 두 가지 방식:
- **kubectl + rsync** (`kube-rsync.sh` / `download.sh`): 별도 설정 없이 바로 사용. K8s API 서버를 경유하는 단일 스트림이라 대용량 파일에서 끊김/속도저하 발생 가능.
- **rclone + Google Drive** (`download_checkpoints_rclone.sh` / `upload_checkpoints_rclone.sh`): 최초 1회 인증 필요하지만, pod→Drive→로컬 구간이 API 서버를 거치지 않아 대용량 전송에 더 안정적.

## 사전 준비

### kubectl, rsync설치
```bash
curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && chmod +x kubectl && sudo mv kubectl /usr/local/bin/

apt install rsync
```

### kubeconfig 설정
`kube/internnav-kubeconfig.yaml`을 배치한 후 환경변수 등록:
```bash
export KUBECONFIG=/ws/src/InternNav/kube/internnav-kubeconfig.yaml
```

> `internnav-kubeconfig.yaml`은 보안상 git에서 제외되어 있습니다. 별도로 전달받아 배치하세요.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `kube-rsync.sh` | rsync transport 래퍼 (`-e` 옵션용, 직접 실행 X) |
| `download.sh` | kubectl exec 기반 pod 파일 다운로드 (`rsync -avz --partial --progress`) |
| `download_checkpoints_rclone.sh` | rclone 기반 Drive → 로컬 다운로드 |
| `upload_checkpoints_rclone.sh` | rclone 기반 로컬(pod) → Drive 업로드 |
| `internnav-kubeconfig.yaml` | 클러스터 접속 설정 (git 제외) |
| `.tools/` | rclone 바이너리 자동 설치 위치 (git 제외) |

## 사용법: kubectl + rsync

### pod 목록 확인
```bash
kubectl get pods
```

### 파일 다운로드
```bash
bash kube/download.sh <pod> <src> <dst>

# 예시
bash kube/download.sh internnav-train1-1-0 /workspace/data ./data
```

## 사용법: rclone + Google Drive

대상: `gdrive` remote 계정의 **My Drive 루트** (`https://drive.google.com/drive/u/1/my-drive`), 그 아래
`InternNav/checkpoints/` 폴더 (스크립트의 `DRIVE_PREFIX` 기본값이 자동으로 붙음, `cloud_rel`에는 그 아래 상대경로만 적으면 됨).
`gdrive` remote는 `~/.config/rclone/rclone.conf`에 저장되며, 이 remote를 만드는 계정이 곧 대상 My Drive 계정입니다.

### 1. 최초 1회 인증 (헤드리스 환경 + 브라우저 있는 PC)
브라우저 없는 서버/컨테이너에서 직접 로그인은 불가능하므로, **브라우저가 있는 PC에서 토큰만 발급**해 이 환경에 주입합니다.

**① 브라우저 있는 PC에서:**
```bash
rclone authorize "drive"
```
→ 브라우저 로그인 → 출력되는 토큰 JSON을 통째로 복사

**② 이 환경(로컬 또는 pod)에서:**
```bash
kube/.tools/rclone-v1.74.3-linux-amd64/rclone config create gdrive drive scope=drive token='<복사한 JSON>'
```
- `Waiting for code...`가 떠도 정상 (remote는 이미 저장됨) → Ctrl-C로 종료

pod에서 업로드하려면 pod 안에서도 동일하게 rclone 설치(`upload_checkpoints_rclone.sh` 최초 실행 시 자동 설치) + 인증이 필요합니다.

### 2. 다운로드 (Drive → 로컬)
```bash
bash kube/download_checkpoints_rclone.sh <local_path> <cloud_rel>

# 예시
bash kube/download_checkpoints_rclone.sh \
  /ws/src/InternNav/checkpoints/image_base/base_s1.fpv_s2.fpv_20260709_130547 \
  image_base/
```

### 3. 업로드 (pod/로컬 → Drive)
```bash
# pod 안에서 실행 (kubectl exec -it <pod> -- bash)
bash kube/upload_checkpoints_rclone.sh <local_path> <cloud_rel>

# 예시
bash kube/upload_checkpoints_rclone.sh \
  /home/irteam/data-vol2/checkpoints/image_base/base_s1.fpv_s2.fpv_20260709_130547 \
  image_base/
```
- 인자 순서 무관 (절대경로 쪽이 자동으로 local로 인식), `cloud_rel`이 `/`로 끝나면 로컬 폴더명 자동 부착
- 재실행 시 이미 전송된 파일은 크기/해시 비교로 skip (resume)
- `--check`: 복사 없이 무결성만 검증 / `--dry-run`(download만): 경로 resolve만 확인, 네트워크 접근 없음
- 상세 옵션은 `bash kube/download_checkpoints_rclone.sh --help` 참고
