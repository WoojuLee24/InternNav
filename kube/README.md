# kube

kubectl + rsync로 Kubernetes pod에서 파일을 다운로드하는 스크립트 모음.

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
| `download.sh` | pod에서 파일 다운로드 |
| `internnav-kubeconfig.yaml` | 클러스터 접속 설정 (git 제외) |

## 사용법

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
