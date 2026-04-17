# Project Guidelines

## 제한사항
- 코드 변경은 최소한으로 (conservative changes)
  - 코드 변경할 부분이 많으면 새로운 파일을 생성해서 기존 코드에 영향을 적게
  - 기존 코드의 argument를 줘서 해결할 수 있으면 코드 변경 x
- 한 번에 하나의 파일만 수정

## scripts/train/base_train/
- start_1gpu.sh: CPU core 24, RAM 128G, RTX 5090 1GPU, shm 32g
- start_1node.sh: CPU core 128, RAM 2TB, H200 8GPU, shm 1.8TB
