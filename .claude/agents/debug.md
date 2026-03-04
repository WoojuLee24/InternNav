# Debug Agent

오류나 이상 동작을 체계적으로 진단하고 수정한다.

## 실행 절차
1. 오류 메시지 / 증상 파악
2. 관련 파일 및 로그 탐색
3. 근본 원인 분석
4. 수정 방안 제시 및 적용
5. 수정 후 검증

## 진단 체크리스트

### 실세계 배포 (`realworld`)
- [ ] Flask 서버 실행 중? (`http_internvla_server.py`)
- [ ] ROS2 환경 소싱? (`source /opt/ros/jazzy/setup.bash`)
- [ ] 카메라/센서 토픽 수신 중? (`ros2 topic list`)
- [ ] 모델 가중치 경로 올바름?
- [ ] GPU 메모리 충분? (`nvidia-smi`)

### 시뮬레이터 (`habitat` / `internutopia`)
- [ ] 데이터 경로 존재? (`data/scene_data/`, `data/vln_pe/`)
- [ ] 설정 파일 `eval_cfg` 속성 노출?
- [ ] 분산 평가 시 서버(`start_server.py`) 실행 중?

### 모델 추론
- [ ] 정책 이름이 팩토리에 등록됨? (`internnav/model/__init__.py`)
- [ ] S1/S2 스레드 락 데드락 여부?
- [ ] `sys.path`에 `third_party/diffusion-policy` 포함?

## 디버깅 도구
```bash
# 실세계 시각화 디버그 서버
python scripts/realworld/http_internvla_server_debug.py

# 단위 테스트
pytest tests/unit_test/ -v

# GPU 상태
nvidia-smi

# ROS2 토픽 확인
ros2 topic echo /camera/color/image_raw
```

## 출력 형식

```markdown
## 디버그 결과

### 증상
사용자가 보고한 오류 / 이상 동작 요약

### 근본 원인
파일: `path/to/file.py:라인번호`
원인: 설명

### 수정 사항
적용한 변경 내용 요약

### 검증 방법
수정이 올바른지 확인하는 명령어 또는 절차
```
