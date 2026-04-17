# Auto Dev Branch Workflow

새 브랜치를 만들고 안전하게 자율 작업을 수행합니다.

## 사용법
/auto-dev <branch-name> "<작업 내용>"

## 실행 순서

### 1. 브랜치 생성
```bash
git checkout -b $ARGUMENTS
```

### 2. tasks/<branch-name>.md에 작업 범위 기록
현재 작업 목표, 수정 가능 범위, 금지 사항, 완료 조건을 
임시 파일 .claude/tasks/<branch-name>.md에 작성한다.

### 3. 작업 규칙 (반드시 준수)
- 수정 가능: 작업 목표와 직접 관련된 파일만. 
- 수정 금지: data/, checkpoints/, *.ply, *.parquet
- 브랜치 금지: dev, main, master 등 기존 작업 branch에 직접 push 금지
- 최대 시도: 실행 3회 실패 시 현재 상태 보고 후 중단
- 자율 진행: 작업 중 사용자에게 질문/확인 금지. 판단이 필요한 경우 추천 선택을 한다.

### 4. 권장 사항
- 코드 수정은 되도록이면 적게 - command의 argument를 수정해서 해결할 수 있는 부분은 수정하지 말 것. 
- 코드 수정이 많으면 새로운 파일을 만들어서 개발 및 import 해라 

### 5. 작업 완료 시
- 변경 파일 목록과 수정 내용 요약 보고
- git diff 출력
- dev branch와 merge 여부를 사용자에게 확인 요청

### 6. 실패 시
- 시도한 내용과 실패 원인 보고
- 브랜치는 삭제하지 않고 유지
- .claude/tasks/<branch-name>.md는 유지 (재시도 컨텍스트용)

### 작업 완료 후
- .claude/tasks/<branch-name>.md 삭제