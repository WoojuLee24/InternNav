# Auto Dev Branch Workflow

새 브랜치를 만들어 자율적으로 작업·검증을 끝까지 수행한 뒤, merge/discard 여부만 사용자에게 묻습니다.

## 사용법
/auto-dev <branch-name> "<작업 내용>"

## 워크플로우

### 1. 원래 브랜치 저장 및 작업 브랜치 생성
```bash
ORIGINAL_BRANCH=$(git branch --show-current)
git checkout -b feature/<branch-name>
```
- 원래 브랜치 이름을 기억해두고 작업 종료 시 반드시 복귀한다.

### 2. 작업 범위 기록
`.claude/tasks/<branch-name>.md`에 작성:
- 작업 목표 / 수정 가능 범위 / 금지 사항 / 완료 조건(검증 방법 포함)
- 이 파일은 실패 시 재시도 컨텍스트로 사용된다.

### 3. 작업 규칙 (반드시 준수)
- 수정 가능: 작업 목표와 직접 관련된 파일만
- 수정 금지: `data/`, `checkpoints/`, `*.ply`, `*.parquet`
- push 금지: dev, main, master 등 기존 브랜치에 직접 push 금지
- 자율 진행: 작업 중 사용자에게 질문/확인 금지. 판단이 필요하면 추천 선택을 하고 태스크 파일에 근거를 기록한다.
- 코드 수정은 최소한으로 — argument로 해결 가능하면 코드 수정 금지, 수정이 많으면 새 파일을 만들어 import

### 4. 검증 (필수)
작업 완료 후 반드시 실제 실행으로 검증한다:
- 완료 조건에 정의한 검증 명령을 실행 (예: `pytest tests/unit_test/`, eval 스크립트, import smoke test)
- python 실행은 `/workspace/isaaclab/_isaac_sim/python.sh` 사용
- 검증 결과(성공/실패 로그 요약)를 태스크 파일에 기록
- **최대 시도 3회**: 검증 3회 실패 시 6-B단계(실패 처리)로 이동

### 5. 커밋
```bash
git add <작업으로 생성/수정한 파일만 명시적으로 나열>
git commit -m "[feat] <작업 내용 요약>"
```
- **`git add -A` / `git add .` 절대 금지** — untracked 잡파일이 함께 커밋된다.
- 커밋 메시지는 기존 컨벤션(`[feat]`, `[chore]`, `[fix]`)을 따른다.

### 6-A. 성공 시 — 보고 및 승인 대기
```bash
git diff $ORIGINAL_BRANCH..HEAD --stat
```
다음을 보고하고 **사용자 응답을 기다린다**:
- 변경 파일 목록 및 주요 변경 요약
- 검증 결과 (실행한 명령 + 결과)
- 발생한 문제와 해결 방법

> ✅ 작업 및 검증 완료. "merge" 또는 "discard"로 답해주세요.

### 6-B. 실패 시
- 시도한 내용과 실패 원인 보고
- 원래 브랜치로 복귀하되, **작업 브랜치와 `.claude/tasks/<branch-name>.md`는 삭제하지 않고 유지** (재시도용)
```bash
git checkout $ORIGINAL_BRANCH
```

### 7-A. "merge" 승인 시
```bash
git checkout $ORIGINAL_BRANCH && git merge --no-ff feature/<branch-name> -m "merge: <작업 내용 요약>" && git branch -d feature/<branch-name>
```
- 충돌 발생 시: merge를 중단하지 말고 사용자에게 충돌 파일 목록과 해결 방안을 보고 후 지시를 기다린다.
- merge 완료 후 `.claude/tasks/<branch-name>.md` 삭제

### 7-B. "discard" 시
```bash
git checkout $ORIGINAL_BRANCH && git branch -D feature/<branch-name>
```
- `.claude/tasks/<branch-name>.md` 삭제

## 주의사항
- 멀티라인 명령어는 항상 한 줄로 출력 (copy-paste 안전)
- `--no-ff`로 merge하여 히스토리를 명확하게 유지
