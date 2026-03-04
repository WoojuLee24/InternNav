---
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git add:*), Bash(git commit:*)
description: Analyze staged changes and create a conventional commit message, then commit
---

## Context

- Git status: !`git status`
- Staged diff: !`git diff --cached`

## Task

위 git 변경사항을 분석해서 **conventional commit** 형식으로 커밋 메시지를 생성하고 확인 후 커밋, push까지 진행해줘.

### Commit Message Rules

형식: `<type>(<scope>): <subject>`

**type** (아래 중 하나):
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `refactor`: 리팩토링 (기능 변화 없음)
- `docs`: 문서 수정
- `chore`: 빌드, 설정, 패키지 등 기타 변경
- `test`: 테스트 추가/수정
- `perf`: 성능 개선
- `style`: 코드 스타일 변경 (포맷, 세미콜론 등)

**scope** (선택, 변경된 모듈/영역):
- 예: `navigation`, `mapping`, `ros2`, `docker`, `vln`, `slam`, `sensor`
- 변경사항이 여러 영역에 걸치면 생략 가능

**subject**:
- 영어로 작성, 소문자 시작, 마침표 없음
- 명령형 동사 사용 (add, fix, update, remove 등)
- 50자 이내

### Steps

1. staged 변경사항이 없으면 `git status`를 보여주고 "No staged changes. Please run `git add` first." 메시지 출력 후 종료
2. diff를 분석해서 적절한 커밋 메시지 생성
3. 생성한 메시지를 출력하고 **유저에게 확인 요청** ("위 메시지로 커밋할까요? (y/n/edit)")
   - `n`: 커밋 취소 후 종료
   - `edit`: 유저가 직접 수정할 메시지를 입력받아 사용
   - `y`: 다음 단계 진행
4. `git commit -m "<message>"` 실행
5. 커밋 완료 후 **push 여부 확인** ("git push 할까요? (y/n)")
   - `n`: 종료
   - `y`: `git push` 실행 후 결과 출력