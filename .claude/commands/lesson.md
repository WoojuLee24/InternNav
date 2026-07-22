---
allowed-tools: Read, Edit, Write, Bash(git log:*)
description: 사용자가 "lesson"이라는 단어를 명시적으로 말했을 때만 사용. 방금 받은 교정(실수 지적)의 원인을 파악하고 일반원리를 추출해 guideline.md 또는 feedback 메모리로 저장
---

## Task

사용자가 방금 Claude의 작업을 교정했다(실수를 지적했거나, 의도와 다르다고 수정을 지시했다). 그 건을 고치고 끝내지 말고:

1. **원인 파악**: 무엇을 잘못했는지, 왜 그 접근이 틀렸는지 한 문장으로 정리한다.
2. **일반원리 추출**: 이번 사례에 국한되지 않게 추상화한다 ("다음에 어떤 상황에서 이 원리를 다시 적용해야 하는가?").
3. **저장 위치 판단**:
   - **코딩/구현 패턴**에 관한 것이면 → `.claude/rules/guideline.md`의 "개발 제한사항" 섹션에 기존 규칙(1~7)과 같은 형식(규칙 + 실제 겪은 사례 + 해결)으로 새 번호 규칙을 추가한다.
   - **협업 방식/워크플로우**에 관한 것이면 → `.claude/memory/feedback_<slug>.md` 신규 작성(name/description/type: feedback frontmatter, 규칙 + **Why:** + **How to apply:**), `.claude/memory/MEMORY.md`에 인덱스 한 줄 추가, `/root/.claude/projects/-ws-src-InternNav/memory/`(auto-memory)에도 동일 파일 + 인덱스 동기화.
4. 어디에 뭘 저장했는지 한 줄로 사용자에게 보고한다. 말로만 "다음엔 안 그럴게요" 하지 않는다.

## 주의
- 이미 같은 원리를 다루는 규칙/메모리가 있으면 새로 만들지 말고 기존 항목을 갱신한다.
- 코드 특정 사실(파일 경로, 함수명 등)이 아니라 **재사용 가능한 판단 기준**을 남긴다.
