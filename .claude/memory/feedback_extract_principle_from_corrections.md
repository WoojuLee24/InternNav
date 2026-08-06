---
name: feedback_extract_principle_from_corrections
description: 개발 중 실수/의도불일치로 사용자가 수정을 지시하면, 그 교정에서 일반원리를 추출해 재발 방지 규칙으로 저장한다
metadata:
  type: feedback
---

Claude가 개발 중 실수하거나 의도와 다르게 작업해서 사용자가 수정을 지시하면, 그 자리에서 고치고 끝내지 말고 원인을 파악해 일반원리를 추출한 뒤 저장한다.

**Why:** 사용자가 매번 같은 종류의 실수를 반복해서 지적하는 걸 원하지 않음. 매번 즉흥적으로 하지 말고 정해진 절차로 하길 원함 (2026-07-21).

**How to apply:** 절차는 `lesson` 스킬(`.agents/skills/lesson/SKILL.md`, `.claude/commands/lesson.md`)로 정의돼 있음 — 교정을 받으면 이 스킬을 사용한다.
