---
name: 기존 파일 수정 금지 — 신규 파일만 추가
description: 코드 변경 시 기존 파일을 수정하지 말고 신규 파일을 만들어 arg/config로 선택
type: feedback
---

기능 추가 시 기존 파일을 수정하지 말고 새로운 파일을 만들고, config arg로 신규/기존 클래스를 선택하도록 설계할 것.

**Why:** 기존 코드의 안정성 유지. 신규 파일에서 기존 클래스를 상속하고, Registry(`@Agent.register`, `@Evaluator.register`)로 디스패치. config의 `eval_type` 또는 `model_name` 변경만으로 선택 가능해야 함.

**How to apply:** 새 feature 제안 시 항상 "신규 파일 + config arg 선택" 구조를 우선 제시. 기존 파일 수정이 포함된 계획은 먼저 거절당할 것.
