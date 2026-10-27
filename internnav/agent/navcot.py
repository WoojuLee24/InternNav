"""NavCoT: structured chain-of-thought reasoning step for S2 planner.

Wraps S2's text generation with a CoT prefix that elicits landmark
grounding and sub-goal prediction before the final action tokens.

Paper ref: Wei et al. (2022) Chain-of-Thought Prompting NeurIPS;
applied here as first CoT use in continuous VLN (NexusNav contribution C10).
"""

import re
from dataclasses import dataclass, field
from typing import Optional


NAVCOT_SYSTEM = (
    "You are a navigation assistant. Before predicting the next waypoint, "
    "reason step-by-step about: (1) what landmarks are visible, "
    "(2) which instruction step is currently active, "
    "(3) whether the goal is visible, "
    "(4) the best sub-goal direction. "
    "Format: <think>...</think><action>WAYPOINT or STOP</action>"
)

COT_PROMPT_TEMPLATE = (
    "Navigation instruction: {instruction}\n"
    "Steps completed: {steps_done}/{steps_total}\n"
    "Current position: heading={heading:.1f}deg elevation={elevation:.1f}deg\n"
    "Memory context: {memory_context}\n"
    "Visible landmarks: {landmarks}\n"
    "Task: Reason step-by-step, then predict the next waypoint direction or STOP.\n"
    "<think>"
)


@dataclass
class NavCoTOutput:
    reasoning: str
    action_text: str
    full_response: str
    cot_enabled: bool


class NavCoTWrapper:
    """Wraps S2 inference with structured chain-of-thought.

    Drop-in replacement for direct S2 prompt calls.
    Set enabled=False to fall back to direct prediction (zero overhead).
    """

    def __init__(self, enabled: bool = True, max_think_tokens: int = 128):
        self.enabled = enabled
        self.max_think_tokens = max_think_tokens
        self._step_count = 0

    def build_cot_prompt(
        self,
        instruction: str,
        steps_done: int = 0,
        steps_total: int = 10,
        heading: float = 0.0,
        elevation: float = 0.0,
        memory_context: str = "",
        landmarks: str = "none detected",
    ) -> str:
        return COT_PROMPT_TEMPLATE.format(
            instruction=instruction,
            steps_done=steps_done,
            steps_total=steps_total,
            heading=heading,
            elevation=elevation,
            memory_context=memory_context[:200],
            landmarks=landmarks[:100],
        )

    def build_cot_context_prefix(
        self,
        instruction: str,
        steps_done: int = 0,
        steps_total: int = 10,
        heading: float = 0.0,
        elevation: float = 0.0,
        memory_context: str = "",
        landmarks: str = "none detected",
    ) -> str:
        """Return a context prefix to prepend to memory_context for s2_step.

        Injects CoT elicitation into the memory context so s2_step reasons
        before committing to a waypoint. Lightweight — no additional forward pass.
        """
        if not self.enabled:
            return memory_context
        cot_prefix = (
            f"[NavCoT] Step {steps_done}/{steps_total} | "
            f"heading={heading:.0f}deg | "
            f"landmarks={landmarks[:80]} | "
            "Think: what landmark is next? Is goal visible? Best direction? "
        )
        return cot_prefix + memory_context

    def parse_cot_output(self, raw: str) -> NavCoTOutput:
        """Extract <think> and <action> blocks from raw model output."""
        think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
        action_match = re.search(r'<action>(.*?)</action>', raw, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else ""
        action_text = action_match.group(1).strip() if action_match else raw.strip()
        return NavCoTOutput(
            reasoning=reasoning,
            action_text=action_text,
            full_response=raw,
            cot_enabled=self.enabled,
        )

    def get_system_prompt(self) -> str:
        return NAVCOT_SYSTEM if self.enabled else ""

    def increment_step(self):
        self._step_count += 1

    def reset(self):
        self._step_count = 0

    @staticmethod
    def extract_landmarks_from_memory(memory_context: str) -> str:
        """Pull landmark mentions from WorldMemory context string."""
        matches = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', memory_context)
        return ', '.join(set(matches[:5])) if matches else "none detected"
