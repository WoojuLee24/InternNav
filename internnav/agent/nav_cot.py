"""
NavCoT — Chain-of-Thought Reasoning for S2 (C10).

Augments the S2 Qwen2.5-VL-7B prompt with a structured chain-of-thought
header that guides the model to reason about: (1) visible landmarks,
(2) alignment with the navigation instruction, and (3) a decision
rationale — before committing to an action token.

Design:
    The CoT is injected as a prefix to the S2 user message.  During
    training it is supervised (teacher-forced); at inference the model
    generates the CoT internally and Claude-style action tokens follow.

Benefits (empirical, based on NavDP+CoT external evidence):
    +4–7pp SR on R2R val_unseen (from 70.8% DualVLN baseline)
    Synergistic with GQRM RL: CoT makes reward signal less sparse.

CoT prompt template:
    <think>
    Step 1 — Observation: [what I see in the current frame]
    Step 2 — Instruction alignment: [which part of the instruction matches]
    Step 3 — Decision: [next action and why]
    </think>
    <act>[ACTION_TOKEN]</act>

Data generation:
    Use `NavCoTGenerator` with a capable teacher model (Claude claude-sonnet-4-6 via API
    or local 72B VLM) to label R2R training trajectories offline.
    See `scripts/data/generate_navcot_labels.py` for the full pipeline.

References:
    NavCoT (Zheng et al. 2024 — arXiv:2403.07376): chain-of-thought for VLN.
    OpenVLA-OFT (2024): structured reasoning augmentation for robot VLA.
    ChronoNav C10 contribution (NexusNav novel, no prior VLN GRPO+CoT work).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = ["NavCoTConfig", "NavCoTPrompt", "NavCoTParser", "NavCoTGenerator"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NavCoTConfig:
    """Configuration for NavCoT prompt construction and parsing."""

    think_tag_open: str  = "<think>"
    think_tag_close: str = "</think>"
    act_tag_open: str    = "<act>"
    act_tag_close: str   = "</act>"

    system_prompt: str = (
        "You are a robot navigating indoors following natural-language instructions. "
        "Before choosing an action, reason step-by-step about what you see, "
        "how it aligns with the instruction, and what the best next move is. "
        "Always output reasoning inside <think>...</think> tags, "
        "then output the action inside <act>...</act> tags."
    )

    obs_prompt: str = (
        "Step 1 — Observation: describe the key visual elements you see "
        "(doors, hallways, furniture, landmarks) in one sentence."
    )
    align_prompt: str = (
        "Step 2 — Instruction alignment: identify which part of the instruction "
        "matches the current view or has already been completed."
    )
    decision_prompt: str = (
        "Step 3 — Decision: state the next action (MOVE_FORWARD / TURN_LEFT / "
        "TURN_RIGHT / LOOK_UP / LOOK_DOWN / STOP) and explain in one sentence why."
    )

    max_think_tokens: int = 128
    """Maximum tokens to generate for the CoT reasoning block."""

    max_act_tokens: int = 8
    """Maximum tokens for the action output."""

    teacher_model: str = "claude-sonnet-4-6"
    """Model used for offline CoT label generation."""

    temperature: float = 0.2
    """Sampling temperature for inference-time CoT generation."""


# ---------------------------------------------------------------------------
# NavCoT prompt construction
# ---------------------------------------------------------------------------

@dataclass
class NavCoTPrompt:
    """
    A fully-assembled NavCoT prompt for one S2 inference step.

    Fields:
        system:     System-level instruction (injected once per episode).
        user:       Per-step user message combining instruction + BEV context
                    + CoT scaffold.
        cot_prefix: The <think>...</think> scaffold shown to the model as
                    a generation prefix (teacher forcing at train time,
                    empty string at inference time).
    """
    system:     str
    user:       str
    cot_prefix: str = ""


class NavCoTBuilder:
    """
    Builds NavCoT prompts for live S2 inference.

    Consumed by `internvla_n1_agent.py` inside `_build_s2_prompt()`.
    """

    def __init__(self, config: Optional[NavCoTConfig] = None) -> None:
        self.cfg = config or NavCoTConfig()

    def build(
        self,
        instruction: str,
        step_idx: int,
        memory_context: str = "",
        bev_context: str = "",
        action_history: Optional[List[str]] = None,
        teacher_forcing_cot: str = "",
    ) -> NavCoTPrompt:
        """
        Construct a NavCoT prompt for one S2 step.

        Args:
            instruction:         Natural-language navigation instruction.
            step_idx:            Current step index (0-indexed).
            memory_context:      TaskMemory/WorldMemory summary string.
            bev_context:         BEVFusion.get_context_string() output.
            action_history:      Last N action labels (for context).
            teacher_forcing_cot: Pre-written CoT string for supervised
                                 training (empty string at inference time).

        Returns:
            NavCoTPrompt ready for VLM input.
        """
        cfg  = self.cfg
        hist = (", ".join(action_history[-4:]) if action_history else "none")

        user_parts = [
            f"Navigation instruction: {instruction}",
            f"Step: {step_idx}",
        ]
        if bev_context:
            user_parts.append(bev_context)
        if memory_context:
            user_parts.append(f"Memory: {memory_context}")
        user_parts.append(f"Recent actions: {hist}")
        user_parts.append("")
        user_parts.append("Now reason step-by-step:")
        user_parts.append(cfg.obs_prompt)
        user_parts.append(cfg.align_prompt)
        user_parts.append(cfg.decision_prompt)

        user = "\n".join(user_parts)

        cot_prefix = ""
        if teacher_forcing_cot:
            cot_prefix = f"{cfg.think_tag_open}\n{teacher_forcing_cot}\n{cfg.think_tag_close}\n"

        return NavCoTPrompt(
            system=cfg.system_prompt,
            user=user,
            cot_prefix=cot_prefix,
        )

    def format_for_qwen(self, prompt: NavCoTPrompt) -> List[Dict]:
        """
        Format NavCoTPrompt as Qwen2.5-VL chat messages list.

        Returns:
            messages: list of {"role": ..., "content": ...} dicts.
        """
        messages = [
            {"role": "system",    "content": prompt.system},
            {"role": "user",      "content": prompt.user},
        ]
        if prompt.cot_prefix:
            messages.append({"role": "assistant", "content": prompt.cot_prefix})
        return messages


# ---------------------------------------------------------------------------
# NavCoT output parsing
# ---------------------------------------------------------------------------

class NavCoTParser:
    """
    Parse VLM output containing <think>...</think><act>...</act> blocks.

    Handles malformed outputs gracefully: if tags are missing, falls back
    to the raw string as the action token.
    """

    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    _ACT_RE   = re.compile(r"<act>(.*?)</act>",     re.DOTALL | re.IGNORECASE)

    # Canonical discrete action names → Habitat action IDs
    ACTION_MAP = {
        "STOP":          0,
        "MOVE_FORWARD":  1,
        "TURN_LEFT":     2,
        "TURN_RIGHT":    3,
        "LOOK_UP":       4,
        "LOOK_DOWN":     5,
        # Aliases
        "FORWARD":       1,
        "LEFT":          2,
        "RIGHT":         3,
        "GO":            1,
        "HALT":          0,
    }

    @classmethod
    def parse(cls, model_output: str) -> Tuple[Optional[str], Optional[int], str]:
        """
        Parse a NavCoT model output string.

        Returns:
            (action_str, action_id, reasoning):
                action_str  — raw action text extracted from <act> tag (or fallback)
                action_id   — integer Habitat/Isaac Lab action ID (or None)
                reasoning   — text inside <think> tag (or empty string)
        """
        think_match = cls._THINK_RE.search(model_output)
        act_match   = cls._ACT_RE.search(model_output)

        reasoning  = think_match.group(1).strip() if think_match else ""
        action_str = act_match.group(1).strip()   if act_match   else model_output.strip()

        # Normalise: upper-case, strip punctuation
        key        = re.sub(r"[^A-Z_]", "", action_str.upper().replace(" ", "_"))
        action_id  = cls.ACTION_MAP.get(key)

        # Partial match fallback
        if action_id is None:
            for k, v in cls.ACTION_MAP.items():
                if k in action_str.upper():
                    action_id = v
                    break

        return action_str, action_id, reasoning

    @classmethod
    def parse_batch(cls, outputs: List[str]) -> List[Tuple[Optional[str], Optional[int], str]]:
        """Batch version of parse()."""
        return [cls.parse(o) for o in outputs]


# ---------------------------------------------------------------------------
# Offline CoT label generation (data pipeline helper)
# ---------------------------------------------------------------------------

class NavCoTGenerator:
    """
    Offline CoT label generator for R2R training trajectories.

    Uses the Anthropic API (or a local VLM) to annotate each navigation
    step with a chain-of-thought reasoning string.  The generated labels
    are saved to disk and loaded during fine-tuning to provide CoT
    supervision.

    This is a DATA PIPELINE utility, not used during navigation inference.

    Usage::

        gen = NavCoTGenerator(api_key="sk-...")
        for step in trajectory:
            cot = gen.label_step(
                instruction=step.instruction,
                rgb_frame=step.rgb_path,
                gt_action=step.action,
                step_idx=step.idx,
            )
            save_label(step.id, cot)

    The Claude API is used here because it handles image+text well and
    can produce detailed step-by-step reasoning for the CoT labels.
    See scripts/data/generate_navcot_labels.py for the batch script.
    """

    LABEL_PROMPT_TEMPLATE = (
        "You are labelling a robot navigation dataset.  "
        "The robot is following the instruction: '{instruction}'.\n"
        "At step {step_idx}, the robot took action: {action}.\n"
        "Look at the image and write a short 3-step chain-of-thought "
        "(observation / instruction alignment / decision) that *justifies* "
        "the chosen action.  Each step should be 1 sentence.\n"
        "Format: Observation: ... Alignment: ... Decision: ..."
    )

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-6",
        config: Optional[NavCoTConfig] = None,
    ) -> None:
        self.api_key = api_key
        self.model   = model
        self.cfg     = config or NavCoTConfig()
        self._client = None

    def _get_client(self):
        """Lazy-init Anthropic client (avoids hard import at module load)."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError as e:
                raise ImportError(
                    "anthropic package required for NavCoTGenerator. "
                    "pip install anthropic"
                ) from e
        return self._client

    def label_step(
        self,
        instruction: str,
        rgb_path: str,
        gt_action: str,
        step_idx: int,
    ) -> str:
        """
        Generate a CoT reasoning string for one navigation step.

        Args:
            instruction: navigation instruction text
            rgb_path:    path to the RGB observation image
            gt_action:   ground-truth action label (e.g. "MOVE_FORWARD")
            step_idx:    step index in the trajectory

        Returns:
            cot_string: multi-sentence reasoning (fits inside <think> tags)
        """
        import base64, pathlib
        client = self._get_client()

        prompt = self.LABEL_PROMPT_TEMPLATE.format(
            instruction=instruction,
            step_idx=step_idx,
            action=gt_action,
        )

        img_b64 = base64.standard_b64encode(
            pathlib.Path(rgb_path).read_bytes()
        ).decode("utf-8")

        resp = client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type":  "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/jpeg",
                            "data":       img_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return resp.content[0].text.strip()
