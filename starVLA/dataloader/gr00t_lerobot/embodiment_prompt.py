"""Structured Embodiment Prompt helpers (paper §3.4).

Renders episode-level ``prompt_fields`` with a frame-level instruction override.
Optionally drops ``embodiment`` / ``speed`` / ``fps`` independently at train time.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

# Fields that may be randomly omitted when field-dropout is enabled.
DROPPABLE_FIELDS: tuple[str, ...] = ("embodiment", "speed", "fps")

# Exact junk labels seen in Galaxea / RoboCOIN task annotations.
_JUNK_EXACT: frozenset[str] = frozenset(
    {
        "",
        "null",
        "none",
        "nan",
        "qualified",
        "unqualified",
        "handle plates",
    }
)

# Render order / display keys matching the paper example.
_FIELD_LINES: tuple[tuple[str, str], ...] = (
    ("embodiment", "embodiment"),
    ("instruction", "instruction"),
    ("speed", "speed"),
    ("fps", "fps"),
    ("camera_view_direction", "camera view direction"),
)


def clean_frame_instruction(task: str | None) -> str | None:
    """Normalize a frame-level task string; return None if unusable."""
    if task is None or not isinstance(task, str):
        return None
    text = task.strip()
    if not text:
        return None

    # Bilingual RoboCOIN style: "中文@English" → prefer English.
    if "@" in text:
        left, right = text.split("@", 1)
        right = right.strip()
        left = left.strip()
        text = right if right else left

    if text.lower() in _JUNK_EXACT:
        return None

    # Dataset-name style labels, e.g. "organize refrigerator items0250703"
    compact = text.replace(" ", "").lower()
    if "0250703" in compact or compact.endswith("items0250703"):
        return None
    if " " not in text and any(ch.isdigit() for ch in text):
        return None

    return text


def format_embodiment_prompt(
    prompt_fields: Mapping[str, Any],
    instruction: str,
    *,
    field_dropout: bool = False,
    field_dropout_prob: float = 0.15,
    droppable_fields: Sequence[str] = DROPPABLE_FIELDS,
    rng: random.Random | None = None,
) -> str:
    """Build a multiline structured embodiment prompt.

    Args:
        prompt_fields: Episode-level fields from ``meta/episodes.jsonl``.
        instruction: Frame-level (or fallback) task text already cleaned.
        field_dropout: If True, independently drop each droppable field with
            ``field_dropout_prob`` (paper: embodiment / speed / fps @ 15%).
        field_dropout_prob: Per-field drop probability.
        droppable_fields: Which keys participate in dropout.
        rng: Optional RNG for deterministic tests.
    """
    values: dict[str, Any] = {
        "embodiment": prompt_fields.get("embodiment"),
        "instruction": instruction,
        "speed": prompt_fields.get("speed"),
        "fps": prompt_fields.get("fps"),
        "camera_view_direction": prompt_fields.get("camera_view_direction"),
    }

    if field_dropout and field_dropout_prob > 0:
        _rng = rng if rng is not None else random
        drop_set = set(droppable_fields)
        for key in list(values.keys()):
            if key in drop_set and _rng.random() < field_dropout_prob:
                values[key] = None

    lines: list[str] = []
    for key, label in _FIELD_LINES:
        val = values.get(key)
        if val is None or val == "":
            continue
        lines.append(f"{label}: {val}")
    return "\n".join(lines)
