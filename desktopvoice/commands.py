from __future__ import annotations

import re
from typing import Literal

Command = Literal["open_gemini", "open_chatgpt"]


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)   # drop punctuation
    text = re.sub(r"\s+", " ", text)       # collapse whitespace
    return text.strip()


def parse_command(text: str) -> Command | None:
    """
    Convert transcribed text into a small set of exact commands.

    We keep this strict for Day 1 (cheap + reliable): only a few phrases map to actions.
    """
    t = _normalize(text)

    if t == "open gemini":
        return "open_gemini"

    if t == "open chat":
        return "open_chatgpt"

    return None
