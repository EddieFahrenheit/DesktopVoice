from __future__ import annotations

import re
from typing import Literal

Command = Literal["open_gemini", "open_chatgpt", "mic", "stop"]

gemini_aliases = {
    "google",
    "gemini",
    "open gemini",
    "open google",
    "kevin i",
    "germany",
    "good night",
    "see you next time",
    "bye bye",
    "bye",
    "go go",
    "let's go",
    "go home",
}

chat_aliases = {
    "chat",
    "check",
    "chat me",
    "open chat",
    "over chat",
    "over the track",
    "cut",
    "yeah",
}

mic_aliases = {
    "mic",
    "mike",
    "like",
    "ask",
    "microphone",
    "voice",
    "please",
    "start voice",
    "boys",
    "nice",
    "oh its"
}

stop_aliases = {
    "stop",
    "cancel",
    "nevermind",
    "never mind",
}

def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)   # drop punctuation
    text = re.sub(r"\s+", " ", text)       # collapse whitespace
    return text.strip()


def parse_command(text: str) -> Command | None:
    """
    Convert transcribed text into a small set of exact commands.
    """
    t = _normalize(text)

    if t in gemini_aliases:
        return "open_gemini"

    if t in chat_aliases:
        return "open_chatgpt"
    
    if t in mic_aliases:
        return "mic"
    
    if t in stop_aliases:
        return "stop"

    return None
