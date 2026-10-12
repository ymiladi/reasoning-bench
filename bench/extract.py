"""Answer extraction from a model's *final* response (never from its reasoning trace)."""
from __future__ import annotations

import json
import re

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str | None) -> str:
    """Remove any <think>...</think> block that leaked into the final content.

    If an opening <think> is never closed, the model ran out of budget mid-thought,
    so there is no final answer at all: return "".
    """
    if not text:
        return ""
    text = THINK_RE.sub("", text)
    if "<think>" in text:
        return ""
    # Some templates emit only the closing tag (the opening one is in the prompt).
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (or \\fbox{...}), with balanced braces."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        # "\boxed 5" form
        m = re.match(r"\\boxed\s+([^\s$]+)", text[idx:])
        return m.group(1) if m else None
    depth, start = 0, i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : j].strip()
    return None


_LETTER_PATTERNS = [
    r"(?i)answer\s*(?:is)?\s*[:：]?\s*\**\s*\(?([A-D])\)?\b",
    r"\\boxed\{\s*\(?([A-D])\)?\s*\}",
    r"(?i)correct (?:option|choice|answer) is\s*\**\s*\(?([A-D])\)?\b",
]


def mcq_letter(text: str) -> str | None:
    """Last stated option letter A-D."""
    found: list[tuple[int, str]] = []
    for pat in _LETTER_PATTERNS:
        for m in re.finditer(pat, text):
            found.append((m.start(), m.group(1).upper()))
    if found:
        return max(found)[1]
    # Bare letter as the entire final line, e.g. "**C**"
    last = text.strip().splitlines()[-1] if text.strip() else ""
    m = re.fullmatch(r"[\s*(\[]*([A-D])[\s*)\].]*", last)
    return m.group(1) if m else None


def code_block(text: str) -> str | None:
    """Last fenced python code block (falls back to any fenced block)."""
    blocks = re.findall(r"```(?:python|py|Python3?)\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```[a-zA-Z0-9]*\s*\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else None


def last_json_object(text: str) -> dict | None:
    """Last parseable top-level JSON object in the text (handles ```json fences)."""
    fenced = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    for cand in reversed(fenced):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # Scan backwards for balanced {...}
    ends = [i for i, c in enumerate(text) if c == "}"]
    for end in reversed(ends):
        depth = 0
        for start in range(end, -1, -1):
            if text[start] == "}":
                depth += 1
            elif text[start] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : end + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass
                    break
    return None
