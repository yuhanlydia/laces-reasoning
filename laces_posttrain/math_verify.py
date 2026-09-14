"""Conservative answer extraction and verification for GSM8K and Hendrycks MATH.

This module deliberately does not use a CAS. Numeric answers are compared exactly via
``fractions.Fraction``; symbolic MATH answers must match after only wrapper/whitespace
normalization. Unsupported forms receive no credit.
"""
from __future__ import annotations

from fractions import Fraction
import re

VERIFIER_VERSION = "laces_math_verify_v1"


def _balanced_braced(text: str, brace_at: int) -> tuple[str, int] | None:
    if brace_at >= len(text) or text[brace_at] != "{":
        return None
    depth = 0
    for i in range(brace_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_at + 1 : i], i + 1
    return None


def _boxed_occurrences(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for m in re.finditer(r"\\boxed\s*", text):
        j = m.end()
        if j < len(text) and text[j] == "{":
            parsed = _balanced_braced(text, j)
            if parsed is not None:
                out.append((m.start(), parsed[0].strip()))
    return out


def _strip_wrappers(value: str) -> str:
    value = value.strip()
    while value.startswith("$") and value.endswith("$") and len(value) >= 2:
        value = value[1:-1].strip()
    if value.startswith(r"\boxed"):
        m = re.match(r"\\boxed\s*", value)
        if m and m.end() < len(value) and value[m.end()] == "{":
            parsed = _balanced_braced(value, m.end())
            if parsed and not value[parsed[1]:].strip():
                value = parsed[0].strip()
    while value.startswith("{") and value.endswith("}"):
        parsed = _balanced_braced(value, 0)
        if parsed is None or parsed[1] != len(value):
            break
        value = parsed[0].strip()
    return value


def extract_final_answer(text: str) -> str | None:
    """Return the last explicitly marked answer, never an arbitrary prose substring."""
    candidates: list[tuple[int, str]] = []
    for m in re.finditer(r"####\s*([^\r\n]+)", text):
        candidates.append((m.start(), m.group(1).strip().rstrip(". ;")))
    explicit_patterns = (r"final\s+answer\s*[:=]\s*([^\r\n]+)", r"the\s+answer\s+is\s+([^\r\n]+)")
    for pattern in explicit_patterns:
        for m in re.finditer(pattern, text, re.I):
            value = m.group(1).strip()
            # Keep one explicit answer expression rather than the following prose.
            value = re.split(r"(?<=[0-9A-Za-z}])\s*[.;](?:\s|$)", value, maxsplit=1)[0].strip()
            candidates.append((m.start(), value.rstrip(". ;")))
    candidates.extend(_boxed_occurrences(text))
    if not candidates:
        return None
    value = max(candidates, key=lambda x: x[0])[1]
    return _strip_wrappers(value) or None


def canonical_numeric(text: str) -> str | None:
    """Convert a conservative numeric/LaTeX-fraction form to an exact reduced fraction."""
    value = _strip_wrappers(text).strip().replace(",", "")
    value = re.sub(r'^\\+', r'\\', value)
    value = value.replace(r"\,", "").replace(" ", "")
    frac = re.fullmatch(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", value)
    try:
        if frac:
            num = canonical_numeric(frac.group(1))
            den = canonical_numeric(frac.group(2))
            if num is None or den is None:
                return None
            q = Fraction(num) / Fraction(den)
        elif re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
            q = Fraction(value)
        elif re.fullmatch(r"[+-]?\d+/[+-]?\d+", value):
            q = Fraction(value)
        else:
            return None
    except (ValueError, ZeroDivisionError):
        return None
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def _canonical_symbolic(text: str) -> str:
    value = _strip_wrappers(text)
    value = value.replace("$", "")
    return re.sub(r"\s+", "", value)


def verify_answer(text: str, gold: str, task: str) -> bool:
    task = task.lower()
    if task not in {"gsm8k", "math"}:
        raise ValueError("task must be gsm8k or math")
    pred = extract_final_answer(text)
    if pred is None:
        return False
    pred_num, gold_num = canonical_numeric(pred), canonical_numeric(gold)
    if pred_num is not None and gold_num is not None:
        return pred_num == gold_num
    if task == "gsm8k":
        return False
    # No algebraic-equivalence claims: exact normalized symbolic surface only.
    return _canonical_symbolic(pred) == _canonical_symbolic(gold)
