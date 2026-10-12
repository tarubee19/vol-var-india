"""Numeric grounding check: every number in an answer must trace back to evidence.

The evidence is whatever the tools returned during the turn (including retrieved knowledge-base
text) plus the numbers the user typed. A number counts as supported when it equals an evidence
number up to rounding to the precision it was written with (so ``1.28`` matches ``1.2802``). Rates
stored as fractions also match their percent form. This catches invented or mis-copied figures; it
does not prove the *wording* around a number is right.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")

# Significance thresholds the assistant may cite without them appearing in tool output
# ("p < 0.05", "p < 0.001", "at the 5% level").
THRESHOLD_CONSTANTS = frozenset({0.001, 0.01, 0.05, 5.0})


@dataclass(frozen=True)
class GroundingReport:
    ok: bool
    unsupported_numbers: list[str]
    unsupported_dates: list[str]


def _numbers(text: str) -> list[tuple[str, float]]:
    out = []
    for token in _NUMBER.findall(_DATE.sub(" ", text)):
        out.append((token, float(token.replace(",", ""))))
    return out


def _walk(obj: Any, nums: set[float], dates: set[str]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        nums.add(abs(float(obj)))
    elif isinstance(obj, str):
        dates.update(_DATE.findall(obj))
        for _, value in _numbers(obj):
            nums.add(abs(value))
    elif isinstance(obj, dict):
        for value in obj.values():
            _walk(value, nums, dates)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _walk(value, nums, dates)


def _decimals(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def verify_answer(answer: str, question: str, evidence: list[Any]) -> GroundingReport:
    """Check that numbers and dates in ``answer`` are supported by ``evidence``/``question``."""
    nums: set[float] = set()
    dates: set[str] = set()
    for item in evidence:
        _walk(item, nums, dates)
    _walk(question, nums, dates)
    supported = nums | {n * 100 for n in nums} | THRESHOLD_CONSTANTS

    bad_numbers = []
    for token, value in _numbers(answer):
        tol = 0.5 * 10 ** (-_decimals(token)) + 1e-9
        if not any(abs(value - s) <= tol for s in supported):
            bad_numbers.append(token)
    bad_dates = [d for d in _DATE.findall(answer) if d not in dates]
    return GroundingReport(not bad_numbers and not bad_dates, bad_numbers, bad_dates)
