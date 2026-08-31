"""Prompt featurizer shared by every router generation.

Deliberately hand-crafted and explainable: each dimension has a name, so any
routing decision can be traced to the signals that drove it.
"""

import re
from typing import Dict, List, Tuple

CODE_KEYWORDS = ("def ", "function", "implement", "python", "regex", "sql",
                 "class ", "algorithm", "bug", "compile", "refactor", "unit test")
MATH_KEYWORDS = ("calculate", "how many", "how much", "average", "percent",
                 "total", "sum of", "solve", "remainder", "divided")
REASONING_KEYWORDS = ("why", "prove", "explain", "compare", "analyze",
                      "trade-off", "tradeoff", "step by step", "derive")
SIMPLE_KEYWORDS = ("what is", "who is", "when was", "capital of", "define",
                   "meaning of", "translate")
MULTISTEP_MARKERS = ("then", "after that", "first", "second", "finally",
                     "each of", "for every")

FEATURE_NAMES: List[str] = [
    "n_chars", "n_words", "avg_word_len", "digit_ratio", "math_op_count",
    "question_marks", "code_kw", "math_kw", "reasoning_kw", "simple_kw",
    "multistep_kw", "line_count", "has_code_block", "upper_ratio",
    "comma_count", "number_count",
]


def _count(text: str, keywords) -> int:
    lowered = text.lower()
    return sum(1 for k in keywords if k in lowered)


def featurize(text: str) -> Tuple[List[float], Dict[str, float]]:
    words = text.split()
    n_chars = len(text)
    n_words = max(len(words), 1)
    digits = sum(c.isdigit() for c in text)
    uppers = sum(c.isupper() for c in text)
    values = [
        min(n_chars / 1000.0, 3.0),
        min(n_words / 200.0, 3.0),
        sum(len(w) for w in words) / n_words / 10.0,
        digits / max(n_chars, 1),
        min(len(re.findall(r"[+\-*/=%^]", text)) / 5.0, 3.0),
        min(text.count("?"), 5) / 5.0,
        min(_count(text, CODE_KEYWORDS), 4) / 4.0,
        min(_count(text, MATH_KEYWORDS), 4) / 4.0,
        min(_count(text, REASONING_KEYWORDS), 4) / 4.0,
        min(_count(text, SIMPLE_KEYWORDS), 3) / 3.0,
        min(_count(text, MULTISTEP_MARKERS), 4) / 4.0,
        min(text.count("\n") / 10.0, 3.0),
        1.0 if "```" in text else 0.0,
        uppers / max(n_chars, 1),
        min(text.count(",") / 10.0, 3.0),
        min(len(re.findall(r"\d+(?:\.\d+)?", text)) / 8.0, 3.0),
    ]
    return values, dict(zip(FEATURE_NAMES, values))
