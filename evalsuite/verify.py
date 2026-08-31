"""Verification of model answers: exact match, sandboxed tests, LLM judge."""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

from evalsuite.tasks import check_math

CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

JUDGE_SYSTEM_PROMPT = """You grade an answer to a question against reference notes.
Score 5: fully correct and complete. 3: partially correct or missing a key point.
1: wrong or off-topic. Use the whole 1-5 range.
Respond with JSON: {"score": <1-5>, "rationale": "<one short sentence>"}"""


def extract_code(response: str) -> str:
    match = CODE_BLOCK_RE.search(response)
    return match.group(1) if match else response


def run_code_tests(task: Dict, response: str, timeout_s: int = 8) -> float:
    """Write the model's code and the task's asserts to a temp dir and run them."""
    code = extract_code(response)
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "solution.py").write_text(code, encoding="utf-8")
        Path(tmp, "check.py").write_text(
            "import os, sys\n"
            "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
            "import solution\n" + task["test"] + "\nprint('PASS')\n",
            encoding="utf-8",
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "check.py"],
                cwd=tmp, capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return 0.0
    return 1.0 if proc.returncode == 0 and "PASS" in proc.stdout else 0.0


def judge_qa(task: Dict, response: str, provider, judge_tier: str = "large") -> Dict:
    payload = {
        "question": task["prompt"],
        "reference_notes": task["reference"],
        "answer_to_grade": response,
    }
    result = provider.chat(
        judge_tier,
        [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
         {"role": "user", "content": json.dumps(payload)}],
    )
    match = re.search(r'"score"\s*:\s*(\d)', result.text)
    score = int(match.group(1)) if match else 1
    return {"score": max(1, min(5, score)), "judge_cost_usd": result.cost_usd}


def quality_of(task: Dict, response: str, provider=None) -> float:
    """Normalized 0-1 quality for any task kind."""
    if task["kind"] == "math":
        return check_math(task, response)
    if task["kind"] == "code":
        return run_code_tests(task, response)
    if task["kind"] == "qa":
        verdict = judge_qa(task, response, provider)
        return (verdict["score"] - 1) / 4.0
    raise ValueError(task["kind"])
