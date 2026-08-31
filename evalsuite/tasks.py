"""Evaluation tasks across three verification regimes.

math: exact-answer arithmetic and word problems, checked by parsing the reply
code: function-writing tasks, checked by running unit tests in a subprocess
qa:   open questions, scored 1-5 by a judge model against reference notes

Difficulty is mixed on purpose: the suite exists to measure where the small
model stops being sufficient, not to flatter either model.
"""

import random
import re
from typing import Dict, List

ANSWER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def math_tasks(seed: int = 11) -> List[Dict]:
    rng = random.Random(seed)
    tasks = []

    for i in range(10):  # easy: single-step arithmetic
        a, b = rng.randint(12, 97), rng.randint(12, 97)
        tasks.append({
            "id": f"math_easy_{i}", "kind": "math", "difficulty": "easy",
            "prompt": f"What is {a} + {b}? Answer with just the number.",
            "answer": float(a + b),
        })

    for i in range(8):  # medium: two-step percent problems
        price, pct = rng.randint(40, 400), rng.choice([10, 15, 20, 25, 30])
        tasks.append({
            "id": f"math_med_{i}", "kind": "math", "difficulty": "medium",
            "prompt": (f"A jacket costs ${price}. It is discounted by {pct}%, and then a "
                       f"10% tax is added to the discounted price. What is the final "
                       f"price in dollars? Answer with just the number."),
            "answer": round(price * (1 - pct / 100) * 1.1, 2),
        })

    for i in range(8):  # hard: multi-step rate/work problems
        w1, w2 = rng.randint(3, 7), rng.randint(8, 14)
        tasks.append({
            "id": f"math_hard_{i}", "kind": "math", "difficulty": "hard",
            "prompt": (f"Worker A finishes a job alone in {w1} hours; worker B alone in "
                       f"{w2} hours. They work together for 1 hour, then A leaves. How "
                       f"many additional hours does B need to finish the job alone? "
                       f"Round to 3 decimal places. Answer with just the number."),
            "answer": round((1 - (1 / w1 + 1 / w2)) * w2, 3),
        })
    return tasks


def check_math(task: Dict, response: str) -> float:
    numbers = ANSWER_RE.findall(response.replace(",", ""))
    if not numbers:
        return 0.0
    try:
        value = float(numbers[-1])
    except ValueError:
        return 0.0
    return 1.0 if abs(value - task["answer"]) < 0.02 else 0.0


CODE_TASKS: List[Dict] = [
    {
        "id": "code_reverse_words", "difficulty": "easy",
        "prompt": "Write a Python function reverse_words(s) that reverses the order of words in a string. Words are separated by single spaces.",
        "test": "assert solution.reverse_words('a b c') == 'c b a'\nassert solution.reverse_words('hello') == 'hello'",
    },
    {
        "id": "code_second_largest", "difficulty": "easy",
        "prompt": "Write a Python function second_largest(nums) returning the second largest distinct value in a list of ints. Assume at least two distinct values.",
        "test": "assert solution.second_largest([3,1,4,4,2]) == 3\nassert solution.second_largest([10,10,9]) == 9",
    },
    {
        "id": "code_balanced", "difficulty": "easy",
        "prompt": "Write a Python function balanced(s) that returns True if the brackets ()[]{} in s are balanced, else False.",
        "test": "assert solution.balanced('([]{})') is True\nassert solution.balanced('(]') is False\nassert solution.balanced('((') is False",
    },
    {
        "id": "code_rle", "difficulty": "medium",
        "prompt": "Write a Python function rle(s) that run-length encodes a string: 'aaabcc' -> 'a3b1c2'.",
        "test": "assert solution.rle('aaabcc') == 'a3b1c2'\nassert solution.rle('') == ''\nassert solution.rle('ab') == 'a1b1'",
    },
    {
        "id": "code_top_k_words", "difficulty": "medium",
        "prompt": "Write a Python function top_k(text, k) returning the k most frequent lowercase words as a list, ties broken alphabetically.",
        "test": "assert solution.top_k('a b a c b a', 2) == ['a', 'b']\nassert solution.top_k('x y z', 2) == ['x', 'y']",
    },
    {
        "id": "code_merge_intervals", "difficulty": "medium",
        "prompt": "Write a Python function merge(intervals) that merges overlapping [start,end] intervals and returns them sorted. Example: [[1,3],[2,6],[8,10]] -> [[1,6],[8,10]].",
        "test": "assert solution.merge([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]]\nassert solution.merge([[1,4],[4,5]]) == [[1,5]]",
    },
    {
        "id": "code_roman", "difficulty": "hard",
        "prompt": "Write a Python function to_roman(n) converting an integer 1..3999 to a Roman numeral string.",
        "test": "assert solution.to_roman(1994) == 'MCMXCIV'\nassert solution.to_roman(58) == 'LVIII'\nassert solution.to_roman(9) == 'IX'",
    },
    {
        "id": "code_edit_distance", "difficulty": "hard",
        "prompt": "Write a Python function edit_distance(a, b) computing the Levenshtein distance between two strings.",
        "test": "assert solution.edit_distance('kitten','sitting') == 3\nassert solution.edit_distance('','abc') == 3\nassert solution.edit_distance('abc','abc') == 0",
    },
    {
        "id": "code_coin_change", "difficulty": "hard",
        "prompt": "Write a Python function min_coins(coins, amount) returning the minimum number of coins to make amount, or -1 if impossible.",
        "test": "assert solution.min_coins([1,2,5], 11) == 3\nassert solution.min_coins([2], 3) == -1\nassert solution.min_coins([1], 0) == 0",
    },
    {
        "id": "code_spiral", "difficulty": "hard",
        "prompt": "Write a Python function spiral(matrix) returning the elements of a 2D list in clockwise spiral order as a flat list.",
        "test": "assert solution.spiral([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]\nassert solution.spiral([[1,2],[3,4]]) == [1,2,4,3]",
    },
]


def code_tasks() -> List[Dict]:
    return [
        {**t, "kind": "code",
         "prompt": t["prompt"] + " Reply with only the Python code, no explanation."}
        for t in CODE_TASKS
    ]


QA_TASKS: List[Dict] = [
    {"id": "qa_capital", "difficulty": "easy",
     "prompt": "What is the capital of Australia?",
     "reference": "Canberra (not Sydney or Melbourne)."},
    {"id": "qa_boiling", "difficulty": "easy",
     "prompt": "At what temperature does water boil at sea level, in Celsius?",
     "reference": "100 degrees Celsius."},
    {"id": "qa_http", "difficulty": "easy",
     "prompt": "In one sentence, what does an HTTP 404 status code mean?",
     "reference": "The requested resource was not found on the server."},
    {"id": "qa_tcp_udp", "difficulty": "medium",
     "prompt": "Explain the main difference between TCP and UDP and give one appropriate use case for each.",
     "reference": "TCP is connection-oriented and reliable (ordered delivery, retransmission), good for web pages or file transfer; UDP is connectionless, lower latency, no delivery guarantee, good for live video, gaming, or DNS."},
    {"id": "qa_index", "difficulty": "medium",
     "prompt": "Why can adding a database index make reads faster but writes slower?",
     "reference": "An index is an extra data structure enabling faster lookups, but every insert/update must also maintain the index, adding write overhead and storage."},
    {"id": "qa_cache_stampede", "difficulty": "medium",
     "prompt": "What is a cache stampede and name one way to prevent it.",
     "reference": "Many clients simultaneously miss an expired cache key and hit the backend at once; preventions include lock/single-flight, staggered or probabilistic early expiration, or serving stale while revalidating."},
    {"id": "qa_cap", "difficulty": "hard",
     "prompt": "Explain the CAP theorem and describe what trade-off a system like a bank transfer ledger should make and why.",
     "reference": "Under network partition a system must choose consistency or availability. A ledger should favor consistency (reject writes during partition) because double-spending or inconsistent balances are worse than downtime."},
    {"id": "qa_idempotency", "difficulty": "hard",
     "prompt": "Why do payment APIs use idempotency keys, and what could go wrong without them? Explain the mechanism.",
     "reference": "Network retries can duplicate a request; an idempotency key lets the server detect a repeat and return the original result instead of charging twice. Server stores key with the first response for a window."},
    {"id": "qa_vector_clock", "difficulty": "hard",
     "prompt": "In distributed systems, what problem do vector clocks solve that simple timestamps cannot? Give a concrete example.",
     "reference": "They capture causality/partial order between events, detecting concurrent updates that wall-clock timestamps (skewed, not causal) cannot; e.g., two replicas updating the same key independently can be detected as a conflict."},
    {"id": "qa_llm_temp", "difficulty": "medium",
     "prompt": "What does the temperature parameter do in LLM sampling, and when would you set it to 0?",
     "reference": "It scales randomness of token sampling; higher is more diverse, lower more deterministic. Temperature 0 (greedy) suits reproducible or factual/structured outputs."},
]


def qa_tasks() -> List[Dict]:
    return [{**t, "kind": "qa"} for t in QA_TASKS]


def all_tasks() -> List[Dict]:
    return math_tasks() + code_tasks() + qa_tasks()
