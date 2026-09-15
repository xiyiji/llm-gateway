"""Schemas for reproducible large-scale routing experiments.

This module deliberately has no provider dependency. Collection, verification,
and policy replay can share the same stable records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional

Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    dataset: str
    category: str
    difficulty: str
    split: Split
    prompt: str
    verifier: str
    reference: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelOutcome:
    benchmark_version: str
    run_id: str
    task_id: str
    model: str
    replicate: int
    response: str
    score: Optional[float]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    latency_ms: Optional[float]
    cost_usd: Optional[float]
    error: Optional[str]
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

    @property
    def observation_key(self) -> str:
        return "|".join(
            [self.benchmark_version, self.task_id, self.model, str(self.replicate)]
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["observation_key"] = self.observation_key
        return result
