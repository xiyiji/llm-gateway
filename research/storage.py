"""Append-only JSONL storage for model outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Set

from research.schema import ModelOutcome


class OutcomeStore:
    """Small append-only store with resumability by observation key.

    Raw model observations are never updated in place. If an experiment needs a
    new verifier or schema, write a new derived artifact rather than mutating the
    raw collection.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._keys: Set[str] = set()
        if self.path.exists():
            for row in self.iter_rows():
                key = row.get("observation_key")
                if key:
                    self._keys.add(key)

    def contains(self, key: str) -> bool:
        return key in self._keys

    def append(self, outcome: ModelOutcome) -> bool:
        key = outcome.observation_key
        if key in self._keys:
            return False
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(outcome.to_dict(), ensure_ascii=False) + "\n")
        self._keys.add(key)
        return True

    def iter_rows(self) -> Iterator[Dict]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def completed_keys(self) -> Set[str]:
        return set(self._keys)
