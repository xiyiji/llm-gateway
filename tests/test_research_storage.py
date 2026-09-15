from pathlib import Path

from research.manifest import existing_tasks, stable_split
from research.schema import ModelOutcome
from research.storage import OutcomeStore


def test_split_is_stable():
    assert stable_split("same-task") == stable_split("same-task")
    assert stable_split("same-task", 1) == stable_split("same-task", 1)


def test_existing_manifest_has_required_fields():
    tasks = existing_tasks()
    assert tasks
    assert {t.category for t in tasks} == {"math", "code", "qa"}
    for task in tasks:
        assert task.task_id
        assert task.dataset
        assert task.split in {"train", "validation", "test"}
        assert task.verifier


def test_outcome_store_is_append_only_and_resumable(tmp_path: Path):
    path = tmp_path / "outcomes.jsonl"
    outcome = ModelOutcome(
        benchmark_version="pilot-v1",
        run_id="run-1",
        task_id="task-1",
        model="small",
        replicate=0,
        response="42",
        score=1.0,
        input_tokens=5,
        output_tokens=1,
        latency_ms=12.0,
        cost_usd=0.001,
        error=None,
        timestamp="2026-09-15T00:00:00Z",
    )
    store = OutcomeStore(path)
    assert store.append(outcome) is True
    assert store.append(outcome) is False

    reopened = OutcomeStore(path)
    assert reopened.contains(outcome.observation_key)
    rows = list(reopened.iter_rows())
    assert len(rows) == 1
    assert rows[0]["task_id"] == "task-1"
