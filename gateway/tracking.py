"""Where online-eval numbers and router versions are recorded.

MLflow is the system of record when it is installed: one long-lived run per
gateway process holds the live quality metrics (per-category small-model win
rate, guard state, judge spend) as step series, guard events are logged as
JSON artifacts, and every router retrained from live labels is its own run
with params, offline and live-shadow metrics, the model artifact and a
registry version. The JSONL fallback keeps the gateway dependency-light when
MLflow isn't installed (CI, small deployments).
"""

import json
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from gateway.config import VAR_DIR


class JsonlTracker:
    name = "jsonl"

    def __init__(self, path: Optional[Path] = None):
        VAR_DIR.mkdir(exist_ok=True)
        self.path = path or (VAR_DIR / "online_eval_events.jsonl")
        self._lock = threading.Lock()

    def _write(self, record: Dict) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        self._write({"type": "metrics", "ts": time.time(), "step": step, **metrics})

    def log_event(self, event: Dict) -> None:
        self._write({"type": "event", **event})

    def log_router_run(self, name: str, params: Dict, metrics: Dict,
                       model_path: Optional[Path] = None, register: bool = False) -> str:
        self._write({"type": "router_run", "ts": time.time(), "name": name,
                     "params": params, "metrics": metrics,
                     "model_path": str(model_path) if model_path else None})
        return name


class MlflowTracker:
    name = "mlflow"

    def __init__(self, tracking_uri: str = "", experiment: str = "llm-gateway-online-eval"):
        import mlflow
        from mlflow.tracking import MlflowClient

        VAR_DIR.mkdir(exist_ok=True)
        self.tracking_uri = tracking_uri or f"sqlite:///{VAR_DIR / 'mlflow.db'}"
        mlflow.set_tracking_uri(self.tracking_uri)
        self._mlflow = mlflow
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        exp = self.client.get_experiment_by_name(experiment)
        if exp is None:
            # keep artifacts next to the sqlite store, not in ./mlruns
            artifacts = None
            if self.tracking_uri.startswith("sqlite:///"):
                db = Path(self.tracking_uri[len("sqlite:///"):])
                artifacts = (db.parent / "mlartifacts").resolve().as_uri()
            self.experiment_id = self.client.create_experiment(
                experiment, artifact_location=artifacts)
        else:
            self.experiment_id = exp.experiment_id
        self._lock = threading.Lock()
        self._live_run_id: Optional[str] = None
        self._n_events = 0

    def _live_run(self) -> str:
        if self._live_run_id is None:
            run = self.client.create_run(
                self.experiment_id,
                run_name=f"live-{time.strftime('%Y%m%d-%H%M%S')}",
                tags={"kind": "online_eval"},
            )
            self._live_run_id = run.info.run_id
        return self._live_run_id

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        with self._lock:
            run_id = self._live_run()
            now = int(time.time() * 1000)
            from mlflow.entities import Metric

            self.client.log_batch(run_id, metrics=[
                Metric(k, float(v), now, step) for k, v in metrics.items()])

    def log_event(self, event: Dict) -> None:
        with self._lock:
            run_id = self._live_run()
            self._n_events += 1
            self.client.log_dict(run_id, event,
                                 f"events/{self._n_events:05d}_{event.get('event', 'event')}.json")
            if event.get("event") in ("guard_on", "guard_off"):
                self.client.set_tag(run_id, f"guard.{event['category']}", event["event"])

    def log_router_run(self, name: str, params: Dict, metrics: Dict,
                       model_path: Optional[Path] = None, register: bool = False) -> str:
        mlflow = self._mlflow
        with self._lock:
            run = self.client.create_run(self.experiment_id, run_name=name,
                                         tags={"kind": "router_training"})
            run_id = run.info.run_id
            for k, v in params.items():
                self.client.log_param(run_id, k, v)
            now = int(time.time() * 1000)
            from mlflow.entities import Metric

            self.client.log_batch(run_id, metrics=[
                Metric(k, float(v), now, 0) for k, v in metrics.items()])
            if model_path is not None:
                self.client.log_artifact(run_id, str(model_path), artifact_path="router")
                if register:
                    try:
                        self.client.create_registered_model("gateway-router")
                    except Exception:
                        pass  # already exists
                    mv = self.client.create_model_version(
                        "gateway-router", f"runs:/{run_id}/router", run_id)
                    self.client.set_registered_model_alias("gateway-router", "candidate",
                                                           mv.version)
            self.client.set_terminated(run_id)
        _ = mlflow
        return run_id


def build_tracker(cfg) -> object:
    choice = getattr(cfg, "tracker", "auto")
    if choice in ("auto", "mlflow"):
        try:
            return MlflowTracker(getattr(cfg, "mlflow_tracking_uri", ""),
                                 getattr(cfg, "mlflow_experiment", "llm-gateway-online-eval"))
        except ImportError:
            if choice == "mlflow":
                raise
    return JsonlTracker()
