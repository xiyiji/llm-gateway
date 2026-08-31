"""Usage accounting: every request through the gateway leaves a priced record."""

import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from gateway.config import VAR_DIR

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    router TEXT NOT NULL,
    decision TEXT NOT NULL,
    model TEXT NOT NULL,
    cached INTEGER NOT NULL,
    fallback_used INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms INTEGER NOT NULL,
    api_key TEXT NOT NULL DEFAULT ''
);
"""

_MIGRATIONS = ("ALTER TABLE usage ADD COLUMN api_key TEXT NOT NULL DEFAULT ''",)


class UsageStore:
    def __init__(self, db_path: Optional[Path] = None):
        VAR_DIR.mkdir(exist_ok=True)
        self.db_path = db_path or (VAR_DIR / "usage.db")
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            for migration in _MIGRATIONS:
                try:
                    conn.execute(migration)
                except sqlite3.OperationalError:
                    pass

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, *, router: str, decision: str, model: str, cached: bool,
               fallback_used: bool, input_tokens: int, output_tokens: int,
               cost_usd: float, latency_ms: int, api_key: str = "") -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO usage (ts, router, decision, model, cached, fallback_used,"
                " input_tokens, output_tokens, cost_usd, latency_ms, api_key)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), router, decision, model, int(cached), int(fallback_used),
                 input_tokens, output_tokens, cost_usd, latency_ms, api_key),
            )

    def spend_since(self, api_key: str, since_ts: float) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) s FROM usage"
                " WHERE api_key = ? AND ts >= ?", (api_key, since_ts),
            ).fetchone()
        return float(row["s"])

    def summary(self) -> Dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) c FROM usage").fetchone()["c"]
            if total == 0:
                return {"requests": 0}
            row = conn.execute(
                "SELECT SUM(cost_usd) cost, AVG(latency_ms) avg_latency,"
                " SUM(cached) cached, SUM(fallback_used) fallbacks FROM usage"
            ).fetchone()
            by_model = {
                r["model"]: {"requests": r["c"], "cost_usd": round(r["cost"], 6)}
                for r in conn.execute(
                    "SELECT model, COUNT(*) c, SUM(cost_usd) cost FROM usage GROUP BY model"
                )
            }
            by_decision = {
                r["decision"]: r["c"]
                for r in conn.execute("SELECT decision, COUNT(*) c FROM usage GROUP BY decision")
            }
            by_router = {
                r["router"]: r["c"]
                for r in conn.execute("SELECT router, COUNT(*) c FROM usage GROUP BY router")
            }
            by_key = {
                (r["api_key"] or "anonymous"): {"requests": r["c"], "cost_usd": round(r["cost"], 6)}
                for r in conn.execute(
                    "SELECT api_key, COUNT(*) c, SUM(cost_usd) cost FROM usage GROUP BY api_key"
                )
            }
        return {
            "requests": total,
            "total_cost_usd": round(row["cost"] or 0.0, 6),
            "avg_latency_ms": round(row["avg_latency"] or 0.0, 1),
            "cache_hits": row["cached"],
            "fallbacks": row["fallbacks"],
            "by_model": by_model,
            "by_decision": by_decision,
            "by_router": by_router,
            "by_key": by_key,
        }

    def clear(self) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM usage")
