"""Streamlit console for the gateway: traffic, router benchmark, agent demo.

Run: ./venv/bin/python -m streamlit run console.py
"""

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("GATEWAY_API", "http://localhost:8020")
VAR = Path(__file__).resolve().parent / "var"

st.set_page_config(page_title="LLM Gateway Console", layout="wide")


def fetch(path: str):
    try:
        resp = requests.get(f"{API}{path}", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


page = st.sidebar.radio("Page", ["Traffic", "Router benchmark", "Agent demo"])
health = fetch("/healthz")
if health:
    provider = "provider ready" if health["provider_available"] else "provider key missing"
    st.sidebar.success(f"Gateway up, router: {health['active_router']}, {provider}")
else:
    st.sidebar.error(f"Gateway unreachable at {API}")

if page == "Traffic":
    st.title("Traffic")
    data = fetch("/admin/metrics")
    if data is None:
        st.error(f"Gateway unreachable at {API}. Start it with python main.py.")
        st.stop()
    usage, cache = data["usage"], data["cache"]
    if usage.get("requests", 0) == 0:
        st.info("No traffic yet. Send requests to POST /v1/chat/completions with model 'auto'.")
    else:
        c = st.columns(5)
        c[0].metric("Requests", usage["requests"])
        c[1].metric("Total cost", f"${usage['total_cost_usd']:.4f}")
        c[2].metric("Avg latency", f"{usage['avg_latency_ms']:.0f} ms")
        c[3].metric("Cache hit rate", f"{cache['hit_rate']:.0%}")
        c[4].metric("Fallbacks", usage["fallbacks"])
        st.subheader("Requests by decision")
        st.bar_chart(pd.Series(usage["by_decision"]))
        st.subheader("Cost by model")
        st.bar_chart(pd.Series({m: v["cost_usd"] for m, v in usage["by_model"].items()}))

elif page == "Router benchmark":
    st.title("Router benchmark")
    bench = VAR / "benchmarks.json"
    if not bench.exists():
        st.info("No benchmark yet. Run scripts/collect_results.py then "
                "scripts/train_routers.py then scripts/benchmark_routers.py.")
    else:
        rows = json.loads(bench.read_text())
        df = pd.DataFrame(rows).set_index("router")
        st.dataframe(df, use_container_width=True)
        st.subheader("Quality retained vs cost spent (% of always-large)")
        st.scatter_chart(df.reset_index(), x="cost_vs_large_pct",
                         y="quality_retention_pct", color="router")

elif page == "Agent demo":
    st.title("Agent demo: direct-to-large vs gateway-routed")
    path = VAR / "agent_demo.json"
    if not path.exists():
        st.info("No run yet. Run scripts/agent_demo.py with the API key set.")
    else:
        report = json.loads(path.read_text())
        direct, routed, savings = report["direct_large"], report["gateway_routed"], report["savings"]
        c = st.columns(3)
        c[0].metric("Cost reduction", f"{savings['cost_reduction_pct']:.0f}%")
        c[1].metric("Latency reduction", f"{savings['latency_reduction_pct']:.0f}%")
        c[2].metric("Accuracy delta", savings["accuracy_delta"])
        st.dataframe(pd.DataFrame([
            {"mode": "direct_large", **direct},
            {"mode": "gateway_routed", **routed},
        ]).set_index("mode"), use_container_width=True)
