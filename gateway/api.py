"""OpenAI-compatible gateway API.

Point any OpenAI SDK at this base URL and request model "auto": the gateway
routes each request to the small or large model, falls back when a call
fails, caches, and accounts for every cent. The routing verdict rides along
in a `gateway` extension field on the response.
"""

import time
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from gateway.config import get_config
from gateway.metrics import UsageStore
from gateway.providers import ChatResult, DeepSeekProvider, ProviderError
from gateway.routers import RouteDecision, build_registry

app = FastAPI(
    title="LLM Gateway",
    description="Cost-aware routing gateway: one API, the right model per request.",
    version="1.0.0",
)

_provider = DeepSeekProvider()
_usage = UsageStore()
_registry = build_registry()
_state = {"active_router": get_config().router.active}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)


def _active_router():
    name = _state["active_router"]
    router = _registry.get(name)
    if router is None:  # configured router not trained/loaded yet
        router = _registry["v1_score"]
    return router


def _tier_for_explicit(model: str) -> Optional[str]:
    for tier, spec in get_config().models.items():
        if spec.name == model or tier == model:
            return tier
    return None


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return """<!doctype html><html><head><meta charset="utf-8">
<title>LLM Gateway</title>
<style>body{font-family:system-ui,sans-serif;max-width:640px;margin:60px auto;padding:0 20px;line-height:1.7;color:#222}
a{color:#345c8a}li{margin:6px 0}</style></head><body>
<h1>LLM Gateway</h1>
<p>One OpenAI-compatible endpoint. Send model "auto" and each request is
routed to the cheapest model that can handle it, with automatic fallback,
caching, and per-request cost accounting.</p>
<ul>
<li><a href="/docs">API docs</a> (POST /v1/chat/completions)</li>
<li><a href="/admin/metrics">Usage metrics</a> &middot; <a href="/admin/routers">Routers</a> &middot; <a href="/healthz">Health</a></li>
<li>Source: <a href="https://github.com/xiyiji/llm-gateway">github.com/xiyiji/llm-gateway</a></li>
</ul>
</body></html>"""


@app.get("/healthz")
def healthz() -> Dict:
    ok, reason = _provider.available()
    return {
        "status": "ok",
        "provider_available": ok,
        **({} if ok else {"reason": reason}),
        "active_router": _state["active_router"],
        "routers_loaded": sorted(_registry.keys()),
    }


@app.get("/admin/metrics")
def admin_metrics() -> Dict:
    return {"usage": _usage.summary(), "cache": _provider.cache.stats()}


@app.get("/admin/routers")
def admin_routers() -> Dict:
    return {"active": _state["active_router"], "available": sorted(_registry.keys())}


@app.post("/admin/routers/{name}")
def activate_router(name: str) -> Dict:
    if name not in _registry:
        raise HTTPException(status_code=404, detail=f"router '{name}' not available "
                            f"(loaded: {sorted(_registry.keys())})")
    _state["active_router"] = name
    return {"active": name}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest) -> Dict:
    user_texts = [m.content for m in request.messages if m.role == "user"]
    if not user_texts:
        raise HTTPException(status_code=422, detail="at least one user message required")
    prompt = user_texts[-1]
    messages = [m.model_dump() for m in request.messages]

    if request.model in ("auto", "", None):
        router = _active_router()
        decision: RouteDecision = router.route(prompt)
    else:
        tier = _tier_for_explicit(request.model)
        if tier is None:
            raise HTTPException(status_code=404, detail=f"unknown model '{request.model}'")
        decision = RouteDecision(tier, "explicitly requested by client", 1.0, "explicit")

    tiers = [decision.action, "large" if decision.action == "small" else "small"]
    errors: Dict[str, str] = {}
    result: Optional[ChatResult] = None
    used_tier = None
    for tier in tiers:
        try:
            result = _provider.chat(tier, messages, request.temperature, request.max_tokens)
            used_tier = tier
            break
        except ProviderError as exc:
            errors[tier] = str(exc)

    if result is None:
        raise HTTPException(status_code=502, detail={"message": "all models failed", "errors": errors})

    fallback_used = used_tier != decision.action
    _usage.record(
        router=decision.router, decision=decision.action, model=result.model,
        cached=result.cached, fallback_used=fallback_used,
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        cost_usd=result.cost_usd, latency_ms=result.latency_ms,
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result.input_tokens,
            "completion_tokens": result.output_tokens,
            "total_tokens": result.input_tokens + result.output_tokens,
        },
        "gateway": {
            **decision.as_dict(),
            "served_by": used_tier,
            "fallback_used": fallback_used,
            **({"fallback_errors": errors} if errors else {}),
            "cached": result.cached,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
        },
    }
