"""OpenAI-compatible gateway API.

Point any OpenAI SDK at this base URL and request model "auto": the gateway
routes each request to the small or large model, falls back when a call
fails, caches, streams, enforces per-key daily budgets, and accounts for
every cent. The routing verdict rides along in a `gateway` extension field.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from gateway.config import get_config
from gateway.metrics import UsageStore
from gateway.providers import ChatResult, DeepSeekProvider, ProviderError, ResponseCache
from gateway.routers import RouteDecision, build_registry

app = FastAPI(
    title="LLM Gateway",
    description="Cost-aware routing gateway: one API, the right model per request.",
    version="1.1.0",
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
    stream: bool = False


def _active_router():
    router = _registry.get(_state["active_router"])
    return router if router is not None else _registry["v1_score"]


def _tier_for_explicit(model: str) -> Optional[str]:
    for tier, spec in get_config().models.items():
        if spec.name == model or tier == model:
            return tier
    return None


def _authenticate(authorization: Optional[str]) -> str:
    """Returns the client key id, or '' when auth is disabled. Enforces budgets."""
    auth = get_config().auth
    if not auth.enabled:
        return ""
    token = (authorization or "").removeprefix("Bearer ").strip()
    spec = auth.keys.get(token)
    if spec is None:
        raise HTTPException(status_code=401, detail="unknown or missing API key")
    midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    spent = _usage.spend_since(token, midnight.timestamp())
    if spent >= spec.daily_budget_usd:
        raise HTTPException(
            status_code=429,
            detail=f"daily budget exhausted for '{spec.name}': "
                   f"${spent:.4f} of ${spec.daily_budget_usd:.2f} spent",
        )
    return token


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return """<!doctype html><html><head><meta charset="utf-8">
<title>LLM Gateway</title>
<style>body{font-family:system-ui,sans-serif;max-width:640px;margin:60px auto;padding:0 20px;line-height:1.7;color:#222}
a{color:#345c8a}li{margin:6px 0}</style></head><body>
<h1>LLM Gateway</h1>
<p>One OpenAI-compatible endpoint. Send model "auto" and each request is
routed to the cheapest model that can handle it, with automatic fallback,
streaming, caching, per-key budgets, and per-request cost accounting.</p>
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
        "auth_enabled": get_config().auth.enabled,
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


def _decide(request: ChatRequest, prompt: str) -> RouteDecision:
    if request.model in ("auto", "", None):
        return _active_router().route(prompt)
    tier = _tier_for_explicit(request.model)
    if tier is None:
        raise HTTPException(status_code=404, detail=f"unknown model '{request.model}'")
    return RouteDecision(tier, "explicitly requested by client", 1.0, "explicit")


def _completion_body(result: ChatResult, decision: RouteDecision, used_tier: str,
                     fallback_used: bool, errors: Dict[str, str]) -> Dict:
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


def _stream_response(request: ChatRequest, messages: List[Dict], decision: RouteDecision,
                     client_key: str) -> StreamingResponse:
    from gateway.providers import price_for

    cfg = get_config()
    tiers = [decision.action, "large" if decision.action == "small" else "small"]
    errors: Dict[str, str] = {}

    # cache hit: replay the stored text as a short stream
    for tier in tiers[:1]:
        spec = cfg.models[tier]
        temp = cfg.llm.temperature if request.temperature is None else request.temperature
        key = ResponseCache.key_for(spec.name, messages, temp)
        cached_text = _provider.cache.get(key) if _provider.cache_enabled else None
        if cached_text is not None:
            def replay():
                cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                chunk = {"id": cid, "object": "chat.completion.chunk",
                         "created": int(time.time()), "model": spec.name,
                         "choices": [{"index": 0, "delta": {"role": "assistant",
                                     "content": cached_text}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
                final = {"id": cid, "object": "chat.completion.chunk",
                         "created": int(time.time()), "model": spec.name,
                         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                         "gateway": {**decision.as_dict(), "served_by": tier,
                                     "cached": True, "cost_usd": 0.0}}
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"
            _usage.record(router=decision.router, decision=decision.action,
                          model=spec.name, cached=True, fallback_used=False,
                          input_tokens=0, output_tokens=0, cost_usd=0.0,
                          latency_ms=0, api_key=client_key)
            return StreamingResponse(replay(), media_type="text/event-stream")

    events = model_name = cache_key = None
    used_tier = None
    for tier in tiers:
        try:
            events, model_name, cache_key = _provider.stream_chat(
                tier, messages, request.temperature, request.max_tokens)
            used_tier = tier
            break
        except ProviderError as exc:
            errors[tier] = str(exc)
    if events is None:
        raise HTTPException(status_code=502, detail={"message": "all models failed", "errors": errors})

    fallback_used = used_tier != decision.action
    start = time.time()

    def generate():
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        pieces: List[str] = []
        usage = {"input": 0, "output": 0}
        for event in events:
            if "delta" in event:
                pieces.append(event["delta"])
                chunk = {"id": cid, "object": "chat.completion.chunk",
                         "created": int(time.time()), "model": model_name,
                         "choices": [{"index": 0, "delta": {"content": event["delta"]},
                                     "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
            elif "usage" in event:
                usage = event["usage"]
        cost = price_for(used_tier, usage["input"], usage["output"])
        latency_ms = int((time.time() - start) * 1000)
        if _provider.cache_enabled and pieces:
            _provider.cache.put(cache_key, "".join(pieces))
        _usage.record(router=decision.router, decision=decision.action,
                      model=model_name, cached=False, fallback_used=fallback_used,
                      input_tokens=usage["input"], output_tokens=usage["output"],
                      cost_usd=cost, latency_ms=latency_ms, api_key=client_key)
        final = {"id": cid, "object": "chat.completion.chunk",
                 "created": int(time.time()), "model": model_name,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": usage["input"],
                           "completion_tokens": usage["output"],
                           "total_tokens": usage["input"] + usage["output"]},
                 "gateway": {**decision.as_dict(), "served_by": used_tier,
                             "fallback_used": fallback_used, "cached": False,
                             "cost_usd": cost, "latency_ms": latency_ms}}
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest, authorization: Optional[str] = Header(None)):
    client_key = _authenticate(authorization)
    user_texts = [m.content for m in request.messages if m.role == "user"]
    if not user_texts:
        raise HTTPException(status_code=422, detail="at least one user message required")
    prompt = user_texts[-1]
    messages = [m.model_dump() for m in request.messages]
    decision = _decide(request, prompt)

    if request.stream:
        return _stream_response(request, messages, decision, client_key)

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
        cost_usd=result.cost_usd, latency_ms=result.latency_ms, api_key=client_key,
    )
    return _completion_body(result, decision, used_tier, fallback_used, errors)
