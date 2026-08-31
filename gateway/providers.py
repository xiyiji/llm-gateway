"""Provider layer: DeepSeek behind a response cache, with real cost accounting.

Token usage comes from the API response and is priced per model, so every
cost figure in the gateway is measured, not estimated.
"""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from gateway.config import api_key, get_config


class ProviderError(Exception):
    pass


class ChatResult:
    def __init__(self, text: str, model: str, cached: bool, latency_ms: int,
                 input_tokens: int, output_tokens: int, cost_usd: float):
        self.text = text
        self.model = model
        self.cached = cached
        self.latency_ms = latency_ms
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd


def price_for(tier: str, input_tokens: int, output_tokens: int) -> float:
    spec = get_config().models.get(tier)
    if spec is None:
        return 0.0
    p = spec.pricing_per_1m
    return round(
        input_tokens / 1_000_000 * p.get("input", 0.0)
        + output_tokens / 1_000_000 * p.get("output", 0.0),
        8,
    )


class ResponseCache:
    """LRU + TTL cache; hit/miss counters are the source of the cached flag."""

    def __init__(self, max_entries: int, ttl_seconds: int):
        self._store: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()
        self._lock = threading.Lock()
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key_for(model: str, messages: List[Dict], temperature: float) -> str:
        payload = json.dumps(
            {"m": model, "msgs": messages, "t": temperature}, sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, text = entry
            if time.time() - stored_at > self.ttl_seconds:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return text

    def put(self, key: str, text: str) -> None:
        with self._lock:
            self._store[key] = (time.time(), text)
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def stats(self) -> Dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._store),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }


class DeepSeekProvider:
    def __init__(self, cache: Optional[ResponseCache] = None):
        cfg = get_config()
        self.cfg = cfg.llm
        self.cache = cache or ResponseCache(cfg.cache.max_entries, cfg.cache.ttl_seconds)
        self.cache_enabled = cfg.cache.enabled
        self._client = None

    def available(self) -> Tuple[bool, Optional[str]]:
        if not api_key():
            return False, f"{self.cfg.api_key_env} not configured"
        return True, None

    def _sdk(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=api_key(), base_url=self.cfg.base_url, timeout=self.cfg.timeout_s
            )
        return self._client

    def chat(self, tier: str, messages: List[Dict],
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> ChatResult:
        ok, reason = self.available()
        if not ok:
            raise ProviderError(reason)
        spec = get_config().models[tier]
        temperature = self.cfg.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.cfg.max_tokens

        key = ResponseCache.key_for(spec.name, messages, temperature)
        if self.cache_enabled:
            hit = self.cache.get(key)
            if hit is not None:
                return ChatResult(hit, spec.name, True, 0, 0, 0, 0.0)

        start = time.time()
        try:
            completion = self._sdk().chat.completions.create(
                model=spec.name, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc
        text = completion.choices[0].message.content or ""
        latency_ms = int((time.time() - start) * 1000)
        usage = getattr(completion, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        if self.cache_enabled:
            self.cache.put(key, text)
        return ChatResult(
            text, spec.name, False, latency_ms, input_tokens, output_tokens,
            price_for(tier, input_tokens, output_tokens),
        )

    def stream_chat(self, tier: str, messages: List[Dict],
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None):
        """Start a streamed completion. Connection errors raise ProviderError
        here (so the caller can fall back before any bytes are sent); the
        returned iterator yields {"delta": str} events and one final
        {"usage": {...}} event. The cache key is returned so the caller can
        store the accumulated text for future non-streamed hits."""
        ok, reason = self.available()
        if not ok:
            raise ProviderError(reason)
        spec = get_config().models[tier]
        temperature = self.cfg.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.cfg.max_tokens
        key = ResponseCache.key_for(spec.name, messages, temperature)
        try:
            stream = self._sdk().chat.completions.create(
                model=spec.name, messages=messages, temperature=temperature,
                max_tokens=max_tokens, stream=True,
                stream_options={"include_usage": True},
            )
        except Exception as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc

        def events():
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield {"delta": chunk.choices[0].delta.content}
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    yield {"usage": {
                        "input": getattr(usage, "prompt_tokens", 0) or 0,
                        "output": getattr(usage, "completion_tokens", 0) or 0,
                    }}

        return events(), spec.name, key
