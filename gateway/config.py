"""Configuration loading."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAR_DIR = PROJECT_ROOT / "var"
MODELS_DIR = PROJECT_ROOT / "models"


class ModelSpec(BaseModel):
    name: str
    pricing_per_1m: Dict[str, float]
    avg_latency_ms: int = 2000


class LLMConfig(BaseModel):
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout_s: int = 120
    temperature: float = 0.0
    max_tokens: int = 1024


class RouterConfig(BaseModel):
    active: str = "v1_score"
    quality_cost_lambda: float = 2.0


class CacheConfig(BaseModel):
    enabled: bool = True
    max_entries: int = 1024
    ttl_seconds: int = 3600


class GatewayConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8020


class AppConfig(BaseModel):
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    models: Dict[str, ModelSpec] = Field(default_factory=dict)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    path = PROJECT_ROOT / "config.yaml"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return AppConfig.model_validate(yaml.safe_load(f) or {})
    return AppConfig()


def api_key() -> str:
    return os.environ.get(get_config().llm.api_key_env, "")
