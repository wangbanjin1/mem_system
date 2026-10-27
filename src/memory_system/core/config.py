from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG: Dict[str, Any] = {
    "data_root": "data_root",
    "embedding": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-5201f73df9c54f6894cb26d64d6eddba",
        "model": "text-embedding-v4",
        "dimension": 1024,
        "timeout_connect": 10.0,
        "timeout_read": 120.0,
        "max_retries": 3,
    },
    "llm": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-5201f73df9c54f6894cb26d64d6eddba",
        "model": "qwen3.8-flash",
        "temperature": 0.0,
        "max_tokens": 2048,
        "enable_thinking": False,
        "timeout_connect": 10.0,
        "timeout_read": 120.0,
        "max_retries": 3,
    },
    "llm_router": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-5201f73df9c54f6894cb26d64d6eddba",
        "model": "qwen3.8-flash",
        "temperature": 0.0,
        "max_tokens": 256,
        "enable_thinking": False,
        "timeout_connect": 10.0,
        "timeout_read": 60.0,
        "max_retries": 2,
    },
    "retrieval": {
        "alpha": 0.5,
        "beta": 0.2,
        "gamma": 0.1,
        "delta": 0.2,
        "temporary_weight": 0.5,
        "top_k": 10,
        "heat_bump_on_hit": 0.05,
    },
    "maintenance": {
        "heat_half_life_hours": 168.0,
        "candidate_timeout_days": 90,
        "eviction_max_active": 500,
        "eviction_score_threshold": 0.15,
        "timeline_compress_days": 60,
        "event_ttl_days": 90,
    },
    "pruning": {
        "enabled": False,
    },
    "queue": {
        "async_mode": False,
        "max_retries": 3,
        "poll_interval_sec": 0.5,
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


@dataclass
class Config:
    raw: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])

    @property
    def data_root(self) -> Path:
        p = Path(self.raw.get("data_root", "data_root"))
        if not p.is_absolute():
            return self.repo_root / p
        return p

    @property
    def embedding(self) -> Dict[str, Any]:
        return self.raw["embedding"]

    @property
    def llm(self) -> Dict[str, Any]:
        return self.raw["llm"]

    @property
    def llm_router(self) -> Dict[str, Any]:
        return self.raw["llm_router"]

    @property
    def retrieval(self) -> Dict[str, Any]:
        return self.raw["retrieval"]

    @property
    def maintenance(self) -> Dict[str, Any]:
        return self.raw["maintenance"]

    @property
    def pruning(self) -> Dict[str, Any]:
        return self.raw["pruning"]

    @property
    def queue(self) -> Dict[str, Any]:
        return self.raw["queue"]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def load_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    repo_root = Path(__file__).resolve().parents[3]
    if config_path is None:
        default_file = repo_root / "config" / "config.json"
        if default_file.exists():
            config_path = default_file

    merged = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is not None and Path(config_path).exists():
        loaded = json.loads(Path(config_path).read_text(encoding="utf-8"))
        merged = deep_merge(merged, loaded)

    # Environment variable overrides
    env_data_root = os.environ.get("MEMORY_DATA_ROOT")
    if env_data_root:
        merged["data_root"] = env_data_root
    env_api_url = os.environ.get("BAILIAN_API_URL")
    if env_api_url:
        merged["embedding"]["base_url"] = env_api_url
        merged["llm"]["base_url"] = env_api_url
        merged["llm_router"]["base_url"] = env_api_url
    env_api_key = os.environ.get("BAILIAN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if env_api_key:
        merged["embedding"]["api_key"] = env_api_key
        merged["llm"]["api_key"] = env_api_key
        merged["llm_router"]["api_key"] = env_api_key

    if overrides:
        merged = deep_merge(merged, overrides)

    return Config(raw=merged, repo_root=repo_root)
