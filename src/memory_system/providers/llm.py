from __future__ import annotations

import json
import re
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional


MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5MB response limit (§4 Pitfall 10)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_from_text(text: str) -> Any:
    """Robustly parse JSON object or array from LLM response text."""
    if not text:
        raise ValueError("Empty LLM response text")
    cleaned = _THINK_TAG_RE.sub("", text).strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Try markdown code fence
    m = _CODE_FENCE_RE.search(cleaned)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Find first '{' .. last '}' or '[' .. ']'
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        sub = cleaned[first_brace : last_brace + 1]
        try:
            return json.loads(sub)
        except Exception:
            pass

    first_bracket = cleaned.find("[")
    last_bracket = cleaned.rfind("]")
    if first_bracket != -1 and last_bracket > first_bracket:
        sub = cleaned[first_bracket : last_bracket + 1]
        try:
            return json.loads(sub)
        except Exception:
            pass

    raise ValueError(f"Failed to parse JSON from LLM output: {cleaned[:200]!r}")


class LLMProvider:
    """OpenAI-compatible HTTP LLM provider with trust_env=False, 2^n retry backoff, and 5MB cap."""

    def __init__(
        self,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key: str = "sk-5201f73df9c54f6894cb26d64d6eddba",
        model: str = "qwen3.8-flash",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
        timeout_connect: float = 10.0,
        timeout_read: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.enable_thinking = bool(enable_thinking)
        self.timeout_connect = float(timeout_connect)
        self.timeout_read = float(timeout_read)
        self.max_retries = int(max_retries)
        self._client: Any = None
        try:
            try:
                import httpx  # type: ignore
            except ImportError:
                import httpx2 as httpx  # type: ignore

            self._client = httpx.Client(
                trust_env=False,
                transport=httpx.HTTPTransport(local_address="0.0.0.0"),
                timeout=httpx.Timeout(connect=self.timeout_connect, read=self.timeout_read, write=30.0, pool=10.0),
            )
        except ImportError:
            self._client = None
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> LLMProvider:
        return cls(
            base_url=str(cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
            api_key=str(cfg.get("api_key", "")),
            model=str(cfg.get("model", "qwen3.8-flash")),
            temperature=float(cfg.get("temperature", 0.0)),
            max_tokens=int(cfg.get("max_tokens", 2048)),
            enable_thinking=bool(cfg.get("enable_thinking", False)),
            timeout_connect=float(cfg.get("timeout_connect", 10.0)),
            timeout_read=float(cfg.get("timeout_read", 120.0)),
            max_retries=int(cfg.get("max_retries", 3)),
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else float(temperature),
            "max_tokens": self.max_tokens if max_tokens is None else int(max_tokens),
            "enable_thinking": self.enable_thinking,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                raw_bytes = self._post_raw(url, headers, payload)
                if len(raw_bytes) > MAX_RESPONSE_BYTES:
                    raise ValueError(f"LLM response exceeded 5MB limit ({len(raw_bytes)} bytes)")
                data = json.loads(raw_bytes.decode("utf-8"))
                choices = data.get("choices") or []
                if not choices:
                    raise ValueError(f"Empty choices in LLM response: {data}")
                msg = choices[0].get("message") or {}
                return str(msg.get("content") or "")
            except ValueError:
                raise
            except Exception as exc:
                last_err = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM chat failed after {self.max_retries} attempts: {last_err}") from last_err

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        text = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return extract_json_from_text(text)

    def _post_raw(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> bytes:
        if self._client is not None:
            resp = self._client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.content
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with self._opener.open(req, timeout=self.timeout_read) as resp:
            return resp.read(MAX_RESPONSE_BYTES + 1)


class MockLLMProvider(LLMProvider):
    """Configurable mock LLM provider for offline unit tests."""

    def __init__(self, responder: Optional[Callable[[List[Dict[str, str]]], Any]] = None) -> None:
        self.base_url = "mock://local"
        self.api_key = "mock"
        self.model = "mock-llm"
        self.temperature = 0.0
        self.max_tokens = 2048
        self.enable_thinking = False
        self.timeout_connect = 1.0
        self.timeout_read = 1.0
        self.max_retries = 1
        self.responder = responder
        self.call_count = 0
        self.last_messages: List[Dict[str, str]] = []

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        self.call_count += 1
        self.last_messages = list(messages)
        if self.responder is not None:
            res = self.responder(messages)
            if isinstance(res, str):
                return res
            return json.dumps(res, ensure_ascii=False)
        return json.dumps({"events": []}, ensure_ascii=False)
