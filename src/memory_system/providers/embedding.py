from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence


MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5MB limit (§4 Pitfall 10)


class EmbeddingProvider:
    """OpenAI-compatible HTTP embedding provider with trust_env=False, 2^n retry backoff,
    5MB cap, and strict batch count validation.
    """

    def __init__(
        self,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key: str = "sk-5201f73df9c54f6894cb26d64d6eddba",
        model: str = "text-embedding-v4",
        dimension: int = 1024,
        timeout_connect: float = 10.0,
        timeout_read: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimension = int(dimension)
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
    def from_config(cls, cfg: Dict[str, Any]) -> EmbeddingProvider:
        return cls(
            base_url=str(cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
            api_key=str(cfg.get("api_key", "")),
            model=str(cfg.get("model", "text-embedding-v4")),
            dimension=int(cfg.get("dimension", 1024)),
            timeout_connect=float(cfg.get("timeout_connect", 10.0)),
            timeout_read=float(cfg.get("timeout_read", 120.0)),
            max_retries=int(cfg.get("max_retries", 3)),
        )

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        clean_texts = [t if (t and t.strip()) else "_" for t in texts]
        max_batch_size = 10
        if len(clean_texts) > max_batch_size:
            all_vecs: List[List[float]] = []
            for start_idx in range(0, len(clean_texts), max_batch_size):
                all_vecs.extend(self.embed(clean_texts[start_idx : start_idx + max_batch_size]))
            return all_vecs
        url = f"{self.base_url}/embeddings"
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": list(clean_texts),
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
                    raise ValueError(f"Embedding response exceeded 5MB cap ({len(raw_bytes)} bytes)")
                data = json.loads(raw_bytes.decode("utf-8"))
                items = data.get("data") or []
                items_sorted = sorted(items, key=lambda x: int(x.get("index", 0)))
                vectors = [[float(v) for v in item["embedding"]] for item in items_sorted]
                if len(vectors) != len(clean_texts):
                    raise ValueError(
                        f"Embedding count mismatch: expected {len(clean_texts)}, got {len(vectors)}"
                    )
                return vectors
            except ValueError:
                raise
            except Exception as exc:
                last_err = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Embedding request failed after {self.max_retries} attempts: {last_err}") from last_err

    def embed_one(self, text: str) -> List[float]:
        res = self.embed([text])
        return res[0] if res else []

    def _post_raw(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> bytes:
        if self._client is not None:
            resp = self._client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.content
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with self._opener.open(req, timeout=self.timeout_read) as resp:
            return resp.read(MAX_RESPONSE_BYTES + 1)


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic local character/token trigram hash embedding provider for offline unit tests."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension
        self.base_url = "mock://local"
        self.api_key = "mock"
        self.model = "mock-embedding"
        self.timeout_connect = 1.0
        self.timeout_read = 1.0
        self.max_retries = 1

    def _compute_mock_vec(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        s = (text or "_").lower()
        # Character unigrams + bigrams so semantically overlapping strings have high cosine similarity
        grams = [s[i] for i in range(len(s))] + [s[i : i + 2] for i in range(max(0, len(s) - 1))]
        if not grams:
            grams = ["_"]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_one(self, text: str) -> List[float]:
        return self._compute_mock_vec(text)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        use_custom_embed_one = (
            type(self).embed_one is not MockEmbeddingProvider.embed_one
            and not hasattr(self, "call_count")
        )
        results: List[List[float]] = []
        for text in texts:
            if use_custom_embed_one:
                results.append(self.embed_one(text))
            else:
                results.append(self._compute_mock_vec(text))
        if len(results) != len(texts):
            raise ValueError(f"Embedding count mismatch: expected {len(texts)}, got {len(results)}")
        return results
