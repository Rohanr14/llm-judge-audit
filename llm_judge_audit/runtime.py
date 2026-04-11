"""Process-wide runtime settings and a small disk-backed judge cache.

Two jobs:

1. ``max_concurrency``: single knob that every parallel API path in the
   codebase consults. The default is intentionally small (4) -- the old
   ``min(32, (cpu_count() or 1) * 5)`` default was nuclear on any rate
   limited provider, and a single bias test could exhaust Anthropic's
   per-minute budget mid-audit. The CLI exposes this as ``--max-concurrency``.

2. Judge cache: a crude, append-only JSONL cache keyed by
   ``(model, prompt, response_a, response_b, temperature, history, kind)``.
   When enabled, judges consult the cache before calling the API. That lets
   a crashed audit resume without re-spending money, and makes re-running a
   subset of biases against the same model effectively free.

Both facilities are deliberately optional. If ``JUDGE_CACHE_PATH`` is not
set and the CLI doesn't enable the cache, nothing changes from the old
behaviour other than the lower concurrency default.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_judge_audit.logger import logger


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


@dataclass
class RuntimeSettings:
    # Process-wide cap on concurrent judge API calls.
    max_concurrency: int = field(default_factory=lambda: _env_int("JUDGE_MAX_CONCURRENCY", 4))
    # Disk-backed cache for judge calls. None means disabled.
    cache_path: Path | None = None
    # Guard for the cache file so multiple threads can't interleave writes.
    _cache_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # In-memory mirror, loaded lazily on first use.
    _cache: dict[str, Any] | None = field(default=None, repr=False)

    def configure(
        self,
        *,
        max_concurrency: int | None = None,
        cache_path: str | Path | None = None,
    ) -> None:
        if max_concurrency is not None:
            self.max_concurrency = max(1, int(max_concurrency))
        if cache_path is not None:
            self.cache_path = Path(cache_path)
            self._cache = None  # force reload with new path

    # --------------------------- cache helpers ---------------------------

    def _load_cache(self) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache
        cache: dict[str, Any] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                with self.cache_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        key = rec.get("key")
                        value = rec.get("value")
                        if key is not None:
                            cache[key] = value
                logger.info(
                    "Judge cache: loaded %d entries from %s", len(cache), self.cache_path
                )
            except OSError as exc:
                logger.warning("Judge cache: could not read %s: %s", self.cache_path, exc)
        self._cache = cache
        return cache

    @staticmethod
    def make_key(*parts: Any) -> str:
        payload = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def cache_get(self, key: str) -> Any | None:
        if self.cache_path is None:
            return None
        return self._load_cache().get(key)

    def cache_put(self, key: str, value: Any) -> None:
        if self.cache_path is None:
            return
        with self._cache_lock:
            cache = self._load_cache()
            if key in cache and cache[key] == value:
                return
            cache[key] = value
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Append-only so a crashed write doesn't corrupt the whole file.
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}) + "\n")


SETTINGS = RuntimeSettings()
