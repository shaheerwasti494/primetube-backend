# cache.py
import time
from typing import Any, Optional

class TTLCache:
    def __init__(self, ttl_seconds: int = 300, max_size: int = 2000):
        self.ttl = ttl_seconds
        self.max = max_size
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if not item:
            return None
        exp, val = item
        if exp < time.time():
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        if len(self._store) >= self.max:
            # naive eviction: pop first
            self._store.pop(next(iter(self._store)))
        self._store[key] = (time.time() + (ttl or self.ttl), value)
