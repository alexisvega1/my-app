#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Iterable
from collections import OrderedDict
import asyncio

try:
    import tensorstore as ts
except Exception:  # pragma: no cover
    ts = None

@dataclass
class NVMeCache:
    root: str
    capacity: int = 128
    _lru: OrderedDict = field(default_factory=OrderedDict)

    def _touch(self, key: str, value: Any):
        if key in self._lru:
            self._lru.move_to_end(key)
        self._lru[key] = value
        while len(self._lru) > self.capacity:
            self._lru.popitem(last=False)

    def get(self, key: str, loader: Callable[[], Any]) -> Any:
        if key in self._lru:
            val = self._lru[key]
            self._touch(key, val)
            return val
        val = loader()
        self._touch(key, val)
        return val

    async def prefetch(self, keys: Iterable[str], loader: Callable[[str], Any]):
        tasks = []
        for k in keys:
            if k in self._lru:
                continue
            tasks.append(asyncio.create_task(self._prefetch_one(k, loader)))
        if tasks:
            await asyncio.gather(*tasks)

    async def _prefetch_one(self, key: str, loader: Callable[[str], Any]):
        val = loader(key)
        self._touch(key, val)
