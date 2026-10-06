"""Client for the public DeepBookV3 indexer run by Mysten Labs.

Endpoints (https://docs.sui.io/standards/deepbookv3-indexer), verified 2026-09-27:
  GET /get_pools
  GET /order_updates/{pool}?start_time&end_time&limit   all statuses when `status` is omitted:
                                                         Placed, Canceled, Modified, Expired
  GET /trades/{pool}?start_time&end_time&limit
  GET /orderbook/{pool}?level=2&depth=N                  current book only, no history

Times are unix SECONDS in the query and MILLISECONDS in the results. Windows behave as
[start, end). Responses are capped at LIMIT rows and come back newest first.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from flowcast.config import settings

LIMIT = 10_000


class DeepBookIndexer:
    def __init__(self, base: str | None = None, timeout: float | None = None):
        self.client = httpx.Client(
            base_url=(base or settings.indexer_base).rstrip("/"),
            timeout=timeout or settings.request_timeout,
            headers={"user-agent": "deepbook-flowcast/0.1"},
        )

    def _get(self, path: str, **params: Any) -> Any:
        for attempt in range(6):
            try:
                r = self.client.get(path, params=params)
            except httpx.TransportError:
                time.sleep(2**attempt)
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"indexer request failed after retries: {path} {params}")

    def pools(self) -> list[dict]:
        return self._get("/get_pools")

    def order_updates(self, pool: str, start: int, end: int, limit: int = LIMIT) -> list[dict]:
        return self._get(f"/order_updates/{pool}", start_time=start, end_time=end, limit=limit)

    def trades(self, pool: str, start: int, end: int, limit: int = LIMIT) -> list[dict]:
        return self._get(f"/trades/{pool}", start_time=start, end_time=end, limit=limit)

    def orderbook(self, pool: str, depth: int = 20) -> dict:
        return self._get(f"/orderbook/{pool}", level=2, depth=depth)
