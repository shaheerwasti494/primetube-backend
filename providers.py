# providers.py
import os
import asyncio
import httpx
from typing import Any, Dict, List, Optional, Tuple

INVIDIOUS_BASE = os.getenv("INVIDIOUS_BASE", "https://yewtu.be")  # change if you like

class InvidiousProvider:
    """
    Thin async client around an Invidious instance.
    Docs: https://github.com/iv-org/documentation/blob/master/API.md
    """

    def __init__(self, base: str = INVIDIOUS_BASE):
        self.base = base.rstrip("/")

    async def _get(self, path: str, params: Dict[str, Any] | None = None) -> Any:
        async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": "PrimeTube/1.0"}) as s:
            r = await s.get(f"{self.base}{path}", params=params)
            r.raise_for_status()
            return r.json()

    # suggestions (Invidious)
    async def suggest(self, q: str) -> List[str]:
        # returns ["term", ["s1","s2",...]] on some instances; others: {"suggestions":[...]}
        try:
            data = await self._get("/api/v1/search/suggestions", {"q": q})
            if isinstance(data, dict) and "suggestions" in data:
                return list(map(str, data["suggestions"]))
            if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
                return [str(x) for x in data[1]]
        except Exception:
            pass
        return []

    # search (videos, channels)
    async def search(self, q: str, page: int = 1, search_type: Optional[str] = None) -> Dict[str, Any]:
        params = {"q": q, "page": page}
        if search_type: params["type"] = search_type  # "video" | "channel"
        return await self._get("/api/v1/search", params)

    async def trending(self, region: str = "US") -> List[Dict[str, Any]]:
        # No paging on most instances; we’ll fake nextPage in API layer.
        return await self._get("/api/v1/trending", {"region": region})

    async def video(self, video_id: str) -> Dict[str, Any]:
        return await self._get(f"/api/v1/videos/{video_id}")

    async def channel(self, channel_id: str) -> Dict[str, Any]:
        return await self._get(f"/api/v1/channels/{channel_id}")
