from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# -------------------- Config --------------------

PIPED_INSTANCES = [
    "https://piped.video",
    "https://pipedapi.kavin.rocks",
    "https://piped.hostux.net",
]
INVIDIOUS_INSTANCES = [
    "https://yt.artemislena.eu",
    "https://invidious.nerdvpn.de",
    "https://invidious.flokinet.to",
]

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "7"))
CACHE_SECONDS_DEFAULT = int(os.getenv("CACHE_SECONDS_DEFAULT", "300"))
REGION_DEFAULT = os.getenv("REGION_DEFAULT", "US")

app = FastAPI(title="PrimeTube Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_methods=["*"],
    allow_headers=["*"],
)

client = httpx.AsyncClient(
    timeout=httpx.Timeout(HTTP_TIMEOUT),
    headers={"User-Agent": "primetube-backend/1.0"},
    http2=True,
)

# -------------------- Helpers --------------------

async def _get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    try:
        r = await client.get(url, params=params)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

async def fetch_first_ok(urls: List[str], path: str, params: Dict[str, Any]) -> Tuple[str, Any]:
    """
    Try a list of base URLs and return the first JSON that works.
    """
    for base in urls:
        data = await _get_json(f"{base}{path}", params)
        if data is not None:
            return base, data
    raise HTTPException(status_code=502, detail="Upstream temporarily unavailable")

def _thumbs_to_set(thumbs: List[Dict[str, Any]] | None) -> Dict[str, Optional[str]]:
    if not thumbs:
        return {"defaultUrl": None, "mediumUrl": None, "highUrl": None}
    # Piped/Inv thumbnails can be in various sizes; pick a few common ones
    best = sorted(thumbs, key=lambda t: t.get("height") or 0)
    return {
        "defaultUrl": best[0]["url"] if best else None,
        "mediumUrl": best[len(best)//2]["url"] if best else None,
        "highUrl": best[-1]["url"] if best else None,
    }

def _video_item_from_piped(v: Dict[str, Any]) -> Dict[str, Any]:
    # Piped trending/search video fields vary a bit across endpoints
    vid = v.get("id") or v.get("url") or v.get("shortId") or v.get("videoId")
    title = v.get("title")
    channel_name = v.get("uploaderName") or v.get("uploader") or v.get("author")
    channel_id = v.get("uploaderId") or v.get("uploaderUrl") or v.get("authorId")
    # stats
    views = v.get("views") or v.get("viewCount") or 0
    likes = v.get("likeCount") or None
    comments = v.get("commentCount") or None
    published_text = v.get("uploadedDate") or v.get("publishedText") or None
    thumbnails = v.get("thumbnail") or v.get("thumbnailUrl")
    if isinstance(thumbnails, str):
        thumbs = [{"url": thumbnails, "height": 720}]
    else:
        thumbs = v.get("thumbnails") or []
    return {
        "type": "video",
        "data": {
            "id": str(vid).replace("/watch?v=", "").replace("/shorts/", ""),
            "title": title,
            "channelId": channel_id,
            "channelName": channel_name,
            "publishedText": published_text,
            "viewCount": int(views) if isinstance(views, (int, float, str)) and str(views).isdigit() else None,
            "likeCount": likes if isinstance(likes, int) else None,
            "commentCount": comments if isinstance(comments, int) else None,
            "thumbnails": _thumbs_to_set(thumbs if isinstance(thumbs, list) else None),
        },
    }

def _channel_item_from_piped(c: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "channel",
        "data": {
            "id": c.get("uploaderId") or c.get("channelId") or c.get("id"),
            "title": c.get("uploaderName") or c.get("name") or c.get("author"),
            "avatar": (c.get("uploaderAvatar") or c.get("authorThumbnails", [{}])[-1].get("url") if c.get("authorThumbnails") else None),
        },
    }

def cacheable(payload: Any, max_age: int = CACHE_SECONDS_DEFAULT) -> JSONResponse:
    return JSONResponse(
        payload,
        headers={"Cache-Control": f"public, max-age={max_age}"}
    )

# -------------------- Routes --------------------

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/trending")
async def trending(
    region: str = Query(REGION_DEFAULT, min_length=2, max_length=5),
    limit: int = Query(20, ge=1, le=50),
    stats: bool = Query(False, description="Fetch per-video stats (slower)")
):
    """
    Get trending videos via Piped (fallback to Invidious).
    """
    # Piped
    try:
        base, data = await fetch_first_ok(PIPED_INSTANCES, "/api/v1/trending", {"region": region})
        items_raw = data if isinstance(data, list) else []
        items = [_video_item_from_piped(v) for v in items_raw[:limit]]
        return cacheable({"items": items, "nextPage": None})
    except HTTPException:
        pass

    # Invidious fallback
    base, data = await fetch_first_ok(INVIDIOUS_INSTANCES, "/api/v1/trending", {"region": region})
    items_raw = data if isinstance(data, list) else []
    items = []
    for v in items_raw[:limit]:
        # map invidious keys to our shape
        mapped = {
            "id": v.get("videoId"),
            "title": v.get("title"),
            "channelId": v.get("authorId"),
            "channelName": v.get("author"),
            "publishedText": v.get("publishedText"),
            "viewCount": v.get("viewCount"),
            "likeCount": v.get("likeCount"),
            "commentCount": v.get("commentCount"),
            "thumbnails": _thumbs_to_set(v.get("videoThumbnails")),
        }
        items.append({"type": "video", "data": mapped})
    return cacheable({"items": items, "nextPage": None})

@app.get("/shorts")
async def shorts(
    page: int | None = Query(None, ge=1),
    limit: int = Query(20, ge=1, le=50),
    region: str = Query(REGION_DEFAULT),
):
    """
    Return short videos (<= 90s). Uses Piped search with a shorts filter when available,
    otherwise filters by duration.
    """
    q = "#shorts"
    params = {"q": q, "page": page or 1}
    # Piped search
    base, data = await fetch_first_ok(PIPED_INSTANCES, "/api/v1/search", params)
    items_raw = data.get("items") or data if isinstance(data, dict) else []
    out: List[Dict[str, Any]] = []
    for it in items_raw:
        # Only keep videos which look like shorts
        dur = it.get("duration") or it.get("lengthSeconds")
        try:
            is_short = dur is not None and int(dur) <= 90
        except Exception:
            is_short = False
        if it.get("type") in ("stream", "video") and (is_short or "#shorts" in (it.get("title") or "").lower()):
            out.append(_video_item_from_piped(it))
        if len(out) >= limit:
            break

    next_page = (page or 1) + 1 if len(items_raw) else None
    return cacheable({"items": out, "nextPage": next_page})

@app.get("/search")
async def search(
    q: str = Query(..., min_length=1),
    page: int | None = Query(None, ge=1),
    limit: int = Query(20, ge=1, le=50),
):
    """
    Mixed search; videos + channels using Piped (fallback to Invidious).
    """
    # Piped
    base, data = await fetch_first_ok(PIPED_INSTANCES, "/api/v1/search", {"q": q, "page": page or 1})
    items_raw = data.get("items") or data if isinstance(data, dict) else []
    items: List[Dict[str, Any]] = []
    for it in items_raw:
        t = (it.get("type") or "").lower()
        if t in ("stream", "video"):
            items.append(_video_item_from_piped(it))
        elif t in ("channel", "channelList"):
            items.append(_channel_item_from_piped(it))
        if len(items) >= limit:
            break
    next_page = (page or 1) + 1 if len(items_raw) else None
    return cacheable({"items": items, "nextPage": next_page})

@app.get("/channels")
async def channels(
    q: str = Query(..., min_length=1),
    page: int | None = Query(None, ge=1)
):
    """
    Channel-only search.
    """
    base, data = await fetch_first_ok(PIPED_INSTANCES, "/api/v1/search", {"q": q, "page": page or 1, "filter": "channels"})
    items_raw = data.get("items") or data if isinstance(data, dict) else []
    items = [_channel_item_from_piped(it) for it in items_raw if (it.get("type") or "").lower().startswith("channel")]
    next_page = (page or 1) + 1 if len(items_raw) else None
    return cacheable({"items": items, "nextPage": next_page})

@app.get("/suggest")
async def suggest(q: str = Query(..., min_length=1)):
    """
    Suggestions via the public YouTube suggest endpoint (no key).
    """
    try:
        r = await client.get(
            "https://suggestqueries.google.com/complete/search",
            params={"client": "youtube", "q": q, "hl": "en"},
        )
        r.raise_for_status()
        data = r.json()
        suggestions = []
        # Format can be ['q', ['a','b','c'], ...]
        if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
            suggestions = [str(s) for s in data[1][:10]]
        return cacheable({"suggestions": suggestions}, max_age=600)
    except Exception:
        return cacheable({"suggestions": []}, max_age=60)

@app.get("/video/{video_id}")
async def video(video_id: str):
    """
    Basic video details (tries Piped then Invidious).
    """
    # Piped video
    try:
        base, data = await fetch_first_ok(PIPED_INSTANCES, f"/api/v1/video/{video_id}", {})
        thumbs = data.get("thumbnails") or data.get("thumbnail") or []
        item = {
            "id": video_id,
            "title": data.get("title"),
            "channelId": data.get("uploaderId") or data.get("uploaderUrl"),
            "channelName": data.get("uploader"),
            "publishedText": data.get("uploadedDate"),
            "viewCount": data.get("views"),
            "likeCount": data.get("likeCount"),
            "commentCount": data.get("commentsCount") or data.get("commentCount"),
            "thumbnails": _thumbs_to_set(thumbs if isinstance(thumbs, list) else None),
        }
        return cacheable(item)
    except HTTPException:
        pass

    # Invidious fallback
    base, data = await fetch_first_ok(INVIDIOUS_INSTANCES, f"/api/v1/videos/{video_id}", {})
    item = {
        "id": video_id,
        "title": data.get("title"),
        "channelId": data.get("authorId"),
        "channelName": data.get("author"),
        "publishedText": data.get("publishedText"),
        "viewCount": data.get("viewCount"),
        "likeCount": data.get("likeCount"),
        "commentCount": data.get("commentCount"),
        "thumbnails": _thumbs_to_set(data.get("videoThumbnails")),
    }
    return cacheable(item)

# -------------------- Shutdown --------------------
@app.on_event("shutdown")
async def _shutdown():
    await client.aclose()
