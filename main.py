# main.py
import os
import time
import json
import itertools
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

# ------------------ Config ------------------
DEFAULT_INVIDIOUS = "https://yewtu.be"
POOL_ENV = os.getenv("INVIDIOUS_POOL", DEFAULT_INVIDIOUS)
INVIDIOUS_POOL: List[str] = [b.strip().rstrip("/") for b in POOL_ENV.split(",") if b.strip()]
if not INVIDIOUS_POOL:
    INVIDIOUS_POOL = [DEFAULT_INVIDIOUS]

PIPED_BASE = os.getenv("PIPED_BASE", "https://piped.video")
CORS = os.getenv("CORS_ORIGINS", "*")
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5 minutes

USER_AGENT = os.getenv(
    "USER_AGENT",
    "PrimeTubeBackend/1.0 (+https://github.com/yourorg/primetube) httpx"
)

# ------------------ App ---------------------
app = FastAPI(title="PrimeTube Backend", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS] if CORS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Models ------------------
class Thumbnail(BaseModel):
    url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

class VideoItem(BaseModel):
    id: str
    title: str
    channelName: Optional[str] = None
    channelId: Optional[str] = None
    viewCount: Optional[int] = 0
    publishedText: Optional[str] = None
    thumbnails: List[Thumbnail] = []

class ChannelItem(BaseModel):
    id: str
    title: Optional[str] = None
    avatar: Optional[str] = None
    subscribersText: Optional[str] = None

class SearchResponse(BaseModel):
    items: List[dict]  # union: {"type": "video"/"channel", "data": ...}
    nextPage: Optional[int] = None

class SuggestResponse(BaseModel):
    suggestions: List[str]

class ShortItemsResponse(BaseModel):
    items: List[VideoItem]
    nextPage: Optional[int] = None

# ------------------ Small cache ------------------
# key -> (expires_at, data)
_cache: Dict[str, Tuple[float, object]] = {}

def _cache_key(path: str, params: dict) -> str:
    # canonicalize
    return f"{path}?{json.dumps(params, sort_keys=True, separators=(',',':'))}"

def cache_get(path: str, params: dict):
    if CACHE_TTL <= 0:
        return None
    key = _cache_key(path, params)
    item = _cache.get(key)
    if not item:
        return None
    exp, data = item
    if time.time() < exp:
        return data
    _cache.pop(key, None)
    return None

def cache_set(path: str, params: dict, data: object, ttl: Optional[int] = None):
    if CACHE_TTL <= 0:
        return
    key = _cache_key(path, params)
    t = ttl if ttl is not None else CACHE_TTL
    _cache[key] = (time.time() + t, data)

# ------------------ Helpers ------------------
def thumbify(arr) -> List[Thumbnail]:
    if not isinstance(arr, list):
        return []
    out: List[Thumbnail] = []
    for t in arr:
        out.append(Thumbnail(url=t.get("url"), width=t.get("width"), height=t.get("height")))
    return out

def map_invidious_video(v: dict) -> VideoItem:
    vid = v.get("videoId") or v.get("id", "")
    vc = v.get("viewCount")
    try:
        vc_int = int(vc) if vc is not None else 0
    except Exception:
        vc_int = 0
    return VideoItem(
        id=vid,
        title=v.get("title", ""),
        channelName=v.get("author"),
        channelId=v.get("authorId"),
        viewCount=vc_int,
        publishedText=v.get("publishedText"),
        thumbnails=thumbify(v.get("videoThumbnails") or v.get("thumbnails") or []),
    )

def map_invidious_channel(c: dict) -> ChannelItem:
    avatar = None
    thumbs = c.get("authorThumbnails") or c.get("thumbnails") or []
    if thumbs:
        avatar = thumbs[-1].get("url")
    return ChannelItem(
        id=c.get("authorId") or c.get("id", ""),
        title=c.get("author") or c.get("title"),
        avatar=avatar,
        subscribersText=c.get("subCountText") or c.get("subscriberCountText"),
    )

_pool_cycle = itertools.cycle(INVIDIOUS_POOL)

async def _fetch_json_once(base: str, path: str, params: dict) -> object:
    url = f"{base}{path}"
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
        r = await client.get(url, params=params)
        # Raise for non-2xx
        r.raise_for_status()
        # Return JSON (could be list or dict)
        return r.json()

async def _get_json_with_failover(path: str, params: dict, cacheable: bool = True) -> object:
    # 1) serve from cache first (if allowed)
    cached = cache_get(path, params) if cacheable else None
    if cached is not None:
        return cached

    # 2) try each base at most once (len(INVIDIOUS_POOL) attempts), with small backoff on 429/5xx
    last_error: Optional[Exception] = None
    tried = set()
    for _ in range(len(INVIDIOUS_POOL)):
        base = next(_pool_cycle)
        if base in tried:
            continue
        tried.add(base)
        try:
            data = await _fetch_json_once(base, path, params)
            if cacheable:
                cache_set(path, params, data)
            return data
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            # On 429/5xx, try next instance after a small backoff; on 4xx (not 429), bail.
            if status == 429 or (500 <= status < 600):
                await _async_backoff(status)
                last_error = e
                continue
            raise HTTPException(status_code=status, detail=f"Upstream error {status} at {base}{path}") from e
        except Exception as e:
            last_error = e
            # transient network failure → try next
            await _async_backoff()
            continue

    # 3) if we have previous cached data, serve it even if stale (best-effort)
    if cacheable:
        stale = cache_get(path, params)  # re-check in case concurrent set happened
        if stale is not None:
            return stale

    # 4) give up with a helpful error
    raise HTTPException(status_code=502, detail=f"All Invidious mirrors failed for {path}: {last_error}")

async def _async_backoff(status: Optional[int] = None):
    # tiny jitter/backoff; could be made exponential with more context
    # we keep it small to meet Cloud Run request timeouts
    delay = 0.25 if status == 429 else 0.15
    await httpx.AsyncClient().aclose()  # no-op but keeps type checker quiet
    await httpx.AsyncClient(timeout=0.01).__aexit__(None, None, None)  # noop trick
    # actually sleep:
    import asyncio
    await asyncio.sleep(delay)

# ------------------ Endpoints ------------------
@app.get("/health")
async def health():
    return {
        "ok": True,
        "active_pool": INVIDIOUS_POOL,
        "piped": PIPED_BASE,
        "cache_ttl": CACHE_TTL,
    }

@app.get("/v1/suggest", response_model=SuggestResponse)
async def suggest(q: str = Query(..., min_length=1)):
    base = q.strip()
    variants = list({base, f"{base} news", f"{base} tutorial", f"{base} live", f"{base} shorts"})
    return SuggestResponse(suggestions=variants[:8])

@app.get("/v1/search", response_model=SearchResponse)
async def search(q: str, page: int = 1):
    path = "/api/v1/search"
    params = {"q": q, "page": page, "type": "all"}
    data = await _get_json_with_failover(path, params, cacheable=True)
    if not isinstance(data, list):
        data = []
    items: List[dict] = []
    for it in data:
        t = it.get("type")
        if t == "video":
            items.append({"type": "video", "data": map_invidious_video(it).model_dump()})
        elif t == "channel":
            items.append({"type": "channel", "data": map_invidious_channel(it).model_dump()})
    next_page = page + 1 if len(data) > 0 else None
    return SearchResponse(items=items, nextPage=next_page)

@app.get("/v1/channels", response_model=SearchResponse)
async def channels(q: str, page: int = 1):
    path = "/api/v1/search"
    params = {"q": q, "page": page, "type": "channel"}
    data = await _get_json_with_failover(path, params, cacheable=True)
    if not isinstance(data, list):
        data = []
    items = [{"type": "channel", "data": map_invidious_channel(it).model_dump()} for it in data]
    return SearchResponse(items=items, nextPage=(page + 1 if items else None))

@app.get("/v1/trending", response_model=SearchResponse)
async def trending(page: int = 1, region: str = "US"):
    path = "/api/v1/trending"
    params = {"region": region}
    data = await _get_json_with_failover(path, params, cacheable=True)
    if not isinstance(data, list):
        data = []
    per_page = 20
    start = (page - 1) * per_page
    end = start + per_page
    chunk = data[start:end]
    items = [{"type": "video", "data": map_invidious_video(v).model_dump()} for v in chunk]
    next_page = page + 1 if end < len(data) else None
    return SearchResponse(items=items, nextPage=next_page)

@app.get("/v1/shorts", response_model=ShortItemsResponse)
async def shorts(page: int = 1, region: str = "US"):
    """
    Use Invidious trending with type=Shorts (supported by many instances).
    If an instance doesn’t support it we’ll failover to the next.
    """
    path = "/api/v1/trending"
    params = {"region": region, "type": "Shorts"}
    data = await _get_json_with_failover(path, params, cacheable=True)
    if not isinstance(data, list):
        data = []

    per_page = 20
    start = (page - 1) * per_page
    end = start + per_page
    chunk = data[start:end]
    vids = [map_invidious_video(v).model_dump() for v in chunk]
    next_page = page + 1 if end < len(data) else None
    return ShortItemsResponse(items=vids, nextPage=next_page)

@app.get("/v1/videoStats")
async def video_stats(ids: str = Query(..., description="comma-separated videoIds")) -> Dict[str, Dict[str, int]]:
    """
    Returns: { videoId: { views, likes, comments } }
    """
    out: Dict[str, Dict[str, int]] = {}
    id_list = [x.strip() for x in ids.split(",") if x.strip()]

    # Try current base first, then rotate on errors for each ID independently.
    async def fetch_one(vid: str) -> Dict[str, int]:
        path = f"/api/v1/videos/{vid}"
        params = {}
        try:
            j = await _get_json_with_failover(path, params, cacheable=True)
            def to_int(x):
                try:
                    return int(x)
                except Exception:
                    return 0
            return {
                "views": to_int(j.get("viewCount")),
                "likes": to_int(j.get("likeCount")),
                "comments": to_int(j.get("commentCount")),
            }
        except Exception:
            return {"views": 0, "likes": 0, "comments": 0}

    for vid in id_list:
        out[vid] = await fetch_one(vid)

    return out

# -------- Nice JSON for unhandled errors ----------
@app.exception_handler(Exception)
async def unhandled_exceptions(request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {str(exc)}"})
