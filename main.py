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
POOL_ENV = os.getenv(
    "INVIDIOUS_POOL",
    ",".join([
        "https://inv.nadeko.net",
        "https://invidious.fdn.fr",
        "https://yewtu.be",
        "https://vid.puffyan.us",
        "https://invidious.nerdvpn.de",
        "https://iv.nboeck.de",
        "https://invidious.privacydev.net",
    ])
)
INVIDIOUS_POOL: List[str] = [b.strip().rstrip("/") for b in POOL_ENV.split(",") if b.strip()]
if not INVIDIOUS_POOL:
    INVIDIOUS_POOL = [DEFAULT_INVIDIOUS]

PIPED_BASE = os.getenv("PIPED_BASE", "https://piped.video").rstrip("/")
CORS = os.getenv("CORS_ORIGINS", "*")
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5 minutes

USER_AGENT = os.getenv(
    "USER_AGENT",
    "PrimeTubeBackend/1.0 (CloudRun) httpx"
)

# ------------------ App ---------------------
app = FastAPI(title="PrimeTube Backend", version="1.2.0")

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

# Comments models
class CommentItem(BaseModel):
    id: Optional[str] = None
    author: Optional[str] = None
    authorId: Optional[str] = None
    text: str
    publishedText: Optional[str] = None
    likeCount: Optional[int] = 0
    replyCount: Optional[int] = 0

class CommentsResponse(BaseModel):
    disabled: bool = False
    items: List[CommentItem] = []
    nextPage: Optional[str] = None  # continuation/nextpage token if available

# ------------------ Small cache ------------------
# key -> (expires_at, data)
_cache: Dict[str, Tuple[float, object]] = {}

def _cache_key(path: str, params: dict) -> str:
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

# ------------------ Invidious helpers ------------------
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

def map_invidious_comment(c: dict) -> CommentItem:
    return CommentItem(
        id=c.get("commentId") or c.get("id"),
        author=c.get("author"),
        authorId=c.get("authorId"),
        text=c.get("content") or c.get("contentHtml") or "",
        publishedText=c.get("publishedText") or c.get("published"),
        likeCount=int(c.get("likeCount") or 0),
        replyCount=int(c.get("repliesCount") or 0),
    )

_pool_cycle = itertools.cycle(INVIDIOUS_POOL)

async def _fetch_json_once(base: str, path: str, params: dict) -> object:
    url = f"{base}{path}"
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

async def _get_json_with_failover(path: str, params: dict, cacheable: bool = True) -> object:
    cached = cache_get(path, params) if cacheable else None
    if cached is not None:
        return cached

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
            # Retry on 429 or 5xx; error out for other 4xx.
            if status == 429 or (500 <= status < 600):
                await _async_backoff(status)
                last_error = e
                continue
            raise HTTPException(status_code=status, detail=f"Upstream error {status} at {base}{path}") from e
        except Exception as e:
            last_error = e
            await _async_backoff()
            continue

    if cacheable:
        stale = cache_get(path, params)
        if stale is not None:
            return stale

    raise HTTPException(status_code=502, detail=f"All Invidious mirrors failed for {path}: {last_error}")

async def _async_backoff(status: Optional[int] = None):
    import asyncio
    delay = 0.25 if status == 429 else 0.15
    await asyncio.sleep(delay)

# ------------------ Piped helpers ------------------
def _piped_extract_id_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(url).query)
        return (qs.get("v") or [""])[0]
    except Exception:
        return ""

def map_piped_video(v: dict) -> VideoItem:
    vid = v.get("id") or _piped_extract_id_from_url(v.get("url", ""))
    thumbs: List[Thumbnail] = []
    thumb_url = v.get("thumbnail") or v.get("thumbnailUrl")
    if thumb_url:
        thumbs = [Thumbnail(url=thumb_url, width=None, height=None)]
    views = 0
    try:
        views = int(v.get("views") or 0)
    except Exception:
        views = 0
    return VideoItem(
        id=vid or "",
        title=v.get("title", ""),
        channelName=v.get("uploaderName") or v.get("uploader"),
        channelId=v.get("uploaderId"),
        viewCount=views,
        publishedText=v.get("uploaded") or v.get("uploadedDate"),
        thumbnails=thumbs,
    )

def map_piped_comment(c: dict) -> CommentItem:
    # Piped typically: commentId, author, commentText, commentedTime, likeCount, replies?
    return CommentItem(
        id=c.get("commentId") or c.get("id"),
        author=c.get("author"),
        authorId=c.get("authorId"),
        text=c.get("commentText") or c.get("text") or "",
        publishedText=c.get("commentedTime") or c.get("published"),
        likeCount=int(c.get("likeCount") or 0),
        replyCount=int(c.get("replyCount") or 0),
    )

async def _piped_json(path: str, params: dict) -> object:
    url = f"{PIPED_BASE}{path}"
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

# ---------- Unified trending/shorts with deep fallback ----------
async def _trending_videos(region: str, shorts_only: bool) -> List[dict]:
    """
    Returns raw list of video dicts from any of:
    1) Invidious /trending (optionally Shorts)
    2) Piped /trending (or Piped search trending for #shorts)
    3) Invidious /popular
    4) Invidious /search sort_by=trending (q empty or #shorts)
    5) Piped /api/search sort=trending (q empty or #shorts)
    """
    # ---- 1) Invidious: /trending ----
    inv_params = {"region": region}
    if shorts_only:
        inv_params["type"] = "Shorts"
    try:
        data = await _get_json_with_failover("/api/v1/trending", inv_params, cacheable=True)
        if isinstance(data, list) and data:
            return data
    except Exception:
        pass

    # ---- 2) Piped ----
    try:
        if shorts_only:
            data = await _piped_json("/api/search", {
                "q": "#shorts",
                "filter": "videos",
                "sort": "trending",
                "region": region,
            })
            items = data.get("items") if isinstance(data, dict) else None
            if items:
                return items
        else:
            data = await _piped_json("/api/trending", {"region": region})
            if isinstance(data, list) and data:
                return data
    except Exception:
        pass

    # ---- 3) Invidious: /popular ----
    try:
        data = await _get_json_with_failover("/api/v1/popular", {}, cacheable=True)
        if isinstance(data, list) and data:
            return data
    except Exception:
        pass

    # ---- 4) Invidious: /search with sort_by=trending ----
    try:
        inv_search_params = {
            "q": "#shorts" if shorts_only else "",
            "type": "video",
            "sort_by": "trending",
            "region": region,
        }
        inv_search = await _get_json_with_failover("/api/v1/search", inv_search_params, cacheable=True)
        if isinstance(inv_search, list) and inv_search:
            return inv_search
    except Exception:
        pass

    # ---- 5) Piped: /api/search with sort=trending ----
    try:
        piped_search = await _piped_json("/api/search", {
            "q": "#shorts" if shorts_only else "",
            "filter": "videos",
            "sort": "trending",
            "region": region,
        })
        items = piped_search.get("items") if isinstance(piped_search, dict) else None
        if items:
            return items
    except Exception:
        pass

    return []

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
    data = await _trending_videos(region=region, shorts_only=False)
    per_page = 20
    start = (page - 1) * per_page
    end = start + per_page
    chunk = data[start:end]

    items = []
    for v in chunk:
        if "type" in v or "videoThumbnails" in v:  # invidious-like
            items.append({"type": "video", "data": map_invidious_video(v).model_dump()})
        else:  # piped-like
            items.append({"type": "video", "data": map_piped_video(v).model_dump()})

    next_page = page + 1 if end < len(data) else None
    return SearchResponse(items=items, nextPage=next_page)

@app.get("/v1/shorts", response_model=ShortItemsResponse)
async def shorts(page: int = 1, region: str = "US"):
    data = await _trending_videos(region=region, shorts_only=True)
    per_page = 20
    start = (page - 1) * per_page
    end = start + per_page
    chunk = data[start:end]

    vids = []
    for v in chunk:
        if "type" in v or "videoThumbnails" in v:
            vids.append(map_invidious_video(v).model_dump())
        else:
            vids.append(map_piped_video(v).model_dump())

    next_page = page + 1 if end < len(data) else None
    return ShortItemsResponse(items=vids, nextPage=next_page)

@app.get("/v1/videoStats")
async def video_stats(ids: str = Query(..., description="comma-separated videoIds")) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    id_list = [x.strip() for x in ids.split(",") if x.strip()]

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

# -------- Comments (Invidious → Piped fallback) ----------
@app.get("/v1/comments", response_model=CommentsResponse)
async def comments(
    videoId: str = Query(..., min_length=3),
    continuation: Optional[str] = Query(None, description="Invidious continuation token"),
    region: str = "US"
):
    # 1) Try Invidious
    try:
        params = {"hl": "en"}
        if continuation:
            params["continuation"] = continuation
        path = f"/api/v1/comments/{videoId}"
        data = await _get_json_with_failover(path, params, cacheable=False)

        if isinstance(data, dict):
            disabled = bool(data.get("disabled") or data.get("commentsDisabled", False))
            raw = data.get("comments") or []
            items = [map_invidious_comment(c).model_dump() for c in raw if isinstance(c, dict)]
            next_token = data.get("continuation")
            return CommentsResponse(disabled=disabled, items=items, nextPage=next_token)
    except Exception:
        pass

    # 2) Fallback Piped
    try:
        piped_params = {}
        if continuation:
            piped_params["nextpage"] = continuation
        data = await _piped_json(f"/api/comments/{videoId}", piped_params)
        # Piped shape: { comments:[...], nextpage?: "...", disabled?: bool }
        if isinstance(data, dict):
            disabled = bool(data.get("disabled", False))
            raw = data.get("comments") or data.get("items") or []
            items = [map_piped_comment(c).model_dump() for c in raw if isinstance(c, dict)]
            next_token = data.get("nextpage") or data.get("nextPage")
            return CommentsResponse(disabled=disabled, items=items, nextPage=next_token)
        if isinstance(data, list):
            items = [map_piped_comment(c).model_dump() for c in data if isinstance(c, dict)]
            return CommentsResponse(disabled=False, items=items, nextPage=None)
    except Exception:
        pass

    # If both sources failed or comments truly disabled:
    return CommentsResponse(disabled=True, items=[], nextPage=None)

# -------- Nice JSON for unhandled errors ----------
@app.exception_handler(Exception)
async def unhandled_exceptions(request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {str(exc)}"})
