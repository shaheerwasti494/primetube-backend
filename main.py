import os
import time
import json
import itertools
from typing import Dict, List, Optional, Tuple, Union
from fastapi import FastAPI, Query, HTTPException, Response
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

# Piped mirror pool with failover
PIPED_POOL_ENV = os.getenv(
    "PIPED_POOL",
    ",".join([
        "https://piped.video",
        "https://piped.projectsegfau.lt",
        "https://piped.privacydev.net",
        "https://piped.mha.fi",
        "https://piped.garudalinux.org",
    ])
)
PIPED_POOL: List[str] = [b.strip().rstrip("/") for b in PIPED_POOL_ENV.split(",") if b.strip()]
if not PIPED_POOL:
    PIPED_POOL = ["https://piped.video"]

CORS = os.getenv("CORS_ORIGINS", "*")
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5 minutes

USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 "
    "PrimeTubeBackend/1.0 (CloudRun) httpx"
)

DEBUG_UPSTREAM = os.getenv("DEBUG_UPSTREAM", "0") == "1"

def _dbg(msg: str):
    if DEBUG_UPSTREAM:
        print(f"[UPSTREAM] {msg}")

# ------------------ App ---------------------
app = FastAPI(title="PrimeTube Backend", version="1.6.0")

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

def _cacheable_nonempty(data: object) -> bool:
    if isinstance(data, list):
        return len(data) > 0
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return len(items) > 0
    return False

# ------------------ Helpers ------------------
def _merge_unique(seq: List[dict], key: str) -> List[dict]:
    seen = set()
    out = []
    for it in seq:
        k = it.get(key) or it.get("id") or it.get("videoId") or it.get("url")
        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out

# ------------------ Invidious helpers ------------------
def thumbify(arr) -> List[Thumbnail]:
    if not isinstance(arr, list):
        return []
    out: List[Thumbnail] = []
    for t in arr:
        url = t.get("url")
        # Normalize relative /vi/... to absolute
        if isinstance(url, str) and url.startswith("/vi/"):
            url = f"https://i.ytimg.com{url}"
        out.append(Thumbnail(url=url, width=t.get("width"), height=t.get("height")))
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
        last = thumbs[-1].get("url")
        if isinstance(last, str) and last.startswith("/vi/"):
            last = f"https://i.ytimg.com{last}"
        avatar = last
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

_inv_cycle = itertools.cycle(INVIDIOUS_POOL)
_piped_cycle = itertools.cycle(PIPED_POOL)

# ---------- HTTP helpers ----------
async def _http_get_json(url: str, params: dict, accept_json: bool = True) -> object:
    headers = {"User-Agent": USER_AGENT}
    if accept_json:
        headers["Accept"] = "application/json"
    async with httpx.AsyncClient(timeout=20.0, headers=headers, follow_redirects=True) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        try:
            return r.json()
        except Exception as e:
            # If server said JSON but body isn't JSON, surface a clear error
            text = r.text
            raise httpx.HTTPStatusError(
                f"Non-JSON body from {url}: {text[:120]}...",
                request=r.request,
                response=r
            ) from e

async def _fetch_json_once(base: str, path: str, params: dict) -> object:
    url = f"{base}{path}"
    return await _http_get_json(url, params, accept_json=True)

async def _async_backoff(status: Optional[int] = None):
    import asyncio
    delay = 0.40 if status == 429 else 0.20
    await asyncio.sleep(delay)

async def _get_json_with_failover(
    path: str,
    params: dict,
    cacheable: bool = True,
    require_nonempty_list: bool = False,
    require_items_key_nonempty: bool = False,
) -> object:
    """Invidious pool failover."""
    cached = cache_get(path, params) if cacheable else None
    if cached is not None:
        return cached

    last_error: Optional[Exception] = None
    tried = set()
    for _ in range(len(INVIDIOUS_POOL)):
        base = next(_inv_cycle)
        if base in tried:
            continue
        tried.add(base)
        try:
            data = await _fetch_json_once(base, path, params)
            if require_nonempty_list and isinstance(data, list) and not data:
                _dbg(f"{base}{path} empty list; next mirror")
                continue
            if require_items_key_nonempty and isinstance(data, dict):
                items = data.get("items")
                if isinstance(items, list) and not items:
                    _dbg(f"{base}{path} empty items[]; next mirror")
                    continue
            if cacheable and _cacheable_nonempty(data):
                cache_set(path, params, data)
            return data
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 429 or (500 <= status < 600):
                last_error = e
                await _async_backoff(status)
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

    raise HTTPException(status_code=502, detail=f"Invidious mirrors failed for {path}: {last_error}")

# ---- Piped failover mirrors ----
async def _piped_json_with_failover(path: str, params: dict) -> object:
    last_error: Optional[Exception] = None
    tried = set()
    for _ in range(len(PIPED_POOL)):
        base = next(_piped_cycle)
        if base in tried:
            continue
        tried.add(base)
        url = f"{base}{path}"
        try:
            data = await _http_get_json(url, params, accept_json=True)
            # Piped returns list for /api/trending and dict with items for /api/search, /api/comments
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(f"Piped error at {url}: {data.get('error')}")
            if isinstance(data, dict) and "items" in data and not data["items"]:
                _dbg(f"{url} empty items[]; next mirror")
                continue
            if isinstance(data, list) and not data:
                _dbg(f"{url} empty list; next mirror")
                continue
            return data
        except Exception as e:
            last_error = e
            _dbg(f"piped fail {url}: {e}")
            await _async_backoff()
            continue
    raise HTTPException(status_code=502, detail=f"Piped mirrors failed for {path}: {last_error}")

# ------------------ Piped/Mapper helpers ------------------
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
        channelName=v.get("uploaderName") or v.get("uploader") or v.get("author"),
        channelId=v.get("uploaderId") or v.get("channelId"),
        viewCount=views,
        publishedText=v.get("uploaded") or v.get("uploadedDate"),
        thumbnails=thumbs,
    )

def map_piped_channel(c: dict) -> ChannelItem:
    return ChannelItem(
        id=c.get("id") or c.get("uploaderId") or c.get("channelId") or "",
        title=c.get("name") or c.get("uploader") or c.get("author") or c.get("title"),
        avatar=c.get("thumbnail") or c.get("avatarUrl"),
        subscribersText=c.get("subscriberCountText") or c.get("subscribersText"),
    )

def map_piped_comment(c: dict) -> CommentItem:
    return CommentItem(
        id=c.get("commentId") or c.get("id"),
        author=c.get("author"),
        authorId=c.get("authorId"),
        text=c.get("commentText") or c.get("text") or "",
        publishedText=c.get("commentedTime") or c.get("published"),
        likeCount=int(c.get("likeCount") or 0),
        replyCount=int(c.get("replyCount") or 0),
    )

# ---------- Unified trending/shorts with deep fallback ----------
async def _trending_videos(region: str, shorts_only: bool) -> Tuple[List[dict], str]:
    # 1) Invidious /trending
    inv_params = {"region": region}
    if shorts_only:
        inv_params["type"] = "Shorts"
    try:
        data = await _get_json_with_failover(
            "/api/v1/trending",
            inv_params,
            cacheable=True,
            require_nonempty_list=True,
        )
        if isinstance(data, list) and data:
            return data, "invidious_trending"
    except Exception as e:
        _dbg(f"invidious /trending error: {e}")

    # 2) Piped /api/trending
    try:
        data = await _piped_json_with_failover("/api/trending", {"region": region})
        if isinstance(data, list) and data:
            return data, "piped_trending"
    except Exception as e:
        _dbg(f"piped /trending error: {e}")

    # 3) Invidious /popular
    try:
        data = await _get_json_with_failover(
            "/api/v1/popular",
            {},
            cacheable=True,
            require_nonempty_list=True,
        )
        if isinstance(data, list) and data:
            return data, "invidious_popular"
    except Exception as e:
        _dbg(f"invidious /popular error: {e}")

    # 4) Invidious search trending (#shorts only)
    if shorts_only:
        try:
            inv_search_params = {
                "q": "#shorts",
                "type": "video",
                "sort_by": "trending",
                "region": region,
            }
            inv_search = await _get_json_with_failover(
                "/api/v1/search",
                inv_search_params,
                cacheable=True,
                require_nonempty_list=True,
            )
            if isinstance(inv_search, list) and inv_search:
                return inv_search, "invidious_search_shorts_trending"
        except Exception as e:
            _dbg(f"invidious search trending error: {e}")

    # 5) Piped search(sort=trending) — works for shorts and non-shorts
    try:
        piped_search = await _piped_json_with_failover("/api/search", {
            "q": "#shorts" if shorts_only else "",
            "filter": "videos",
            "sort": "trending",
            "region": region,
        })
        items = piped_search.get("items") if isinstance(piped_search, dict) else None
        if items:
            return items, "piped_search_trending"
    except Exception as e:
        _dbg(f"piped search trending error: {e}")

    return [], "none"

# ---------- Unified Search (deep Invidious + Piped fallback/merge) ----------
async def _unified_search(q: str, page: int, kind: str, region: str) -> List[dict]:
    # Invidious: try "all", then explicit "video" and "channel" and merge
    try:
        merged: List[dict] = []
        base_params = {"q": q, "page": page, "region": region}

        inv_all = await _get_json_with_failover("/api/v1/search", base_params, cacheable=True)
        if isinstance(inv_all, list):
            merged.extend(inv_all)

        inv_vid = await _get_json_with_failover("/api/v1/search",
                                                {**base_params, "type": "video"},
                                                cacheable=True)
        if isinstance(inv_vid, list):
            merged.extend(inv_vid)

        inv_ch = await _get_json_with_failover("/api/v1/search",
                                               {**base_params, "type": "channel"},
                                               cacheable=True)
        if isinstance(inv_ch, list):
            merged.extend(inv_ch)

        merged = _merge_unique(merged, key="videoId")
        if merged:
            return merged
    except Exception as e:
        _dbg(f"invidious unified search error: {e}")

    # Piped: when "all", query both videos and channels, then merge
    try:
        piped_items: List[dict] = []
        if kind == "channel":
            pip = await _piped_json_with_failover("/api/search", {
                "q": q, "filter": "channels", "sort": "relevance", "page": page
            })
            items = pip.get("items") if isinstance(pip, dict) else None
            if items:
                piped_items.extend(items)
        else:
            pip_v = await _piped_json_with_failover("/api/search", {
                "q": q, "filter": "videos", "sort": "relevance", "page": page
            })
            v_items = pip_v.get("items") if isinstance(pip_v, dict) else None
            if v_items:
                piped_items.extend(v_items)

            if kind in ("all",):
                pip_c = await _piped_json_with_failover("/api/search", {
                    "q": q, "filter": "channels", "sort": "relevance", "page": page
                })
                c_items = pip_c.get("items") if isinstance(pip_c, dict) else None
                if c_items:
                    piped_items.extend(c_items)

        piped_items = _merge_unique(piped_items, key="id")
        return piped_items
    except Exception as e:
        _dbg(f"piped unified search error: {e}")

    return []

# ------------------ Endpoints ------------------
@app.get("/health")
async def health(deep: bool = False):
    info: Dict[str, Union[bool, int, List[str], str, List[dict]]] = {
        "ok": True,
        "active_pool": INVIDIOUS_POOL,
        "piped_pool": PIPED_POOL,
        "cache_ttl": CACHE_TTL,
    }
    if deep:
        checks = []
        try:
            d = await _get_json_with_failover("/api/v1/trending", {"region":"US"}, cacheable=False, require_nonempty_list=False)
            checks.append({"invidious_trending_len": len(d) if isinstance(d, list) else None})
        except Exception as e:
            checks.append({"invidious_trending_error": str(e)})
        try:
            d = await _piped_json_with_failover("/api/trending", {"region":"US"})
            checks.append({"piped_trending_len": len(d) if isinstance(d, list) else None})
        except Exception as e:
            checks.append({"piped_trending_error": str(e)})
        info["checks"] = checks
    return info

@app.get("/v1/suggest", response_model=SuggestResponse)
async def suggest(q: str = Query(..., min_length=1)):
    base = q.strip()
    variants = list({base, f"{base} news", f"{base} tutorial", f"{base} live", f"{base} shorts"})
    return SuggestResponse(suggestions=variants[:8])

@app.get("/v1/search", response_model=SearchResponse)
async def search(q: str, page: int = 1, region: str = "US"):
    raw = await _unified_search(q=q, page=page, kind="all", region=region)
    items: List[dict] = []
    for it in raw:
        t = it.get("type")

        # Invidious explicit
        if t == "video":
            items.append({"type": "video", "data": map_invidious_video(it).model_dump()})
            continue
        if t == "channel":
            items.append({"type": "channel", "data": map_invidious_channel(it).model_dump()})
            continue

        # Heuristics for Piped items (search results)
        if any(k in it for k in ("uploader", "uploaderName", "url", "thumbnail", "thumbnailUrl")):
            items.append({"type": "video", "data": map_piped_video(it).model_dump()})
        elif any(k in it for k in ("name", "avatarUrl")) or (it.get("channelId") and not it.get("url")):
            items.append({"type": "channel", "data": map_piped_channel(it).model_dump()})
        else:
            # Fallback try video then channel
            try:
                items.append({"type": "video", "data": map_piped_video(it).model_dump()})
            except Exception:
                items.append({"type": "channel", "data": map_piped_channel(it).model_dump()})

    next_page = page + 1 if raw else None
    return SearchResponse(items=items, nextPage=next_page)

@app.get("/v1/channels", response_model=SearchResponse)
async def channels(q: str, page: int = 1, region: str = "US"):
    raw = await _unified_search(q=q, page=page, kind="channel", region=region)
    items: List[dict] = []
    for it in raw:
        if it.get("type") == "channel":
            ch = map_piped_channel(it).model_dump()
            if not ch.get("id") and (it.get("authorId") or it.get("author")):
                ch = map_invidious_channel(it).model_dump()
            items.append({"type": "channel", "data": ch})
        elif any(k in it for k in ("authorId", "author", "authorThumbnails", "thumbnails")):
            items.append({"type": "channel", "data": map_invidious_channel(it).model_dump()})
        else:
            items.append({"type": "channel", "data": map_piped_channel(it).model_dump()})
    next_page = page + 1 if raw else None
    return SearchResponse(items=items, nextPage=next_page)

@app.get("/v1/trending", response_model=SearchResponse)
async def trending(response: Response, page: int = 1, region: str = "US"):
    data, source = await _trending_videos(region=region, shorts_only=False)
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
    if DEBUG_UPSTREAM:
        response.headers["X-PT-Source"] = source
        response.headers["X-PT-Total"] = str(len(data))
    return SearchResponse(items=items, nextPage=next_page)

@app.get("/v1/shorts", response_model=ShortItemsResponse)
async def shorts(response: Response, page: int = 1, region: str = "US"):
    data, source = await _trending_videos(region=region, shorts_only=True)
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
    if DEBUG_UPSTREAM:
        response.headers["X-PT-Source"] = source
        response.headers["X-PT-Total"] = str(len(data))
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

# -------- Related videos (Invidious → Piped fallback) ----------
def map_invidious_related_item(x: dict) -> dict:
    return {"type": "video", "data": map_invidious_video(x).model_dump()}

def map_piped_related_item(x: dict) -> dict:
    return {"type": "video", "data": map_piped_video(x).model_dump()}

@app.get("/v1/related", response_model=SearchResponse)
async def related(videoId: str = Query(..., min_length=3), page: int = 1, region: str = "US"):
    # 1) Invidious
    try:
        data = await _get_json_with_failover(f"/api/v1/related/{videoId}", {"region": region}, cacheable=True)
        if isinstance(data, list) and data:
            items = [map_invidious_related_item(d) for d in data if isinstance(d, dict)]
            per_page = 20
            start, end = (page - 1) * per_page, page * per_page
            chunk = items[start:end]
            return SearchResponse(items=chunk, nextPage=(page + 1 if end < len(items) else None))
    except Exception as e:
        _dbg(f"invidious related error: {e}")

    # 2) Piped fallback
    try:
        pip = await _piped_json_with_failover(f"/api/related/{videoId}", {})
        if isinstance(pip, dict):
            rel = pip.get("relatedStreams") or pip.get("items") or []
        else:
            rel = pip if isinstance(pip, list) else []
        items = [map_piped_related_item(d) for d in rel if isinstance(d, dict)]
        per_page = 20
        start, end = (page - 1) * per_page, page * per_page
        chunk = items[start:end]
        return SearchResponse(items=chunk, nextPage=(page + 1 if end < len(items) else None))
    except Exception as e:
        _dbg(f"piped related error: {e}")

    return SearchResponse(items=[], nextPage=None)

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
        data = await _piped_json_with_failover(f"/api/comments/{videoId}", piped_params)
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

    return CommentsResponse(disabled=True, items=[], nextPage=None)

# -------- Nice JSON for unhandled errors ----------
@app.exception_handler(Exception)
async def unhandled_exceptions(request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {str(exc)}"})
