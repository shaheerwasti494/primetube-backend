"""
PrimeTube Backend — v1.8.0 (FastAPI)

Highlights vs 1.7.x
- Robust upstream failover (Invidious + Piped) with clear 5xx when all mirrors fail
- Non‑JSON upstream responses (CAPTCHA/HTML) are treated as failures, not success
- Provider/debug headers for search/channels/trending/shorts/related when DEBUG_UPSTREAM=1
- Safer JSON mapping and dedupe across mirrors
- Health deep check endpoint improvements

Environment variables
- INVIDIOUS_POOL: comma‑separated list (default set below)
- PIPED_POOL: comma‑separated list (default set below)
- USER_AGENT: UA string (default: Chrome-like PrimeTubeBackend/1.0)
- CORS_ORIGINS: origins or "*" (default "*")
- CACHE_TTL_SECONDS: in‑memory cache TTL for non-auth GETs (default 300)
- DEBUG_UPSTREAM: if "1", adds X‑PT-* headers and prints upstream debug

Run locally
- uvicorn primetube_backend_v1_8_0:app --host 0.0.0.0 --port 8080 --reload

Deploy (Cloud Run, example)
- Build: gcloud builds submit --tag gcr.io/PROJECT/primetube-backend:v1.8.0
- Deploy: gcloud run deploy primetube-backend --image gcr.io/PROJECT/primetube-backend:v1.8.0 --allow-unauthenticated --region=us-central1 --set-env-vars=DEBUG_UPSTREAM=1
"""
from __future__ import annotations

import os
import time
import json
import itertools
from typing import Dict, List, Optional, Tuple, Union, Iterable

from fastapi import FastAPI, Query, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

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
    "PrimeTubeBackend/1.8.0 (CloudRun) httpx"
)
DEBUG_UPSTREAM = os.getenv("DEBUG_UPSTREAM", "0") == "1"


def _dbg(msg: str):
    if DEBUG_UPSTREAM:
        print(f"[UPSTREAM] {msg}")


# ------------------ App ---------------------
app = FastAPI(title="PrimeTube Backend", version="1.8.0")
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
    items: List[dict]
    nextPage: Optional[int] = None


class SuggestResponse(BaseModel):
    suggestions: List[str]


class ShortItemsResponse(BaseModel):
    items: List[VideoItem]
    nextPage: Optional[int] = None


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
    nextPage: Optional[str] = None


# ------------------ Small cache ------------------
_cache: Dict[str, Tuple[float, object]] = {}

def _cache_key(path: str, params: dict) -> str:
    return f"{path}?{json.dumps(params, sort_keys=True, separators=(',',':'))}"

def cache_get(path: str, params: dict):
    if CACHE_TTL <= 0:
        return None
    k = _cache_key(path, params)
    it = _cache.get(k)
    if not it:
        return None
    exp, data = it
    if time.time() < exp:
        return data
    _cache.pop(k, None)
    return None

def cache_set(path: str, params: dict, data: object, ttl: Optional[int] = None):
    if CACHE_TTL <= 0:
        return
    k = _cache_key(path, params)
    _cache[k] = (time.time() + (ttl if ttl is not None else CACHE_TTL), data)

def _cacheable_nonempty(data: object) -> bool:
    if isinstance(data, list):
        return len(data) > 0
    if isinstance(data, dict):
        it = data.get("items")
        if isinstance(it, list):
            return len(it) > 0
    return False


# ------------------ Utils ------------------
def _pick_id(it: dict) -> Optional[str]:
    return (
        it.get("videoId")
        or it.get("id")
        or it.get("url")
        or it.get("authorId")
        or it.get("ucid")
        or it.get("channelId")
    )


def _merge_unique(seq: Iterable[dict]) -> List[dict]:
    seen, out = set(), []
    for it in seq:
        if not isinstance(it, dict):
            continue
        k = _pick_id(it)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


# ------------------ Invidious mappers ------------------

def thumbify(arr) -> List[Thumbnail]:
    if not isinstance(arr, list):
        return []
    out: List[Thumbnail] = []
    for t in arr:
        url = t.get("url") if isinstance(t, dict) else None
        if isinstance(url, str) and url.startswith("/vi/"):
            url = f"https://i.ytimg.com{url}"
        out.append(Thumbnail(url=url, width=t.get("width") if isinstance(t, dict) else None, height=t.get("height") if isinstance(t, dict) else None))
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
    if isinstance(thumbs, list) and thumbs:
        last = thumbs[-1].get("url") if isinstance(thumbs[-1], dict) else None
        if isinstance(last, str) and last.startswith("/vi/"):
            last = f"https://i.ytimg.com{last}"
        avatar = last
    return ChannelItem(
        id=c.get("authorId") or c.get("ucid") or c.get("id", ""),
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


# ------------------ Piped mappers ------------------

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
    u = v.get("thumbnail") or v.get("thumbnailUrl")
    if u:
        thumbs = [Thumbnail(url=u)]
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


# ------------------ HTTP helpers ------------------
_inv_cycle = itertools.cycle(INVIDIOUS_POOL)
_piped_cycle = itertools.cycle(PIPED_POOL)


async def _http_get_json(url: str, params: dict, accept_json: bool = True) -> object:
    headers = {
        "User-Agent": USER_AGENT,
        # Be permissive; some instances send odd content-types
        "Accept": "application/json, text/json, */*",
    }
    if not accept_json:
        headers.pop("Accept", None)

    timeout_s = float(os.getenv("HTTPX_TIMEOUT_SECONDS", "25.0"))

    async with httpx.AsyncClient(timeout=timeout_s, headers=headers, follow_redirects=True) as client:
        r = await client.get(url, params=params)
        # If non-2xx, raise.
        r.raise_for_status()
        try:
            return r.json()
        except Exception as e:
            # Treat non-JSON as a failure so caller can failover.
            ct = r.headers.get("content-type", "")
            textpeek = r.text[:200]
            raise RuntimeError(
                f"Non-JSON response from {url} (status={r.status_code}, ct={ct}): {textpeek}"
            ) from e


async def _fetch_json_once(base: str, path: str, params: dict) -> object:
    return await _http_get_json(f"{base}{path}", params, accept_json=True)


async def _async_backoff(status: Optional[int] = None):
    import asyncio
    await asyncio.sleep(0.4 if status == 429 else 0.2)


async def _get_json_with_failover(
    path: str,
    params: dict,
    cacheable: bool = True,
    require_nonempty_list: bool = False,
    require_items_key_nonempty: bool = False,
) -> object:
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
                continue
            if require_items_key_nonempty and isinstance(data, dict):
                items = data.get("items")
                if isinstance(items, list) and not items:
                    continue
            if cacheable and _cacheable_nonempty(data):
                cache_set(path, params, data)
            return data
        except httpx.HTTPStatusError as e:
            st = e.response.status_code
            last_error = e
            if st == 429 or 500 <= st < 600:
                await _async_backoff(st)
                continue
            # Non-retryable HTTP errors bubble up immediately
            raise HTTPException(status_code=st, detail=f"Upstream error {st} at {base}{path}") from e
        except Exception as e:
            # Non-HTTP (parse, JSON, TLS, etc.): try next mirror
            last_error = e
            await _async_backoff()
            continue

    if cacheable:
        stale = cache_get(path, params)
        if stale is not None:
            return stale

    raise HTTPException(status_code=502, detail=f"Invidious mirrors failed for {path}: {last_error}")


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
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(data["error"])
            if isinstance(data, dict) and "items" in data and not data["items"]:
                continue
            if isinstance(data, list) and not data:
                continue
            return data
        except Exception as e:
            last_error = e
            _dbg(f"piped fail {url}: {e}")
            await _async_backoff()
            continue
    raise HTTPException(status_code=502, detail=f"Piped mirrors failed for {path}: {last_error}")


# ------------------ Trending/Shorts helpers ------------------
async def _trending_videos(region: str, shorts_only: bool) -> Tuple[List[dict], str]:
    inv_params = {"region": region}
    if shorts_only:
        inv_params["type"] = "Shorts"
    try:
        data = await _get_json_with_failover("/api/v1/trending", inv_params, cacheable=True, require_nonempty_list=True)
        if isinstance(data, list) and data:
            return data, "invidious_trending"
    except Exception as e:
        _dbg(f"invidious /trending error: {e}")

    try:
        data = await _piped_json_with_failover("/api/trending", {"region": region})
        if isinstance(data, list) and data:
            return data, "piped_trending"
    except Exception as e:
        _dbg(f"piped /trending error: {e}")

    try:
        data = await _get_json_with_failover("/api/v1/popular", {}, cacheable=True, require_nonempty_list=True)
        if isinstance(data, list) and data:
            return data, "invidious_popular"
    except Exception as e:
        _dbg(f"invidious /popular error: {e}")

    if shorts_only:
        try:
            inv_search = await _get_json_with_failover(
                "/api/v1/search",
                {"q": "#shorts", "type": "video", "sort_by": "trending", "region": region},
                cacheable=True,
                require_nonempty_list=True,
            )
            if isinstance(inv_search, list) and inv_search:
                return inv_search, "inv_search_shorts_trending"
        except Exception as e:
            _dbg(f"invidious search trending error: {e}")
        try:
            piped_search = await _piped_json_with_failover(
                "/api/search",
                {"q": "#shorts", "filter": "videos", "sort": "trending", "region": region},
            )
            items = piped_search.get("items") if isinstance(piped_search, dict) else None
            if items:
                return items, "piped_search_trending"
        except Exception as e:
            _dbg(f"piped search trending error: {e}")

    return [], "none"


# ------------------ Unified Search (fan-out + merge) ------------------
async def _unified_search(q: str, page: int, kind: str, region: str) -> Tuple[List[dict], str]:
    merged: List[dict] = []
    provider = "none"

    # Invidious fan-out (with and without region)
    for with_region in (True, False):
        base = {"q": q, "page": page}
        if with_region:
            base["region"] = region
        try:
            inv_all = await _get_json_with_failover("/api/v1/search", base, cacheable=True)
            if isinstance(inv_all, list):
                merged.extend(inv_all)
                provider = "invidious"
        except Exception as e:
            _dbg(f"inv all ({'with' if with_region else 'no'} region) err {e}")
        try:
            inv_v = await _get_json_with_failover("/api/v1/search", {**base, "type": "video"}, cacheable=True)
            if isinstance(inv_v, list):
                merged.extend(inv_v)
                provider = "invidious"
        except Exception as e:
            _dbg(f"inv video err {e}")
        try:
            inv_c = await _get_json_with_failover("/api/v1/search", {**base, "type": "channel"}, cacheable=True)
            if isinstance(inv_c, list):
                merged.extend(inv_c)
                provider = "invidious"
        except Exception as e:
            _dbg(f"inv channel err {e}")

    merged = _merge_unique(merged)
    if merged:
        return merged, provider

    # Piped fallback: videos + channels
    try:
        piped_items: List[dict] = []
        if kind in ("all", "video"):
            pv = await _piped_json_with_failover(
                "/api/search",
                {"q": q, "filter": "videos", "sort": "relevance", "page": page, "region": region},
            )
            vi = pv.get("items") if isinstance(pv, dict) else None
            if vi:
                piped_items.extend(vi)
                provider = "piped"
        if kind in ("all", "channel"):
            pc = await _piped_json_with_failover(
                "/api/search",
                {"q": q, "filter": "channels", "sort": "relevance", "page": page, "region": region},
            )
            ci = pc.get("items") if isinstance(pc, dict) else None
            if ci:
                piped_items.extend(ci)
                provider = "piped"
        return _merge_unique(piped_items), provider
    except Exception as e:
        _dbg(f"piped unified search error: {e}")
        return [], provider


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
            d = await _get_json_with_failover("/api/v1/trending", {"region": "US"}, cacheable=False)
            checks.append({"invidious_trending_len": len(d) if isinstance(d, list) else None})
        except Exception as e:
            checks.append({"invidious_trending_error": str(e)})
        try:
            d = await _piped_json_with_failover("/api/trending", {"region": "US"})
            checks.append({"piped_trending_len": len(d) if isinstance(d, list) else None})
        except Exception as e:
            checks.append({"piped_trending_error": str(e)})
        info["checks"] = checks
    return info


@app.get("/v1/suggest", response_model=SuggestResponse)
async def suggest(q: str = Query(..., min_length=1)):
    base = q.strip()
    variants = list({base, f"{base} news", f"{base} tutorial", f"{base} live", f"{base} shorts"})
    # deterministic ordering for better UX
    order = ["tutorial", "live", "shorts", "news", ""]
    def key(s: str) -> int:
        for i, tag in enumerate(order):
            if tag and s.endswith(" " + tag):
                return i
            if tag == "" and (" " not in s):
                return i
        return len(order)
    variants.sort(key=key)
    return SuggestResponse(suggestions=variants[:8])


@app.get("/v1/search", response_model=SearchResponse)
async def search(request: Request, response: Response, q: str, page: int = 1, region: str = "US"):
    raw, provider = await _unified_search(q=q, page=page, kind="all", region=region)
    items: List[dict] = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        t = it.get("type")
        if t == "video":
            items.append({"type": "video", "data": map_invidious_video(it).model_dump()})
            continue
        if t == "channel":
            items.append({"type": "channel", "data": map_invidious_channel(it).model_dump()})
            continue
        # Piped-ish detection:
        if any(k in it for k in ("uploader", "uploaderName", "url", "thumbnail", "thumbnailUrl")):
            items.append({"type": "video", "data": map_piped_video(it).model_dump()})
        elif any(k in it for k in ("name", "avatarUrl", "subscriberCountText", "subscribersText", "channelId")):
            items.append({"type": "channel", "data": map_piped_channel(it).model_dump()})
        else:
            # Last resort: try both mappers
            try:
                items.append({"type": "video", "data": map_piped_video(it).model_dump()})
            except Exception:
                items.append({"type": "channel", "data": map_piped_channel(it).model_dump()})

    if DEBUG_UPSTREAM:
        response.headers["X-PT-Provider"] = provider
        response.headers["X-PT-RawCount"] = str(len(raw))
        response.headers["X-PT-Items"] = str(len(items))

    return SearchResponse(items=items, nextPage=(page + 1 if raw else None))


@app.get("/v1/channels", response_model=SearchResponse)
async def channels(response: Response, q: str, page: int = 1, region: str = "US"):
    raw, provider = await _unified_search(q=q, page=page, kind="channel", region=region)
    items: List[dict] = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        if it.get("type") == "channel":
            ch = map_invidious_channel(it).model_dump()
            if not ch.get("id"):
                ch = map_piped_channel(it).model_dump()
            items.append({"type": "channel", "data": ch})
        elif any(k in it for k in ("authorId", "author", "authorThumbnails", "thumbnails")):
            items.append({"type": "channel", "data": map_invidious_channel(it).model_dump()})
        else:
            items.append({"type": "channel", "data": map_piped_channel(it).model_dump()})

    if DEBUG_UPSTREAM:
        response.headers["X-PT-Provider"] = provider
        response.headers["X-PT-RawCount"] = str(len(raw))
        response.headers["X-PT-Items"] = str(len(items))

    return SearchResponse(items=items, nextPage=(page + 1 if raw else None))


@app.get("/v1/trending", response_model=SearchResponse)
async def trending(response: Response, page: int = 1, region: str = "US"):
    data, source = await _trending_videos(region=region, shorts_only=False)
    per = 20
    start, end = (page - 1) * per, (page - 1) * per + per
    chunk = data[start:end]
    items = [
        {
            "type": "video",
            "data": (
                map_invidious_video(v) if (isinstance(v, dict) and ("type" in v or "videoThumbnails" in v)) else map_piped_video(v)
            ).model_dump(),
        }
        for v in chunk
        if isinstance(v, dict)
    ]
    if DEBUG_UPSTREAM:
        response.headers["X-PT-Source"] = source
        response.headers["X-PT-Total"] = str(len(data))
        response.headers["X-PT-PageCount"] = str(len(items))
    return SearchResponse(items=items, nextPage=(page + 1 if end < len(data) else None))


@app.get("/v1/shorts", response_model=ShortItemsResponse)
async def shorts(response: Response, page: int = 1, region: str = "US"):
    data, source = await _trending_videos(region=region, shorts_only=True)
    per = 20
    start, end = (page - 1) * per, (page - 1) * per + per
    chunk = data[start:end]
    vids = [
        (
            map_invidious_video(v) if (isinstance(v, dict) and ("type" in v or "videoThumbnails" in v)) else map_piped_video(v)
        ).model_dump()
        for v in chunk
        if isinstance(v, dict)
    ]
    if DEBUG_UPSTREAM:
        response.headers["X-PT-Source"] = source
        response.headers["X-PT-Total"] = str(len(data))
        response.headers["X-PT-PageCount"] = str(len(vids))
    return ShortItemsResponse(items=vids, nextPage=(page + 1 if end < len(data) else None))


@app.get("/v1/related", response_model=SearchResponse)
async def related(response: Response, videoId: str = Query(..., min_length=3), page: int = 1, region: str = "US"):
    provider = "none"
    mapped: List[dict] = []
    # Try Invidious first
    try:
        data = await _get_json_with_failover(f"/api/v1/related/{videoId}", {"region": region}, cacheable=True)
        if isinstance(data, list) and data:
            mapped = [{"type": "video", "data": map_invidious_video(d).model_dump()} for d in data if isinstance(d, dict)]
            provider = "invidious"
    except Exception as e:
        _dbg(f"invidious related error: {e}")

    # Always try Piped next if empty (more reliable for related)
    if not mapped:
        try:
            j = await _piped_json_with_failover(f"/api/streams/{videoId}", {})
            rel = j.get("relatedStreams") if isinstance(j, dict) else []
            mapped = [{"type": "video", "data": map_piped_video(d).model_dump()} for d in rel if isinstance(d, dict)]
            if mapped:
                provider = "piped"
        except Exception as e:
            _dbg(f"piped related error: {e}")

    per = 20
    start, end = (page - 1) * per, (page - 1) * per + per
    page_items = mapped[start:end]

    if DEBUG_UPSTREAM:
        response.headers["X-PT-Provider"] = provider
        response.headers["X-PT-Total"] = str(len(mapped))
        response.headers["X-PT-PageCount"] = str(len(page_items))

    return SearchResponse(items=page_items, nextPage=(page + 1 if end < len(mapped) else None))


@app.get("/v1/videoStats")
async def video_stats(ids: str = Query(..., description="comma-separated videoIds")) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    id_list = [x.strip() for x in ids.split(",") if x.strip()]

    async def fetch_one(vid: str) -> Dict[str, int]:
        path = f"/api/v1/videos/{vid}"
        try:
            j = await _get_json_with_failover(path, {}, cacheable=True)
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


@app.get("/v1/comments", response_model=CommentsResponse)
async def comments(
    videoId: str = Query(..., min_length=3),
    continuation: Optional[str] = Query(None),
    region: str = "US",
):
    # Invidious
    try:
        params = {"hl": "en"}
        if continuation:
            params["continuation"] = continuation
        d = await _get_json_with_failover(f"/api/v1/comments/{videoId}", params, cacheable=False)
        if isinstance(d, dict):
            disabled = bool(d.get("disabled") or d.get("commentsDisabled", False))
            raw = d.get("comments") or []
            items = [map_invidious_comment(c).model_dump() for c in raw if isinstance(c, dict)]
            return CommentsResponse(disabled=disabled, items=items, nextPage=d.get("continuation"))
    except Exception:
        pass

    # Piped
    try:
        pp: dict = {}
        if continuation:
            pp["nextpage"] = continuation
        d = await _piped_json_with_failover(f"/api/comments/{videoId}", pp)
        if isinstance(d, dict):
            disabled = bool(d.get("disabled", False))
            raw = d.get("comments") or d.get("items") or []
            items = [map_piped_comment(c).model_dump() for c in raw if isinstance(c, dict)]
            return CommentsResponse(disabled=disabled, items=items, nextPage=d.get("nextpage") or d.get("nextPage"))
        if isinstance(d, list):
            items = [map_piped_comment(c).model_dump() for c in d if isinstance(c, dict)]
            return CommentsResponse(disabled=False, items=items, nextPage=None)
    except Exception:
        pass

    return CommentsResponse(disabled=True, items=[], nextPage=None)


# ------------------ Error handler ------------------
@app.exception_handler(Exception)
async def unhandled_exceptions(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    # Generic 500 with safe detail (no giant tracebacks)
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {str(exc)}"})


# ------------------ Entrypoint ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("primetube_backend_v1_8_0:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)
