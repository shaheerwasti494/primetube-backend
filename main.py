"""
PrimeTube Backend — v2.0 (FastAPI)

Goals
- Simplified, readable, and maintainable single-file backend
- Shared httpx client with connection pooling
- Clean failover across Invidious & Piped with one helper
- Self-healing instance discovery (startup + periodic)
- Small in‑memory TTL cache for non‑auth GETs
- Useful debugging headers behind DEBUG_UPSTREAM
- Same external API surface as v1.9 (drop-in)

Run locally
  uvicorn primetube_backend:app --host 0.0.0.0 --port 8080 --reload

Deploy (Cloud Run)
  gcloud builds submit --tag gcr.io/PROJECT/primetube-backend:v2.0
  gcloud run deploy primetube-backend \
    --image gcr.io/PROJECT/primetube-backend:v2.0 \
    --allow-unauthenticated --region=us-central1 \
    --set-env-vars=DEBUG_UPSTREAM=1
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import random
import time
from typing import Dict, Iterable, List, Optional, Tuple, Union

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ------------ Safe env helpers (tolerate empty/invalid) ------------
def _env_str(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v in (None, "", "None", "null"):
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v in (None, "", "None", "null"):
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v in (None, ""):
        return default
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# ------------------ Config ------------------
class Settings:
    # Static fallbacks (kept short; discovery will override when enabled)
    FALLBACK_INVIDIOUS = [
        "https://invidious.privacydev.net",
        "https://invidious.fdn.fr",
        "https://yewtu.be",
        "https://iv.nboeck.de",
    ]
    FALLBACK_PIPED = [
        "https://piped.video",
        "https://piped.privacydev.net",
        "https://piped.mha.fi",
    ]

    INVIDIOUS_POOL = [
        b.strip().rstrip("/")
        for b in os.getenv("INVIDIOUS_POOL", ",".join(FALLBACK_INVIDIOUS)).split(",")
        if b.strip()
    ]
    PIPED_POOL = [
        b.strip().rstrip("/")
        for b in os.getenv("PIPED_POOL", ",".join(FALLBACK_PIPED)).split(",")
        if b.strip()
    ]

    CORS_ORIGINS = _env_str("CORS_ORIGINS", "*")
    CACHE_TTL_SECONDS = _env_int("CACHE_TTL_SECONDS", 300)
    DEBUG_UPSTREAM = _env_bool("DEBUG_UPSTREAM", False)

    USER_AGENT = _env_str(
        "USER_AGENT",
        (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 "
            "PrimeTubeBackend/2.0 httpx"
        ),
    )

    # Discovery
    DISCOVERY_ENABLED = _env_bool("DISCOVERY_ENABLED", True)
    DISCOVERY_INTERVAL_SECONDS = _env_int("DISCOVERY_INTERVAL_SECONDS", 1800)
    INVIDIOUS_DISCOVERY_MAX = _env_int("INVIDIOUS_DISCOVERY_MAX", 8)
    PIPED_DISCOVERY_MAX = _env_int("PIPED_DISCOVERY_MAX", 5)

    INVIDIOUS_DISCOVERY_URL = os.getenv(
        "INVIDIOUS_DISCOVERY_URL", "https://api.invidious.io/instances.json"
    )
    PIPED_DISCOVERY_URLS = [
        os.getenv("PIPED_DISCOVERY_URL1", "https://piped.video/api/instances"),
        os.getenv("PIPED_DISCOVERY_URL2", "https://piped-instances.kavin.rocks/instances.json"),
    ]

    HTTPX_TIMEOUT_SECONDS = _env_float("HTTPX_TIMEOUT_SECONDS", 25.0)


S = Settings()

# ------------------ App ------------------
app = FastAPI(title="PrimeTube Backend", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[S.CORS_ORIGINS] if S.CORS_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared httpx client in app.state
async def _new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=S.HTTPX_TIMEOUT_SECONDS,
        headers={"User-Agent": S.USER_AGENT, "Accept": "application/json, */*"},
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    )


# Pools + discovery state
INVIDIOUS_POOL: List[str] = S.INVIDIOUS_POOL or ["https://yewtu.be"]
PIPED_POOL: List[str] = S.PIPED_POOL or ["https://piped.video"]
_inv_cycle = itertools.cycle(INVIDIOUS_POOL)
_piped_cycle = itertools.cycle(PIPED_POOL)
_last_discovery_ts: Optional[float] = None
_last_discovery_err: Optional[str] = None


def _dbg(msg: str):
    if S.DEBUG_UPSTREAM:
        print(f"[UPSTREAM] {msg}")


# ------------------ Tiny TTL cache ------------------
_cache: Dict[str, Tuple[float, object]] = {}


def _cache_key(path: str, params: dict) -> str:
    return f"{path}?{json.dumps(params, sort_keys=True, separators=(",", ":"))}"


def cache_get(path: str, params: dict):
    if S.CACHE_TTL_SECONDS <= 0:
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
    if S.CACHE_TTL_SECONDS <= 0:
        return
    k = _cache_key(path, params)
    _cache[k] = (time.time() + (ttl if ttl is not None else S.CACHE_TTL_SECONDS), data)


def _cacheable_nonempty(data: object) -> bool:
    if isinstance(data, list):
        return bool(data)
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return bool(data["items"])
    return False


# ------------------ Discovery ------------------
async def _discover_invidious(client: httpx.AsyncClient) -> List[str]:
    try:
        r = await client.get(S.INVIDIOUS_DISCOVERY_URL)
        r.raise_for_status()
        data = r.json()
        hosts: List[str] = []
        if isinstance(data, list):
            for row in data:
                if not (isinstance(row, list) and len(row) >= 2):
                    continue
                base = str(row[0]).rstrip("/")
                meta = row[1] if isinstance(row[1], dict) else {}
                if meta.get("api") is True:
                    hosts.append(base)
        random.shuffle(hosts)
        return hosts[: S.INVIDIOUS_DISCOVERY_MAX]
    except Exception as e:
        _dbg(f"invidious discovery failed: {e}")
        return []


async def _discover_piped(client: httpx.AsyncClient) -> List[str]:
    all_hosts: List[str] = []
    for url in S.PIPED_DISCOVERY_URLS:
        try:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
            hosts: List[str] = []
            if isinstance(data, list):
                for it in data:
                    if isinstance(it, str):
                        hosts.append(it)
                    elif isinstance(it, dict):
                        cand = it.get("api_url") or it.get("apiUrl") or it.get("url") or it.get("frontend")
                        if isinstance(cand, str):
                            hosts.append(cand)
            elif isinstance(data, dict):
                arr = data.get("instances") or data.get("items") or []
                if isinstance(arr, list):
                    for s in arr:
                        if isinstance(s, str):
                            hosts.append(s)
            for h in hosts:
                base = h.split("/api")[0].rstrip("/")
                all_hosts.append(base)
        except Exception as e:
            _dbg(f"piped discovery from {url} failed: {e}")
    random.shuffle(all_hosts)
    # Deduplicate preserving order
    uniq = list(dict.fromkeys(all_hosts))
    return uniq[: S.PIPED_DISCOVERY_MAX]


async def _apply_discovery(new_inv: List[str], new_piped: List[str]):
    global INVIDIOUS_POOL, PIPED_POOL, _inv_cycle, _piped_cycle, _last_discovery_ts
    changed = False
    if new_inv:
        INVIDIOUS_POOL = new_inv
        _inv_cycle = itertools.cycle(INVIDIOUS_POOL)
        changed = True
    if new_piped:
        PIPED_POOL = new_piped
        _piped_cycle = itertools.cycle(PIPED_POOL)
        changed = True
    _last_discovery_ts = time.time()
    if changed:
        _dbg(f"discovery applied inv={INVIDIOUS_POOL} piped={PIPED_POOL}")


async def _refresh_instances(client: httpx.AsyncClient):
    global _last_discovery_err
    if not S.DISCOVERY_ENABLED:
        return
    inv = await _discover_invidious(client)
    pip = await _discover_piped(client)
    if inv or pip:
        await _apply_discovery(inv or INVIDIOUS_POOL, pip or PIPED_POOL)
        _last_discovery_err = None
    else:
        _last_discovery_err = "discovery yielded no instances"


# ------------------ Startup/Shutdown ------------------
@app.on_event("startup")
async def _on_startup():
    app.state.client = await _new_client()
    try:
        await _refresh_instances(app.state.client)
    except Exception as e:
        _dbg(f"startup discovery error: {e}")

    if S.DISCOVERY_ENABLED and S.DISCOVERY_INTERVAL_SECONDS > 0:
        async def _loop():
            while True:
                try:
                    await asyncio.sleep(S.DISCOVERY_INTERVAL_SECONDS)
                    await _refresh_instances(app.state.client)
                except Exception as e:
                    _dbg(f"periodic discovery error: {e}")

        asyncio.create_task(_loop())


@app.on_event("shutdown")
async def _on_shutdown():
    try:
        await app.state.client.aclose()
    except Exception:
        pass


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


# ------------------ Helpers ------------------
async def _async_backoff(status: Optional[int] = None):
    await asyncio.sleep(0.4 if status == 429 else 0.2)


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


# Invidious mappers

def _thumbify(arr) -> List[Thumbnail]:
    if not isinstance(arr, list):
        return []
    out: List[Thumbnail] = []
    for t in arr:
        if not isinstance(t, dict):
            continue
        url = t.get("url")
        if isinstance(url, str) and url.startswith("/vi/"):
            url = f"https://i.ytimg.com{url}"
        out.append(Thumbnail(url=url, width=t.get("width"), height=t.get("height")))
    return out


def map_invidious_video(v: dict) -> VideoItem:
    vid = v.get("videoId") or v.get("id", "")
    try:
        vc_int = int(v.get("viewCount") or 0)
    except Exception:
        vc_int = 0
    return VideoItem(
        id=vid,
        title=v.get("title", ""),
        channelName=v.get("author"),
        channelId=v.get("authorId"),
        viewCount=vc_int,
        publishedText=v.get("publishedText"),
        thumbnails=_thumbify(v.get("videoThumbnails") or v.get("thumbnails") or []),
    )


def map_invidious_channel(c: dict) -> ChannelItem:
    thumbs = c.get("authorThumbnails") or c.get("thumbnails") or []
    avatar = None
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


# Piped mappers

def _piped_extract_id_from_url(url: str) -> str:
    try:
        from urllib.parse import parse_qs, urlparse

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


# ------------------ Unified upstream fetchers ------------------
async def _fetch_json(client: httpx.AsyncClient, url: str, params: dict) -> object:
    r = await client.get(url, params=params)
    r.raise_for_status()
    try:
        return r.json()
    except Exception as e:
        ct = r.headers.get("content-type", "")
        peek = r.text[:200]
        raise RuntimeError(f"Non-JSON from {url} ct={ct}: {peek}") from e


async def _get_with_failover(
    provider: str, path: str, params: dict, *,
    cacheable: bool = True,
    require_nonempty_list: bool = False,
    require_items_key_nonempty: bool = False,
) -> object:
    """Generic failover over INVIDIOUS_POOL or PIPED_POOL."""
    cached = cache_get(f"{provider}:{path}", params) if cacheable else None
    if cached is not None:
        return cached

    bases = (INVIDIOUS_POOL if provider == "invidious" else PIPED_POOL)[:]
    last_error: Optional[Exception] = None
    tried: set[str] = set()

    for _ in range(len(bases)):
        base = bases[_ % len(bases)]
        if base in tried:
            continue
        tried.add(base)
        url = f"{base}{path}"
        try:
            data = await _fetch_json(app.state.client, url, params)
            if require_nonempty_list and isinstance(data, list) and not data:
                continue
            if require_items_key_nonempty and isinstance(data, dict):
                items = data.get("items")
                if isinstance(items, list) and not items:
                    continue
            if cacheable and _cacheable_nonempty(data):
                cache_set(f"{provider}:{path}", params, data)
            return data
        except httpx.HTTPStatusError as e:
            st = e.response.status_code
            last_error = e
            if st == 429 or 500 <= st < 600:
                await _async_backoff(st)
                continue
            raise HTTPException(status_code=st, detail=f"Upstream {provider} error {st} at {url}") from e
        except Exception as e:
            last_error = e
            _dbg(f"{provider} fail {url}: {e}")
            await _async_backoff()
            continue

    if cacheable:
        stale = cache_get(f"{provider}:{path}", params)
        if stale is not None:
            return stale

    raise HTTPException(status_code=502, detail=f"{provider} mirrors failed for {path}: {last_error}")


# ------------------ Feature helpers ------------------
async def _trending_videos(region: str, shorts_only: bool) -> Tuple[List[dict], str]:
    inv_params = {"region": region}
    if shorts_only:
        inv_params["type"] = "Shorts"
    # Invidious trending
    try:
        data = await _get_with_failover(
            "invidious", "/api/v1/trending", inv_params, cacheable=True, require_nonempty_list=True
        )
        if isinstance(data, list) and data:
            return data, "invidious_trending"
    except Exception as e:
        _dbg(f"invidious /trending error: {e}")

    # Piped trending
    try:
        data = await _get_with_failover("piped", "/api/trending", {"region": region})
        if isinstance(data, list) and data:
            return data, "piped_trending"
    except Exception as e:
        _dbg(f"piped /trending error: {e}")

    # Invidious popular fallback
    try:
        data = await _get_with_failover("invidious", "/api/v1/popular", {}, cacheable=True, require_nonempty_list=True)
        if isinstance(data, list) and data:
            return data, "invidious_popular"
    except Exception as e:
        _dbg(f"invidious /popular error: {e}")

    # Shorts-specific searches
    if shorts_only:
        try:
            inv_search = await _get_with_failover(
                "invidious",
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
            piped_search = await _get_with_failover(
                "piped",
                "/api/search",
                {"q": "#shorts", "filter": "videos", "sort": "trending", "region": region},
            )
            items = piped_search.get("items") if isinstance(piped_search, dict) else None
            if items:
                return items, "piped_search_trending"
        except Exception as e:
            _dbg(f"piped search trending error: {e}")

    return [], "none"


async def _unified_search(q: str, page: int, kind: str, region: str) -> Tuple[List[dict], str]:
    merged: List[dict] = []
    provider = "none"

    # Invidious fanout (with & without region)
    for with_region in (True, False):
        base = {"q": q, "page": page}
        if with_region:
            base["region"] = region
        for t in (None, "video", "channel"):
            params = dict(base)
            if t:
                params["type"] = t
            try:
                inv = await _get_with_failover("invidious", "/api/v1/search", params, cacheable=True)
                if isinstance(inv, list):
                    merged.extend(inv)
                    provider = "invidious"
            except Exception as e:
                _dbg(f"inv search err {e}")

    merged = _merge_unique(merged)
    if merged:
        return merged, provider

    # Piped fallback
    try:
        piped_items: List[dict] = []
        if kind in ("all", "video"):
            pv = await _get_with_failover(
                "piped",
                "/api/search",
                {"q": q, "filter": "videos", "sort": "relevance", "page": page, "region": region},
            )
            vi = pv.get("items") if isinstance(pv, dict) else None
            if vi:
                piped_items.extend(vi)
                provider = "piped"
        if kind in ("all", "channel"):
            pc = await _get_with_failover(
                "piped",
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
    info: Dict[str, Union[bool, int, List[str], str, List[dict], float, None]] = {
        "ok": True,
        "active_pool": INVIDIOUS_POOL,
        "piped_pool": PIPED_POOL,
        "cache_ttl": S.CACHE_TTL_SECONDS,
        "discovery_enabled": S.DISCOVERY_ENABLED,
        "last_discovery_ts": _last_discovery_ts,
        "last_discovery_err": _last_discovery_err,
    }
    if deep:
        checks = []
        try:
            d = await _get_with_failover("invidious", "/api/v1/trending", {"region": "US"}, cacheable=False)
            checks.append({"invidious_trending_len": len(d) if isinstance(d, list) else None})
        except Exception as e:
            checks.append({"invidious_trending_error": str(e)})
        try:
            d = await _get_with_failover("piped", "/api/trending", {"region": "US"})
            checks.append({"piped_trending_len": len(d) if isinstance(d, list) else None})
        except Exception as e:
            checks.append({"piped_trending_error": str(e)})
        info["checks"] = checks
    return info


@app.get("/health/instances")
async def health_instances():
    return {
        "invidious_pool": INVIDIOUS_POOL,
        "piped_pool": PIPED_POOL,
        "discovery_enabled": S.DISCOVERY_ENABLED,
        "last_discovery_ts": _last_discovery_ts,
        "last_discovery_err": _last_discovery_err,
        "limits": {
            "invidious_max": S.INVIDIOUS_DISCOVERY_MAX,
            "piped_max": S.PIPED_DISCOVERY_MAX,
        },
        "discovery_sources": {
            "invidious": S.INVIDIOUS_DISCOVERY_URL,
            "piped": S.PIPED_DISCOVERY_URLS,
        },
    }


@app.get("/v1/suggest", response_model=SuggestResponse)
async def suggest(q: str = Query(..., min_length=1)):
    base = q.strip()
    variants = list({base, f"{base} news", f"{base} tutorial", f"{base} live", f"{base} shorts"})
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
        # Piped-ish detection
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

    if S.DEBUG_UPSTREAM:
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
            ch = map_invidious_channel(it).model_dump() or map_piped_channel(it).model_dump()
            items.append({"type": "channel", "data": ch})
        elif any(k in it for k in ("authorId", "author", "authorThumbnails", "thumbnails")):
            items.append({"type": "channel", "data": map_invidious_channel(it).model_dump()})
        else:
            items.append({"type": "channel", "data": map_piped_channel(it).model_dump()})

    if S.DEBUG_UPSTREAM:
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
                map_invidious_video(v)
                if (isinstance(v, dict) and ("type" in v or "videoThumbnails" in v))
                else map_piped_video(v)
            ).model_dump(),
        }
        for v in chunk
        if isinstance(v, dict)
    ]
    if S.DEBUG_UPSTREAM:
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
            map_invidious_video(v)
            if (isinstance(v, dict) and ("type" in v or "videoThumbnails" in v))
            else map_piped_video(v)
        ).model_dump()
        for v in chunk
        if isinstance(v, dict)
    ]
    if S.DEBUG_UPSTREAM:
        response.headers["X-PT-Source"] = source
        response.headers["X-PT-Total"] = str(len(data))
        response.headers["X-PT-PageCount"] = str(len(vids))
    return ShortItemsResponse(items=vids, nextPage=(page + 1 if end < len(data) else None))


@app.get("/v1/related", response_model=SearchResponse)
async def related(response: Response, videoId: str = Query(..., min_length=3), page: int = 1, region: str = "US"):
    provider = "none"
    mapped: List[dict] = []
    # Invidious first
    try:
        data = await _get_with_failover("invidious", f"/api/v1/related/{videoId}", {"region": region}, cacheable=True)
        if isinstance(data, list) and data:
            mapped = [{"type": "video", "data": map_invidious_video(d).model_dump()} for d in data if isinstance(d, dict)]
            provider = "invidious"
    except Exception as e:
        _dbg(f"invidious related error: {e}")

    if not mapped:
        try:
            j = await _get_with_failover("piped", f"/api/streams/{videoId}", {})
            rel = j.get("relatedStreams") if isinstance(j, dict) else []
            mapped = [{"type": "video", "data": map_piped_video(d).model_dump()} for d in rel if isinstance(d, dict)]
            if mapped:
                provider = "piped"
        except Exception as e:
            _dbg(f"piped related error: {e}")

    per = 20
    start, end = (page - 1) * per, (page - 1) * per + per
    page_items = mapped[start:end]

    if S.DEBUG_UPSTREAM:
        response.headers["X-PT-Provider"] = provider
        response.headers["X-PT-Total"] = str(len(mapped))
        response.headers["X-PT-PageCount"] = str(len(page_items))

    return SearchResponse(items=page_items, nextPage=(page + 1 if end < len(mapped) else None))


@app.get("/v1/videoStats")
async def video_stats(ids: str = Query(..., description="comma-separated videoIds")) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    id_list = [x.strip() for x in ids.split(",") if x.strip()]

    async def fetch_one(vid: str) -> Dict[str, int]:
        try:
            j = await _get_with_failover("invidious", f"/api/v1/videos/{vid}", {}, cacheable=True)

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
        d = await _get_with_failover("invidious", f"/api/v1/comments/{videoId}", params, cacheable=False)
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
        d = await _get_with_failover("piped", f"/api/comments/{videoId}", pp)
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
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {str(exc)}"})


# ------------------ Entrypoint ------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("primetube_backend:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)
