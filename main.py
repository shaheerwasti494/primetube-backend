# main.py
import os
from typing import Dict, List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import httpx
from dotenv import load_dotenv

load_dotenv()

INVIDIOUS_BASE = os.getenv("INVIDIOUS_BASE", "https://yewtu.be")
PIPED_BASE = os.getenv("PIPED_BASE", "https://piped.video")  # just for health info
CORS = os.getenv("CORS_ORIGINS", "*")

app = FastAPI(title="PrimeTube Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS] if CORS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- models --------
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
    items: List[dict]  # union {type:"video"/"channel", data: VideoItem/ChannelItem}
    nextPage: Optional[int] = None

class SuggestResponse(BaseModel):
    suggestions: List[str]

class ShortItemsResponse(BaseModel):
    items: List[VideoItem]
    nextPage: Optional[int] = None


# -------- helpers --------
def thumbify(arr) -> List[Thumbnail]:
    if not isinstance(arr, list):
        return []
    out: List[Thumbnail] = []
    for t in arr:
        out.append(Thumbnail(url=t.get("url"), width=t.get("width"), height=t.get("height")))
    return out

def map_invidious_video(v: dict) -> VideoItem:
    return VideoItem(
        id=v.get("videoId") or v.get("id", ""),
        title=v.get("title", ""),
        channelName=v.get("author"),
        channelId=v.get("authorId"),
        # viewCount may be int or text on some instances — coerce carefully
        viewCount=(int(v.get("viewCount")) if isinstance(v.get("viewCount"), int)
                   else int(v.get("viewCount") or 0)),
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

async def _get_json(url: str, params: dict | None = None) -> list | dict:
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        # Return a FastAPI HTTPException with upstream status
        raise HTTPException(status_code=e.response.status_code,
                            detail=f"Upstream error {e.response.status_code} for {url}") from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream fetch failed for {url}: {e}") from e


# -------- endpoints --------
@app.get("/health")
async def health():
    return {"ok": True, "invidious": INVIDIOUS_BASE, "piped": PIPED_BASE}

@app.get("/v1/suggest", response_model=SuggestResponse)
async def suggest(q: str = Query(..., min_length=1)):
    base = q.strip()
    # super-light suggester to avoid hitting external services
    variants = list({base, f"{base} news", f"{base} tutorial", f"{base} live", f"{base} shorts"})
    return SuggestResponse(suggestions=variants[:8])

@app.get("/v1/search", response_model=SearchResponse)
async def search(q: str, page: int = 1):
    url = f"{INVIDIOUS_BASE}/api/v1/search"
    params = {"q": q, "page": page, "type": "all"}
    data = await _get_json(url, params)
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
    url = f"{INVIDIOUS_BASE}/api/v1/search"
    params = {"q": q, "page": page, "type": "channel"}
    data = await _get_json(url, params)
    if not isinstance(data, list):
        data = []
    items = [{"type": "channel", "data": map_invidious_channel(it).model_dump()} for it in data]
    return SearchResponse(items=items, nextPage=(page + 1 if items else None))

@app.get("/v1/trending", response_model=SearchResponse)
async def trending(page: int = 1, region: str = "US"):
    # Invidious trending supports type selection; default “Videos”.
    url = f"{INVIDIOUS_BASE}/api/v1/trending"
    params = {"region": region}
    data = await _get_json(url, params)
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
    Use Invidious trending with type=Shorts for consistent, quota-free Shorts.
    Then emulate paging by slicing.
    """
    url = f"{INVIDIOUS_BASE}/api/v1/trending"
    params = {"region": region, "type": "Shorts"}
    data = await _get_json(url, params)
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
    Returns a map: { videoId: { views, likes, comments } }
    Invidious exposes /api/v1/videos/:id (viewCount, likeCount, commentCount* if available).
    """
    out: Dict[str, Dict[str, int]] = {}
    id_list = [x.strip() for x in ids.split(",") if x.strip()]
    async with httpx.AsyncClient(timeout=20.0) as client:
        for vid in id_list:
            try:
                r = await client.get(f"{INVIDIOUS_BASE}/api/v1/videos/{vid}")
                r.raise_for_status()
                j = r.json()
                def _to_int(x):  # handle strings/None
                    try:
                        return int(x)
                    except Exception:
                        return 0
                out[vid] = {
                    "views": _to_int(j.get("viewCount")),
                    "likes": _to_int(j.get("likeCount")),
                    "comments": _to_int(j.get("commentCount")),
                }
            except Exception:
                out[vid] = {"views": 0, "likes": 0, "comments": 0}
    return out

# ---- error handlers (optional nice JSON for unexpected errors) ----
@app.exception_handler(Exception)
async def unhandled_exceptions(request, exc: Exception):
    # Let HTTPException bubble as-is
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    # Fallback
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {str(exc)}"})
