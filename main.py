import os
from typing import Dict, List, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

INVIDIOUS_BASE = os.getenv("INVIDIOUS_BASE", "https://yewtu.be")
CORS = os.getenv("CORS_ORIGINS", "*")

app = FastAPI(title="PrimeTube Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS] if CORS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- models (Kotlin will mirror these) --------
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
    out = []
    for t in arr:
        out.append(Thumbnail(url=t.get("url"), width=t.get("width"), height=t.get("height")))
    return out

def map_invidious_video(v: dict) -> VideoItem:
    return VideoItem(
        id=v.get("videoId") or v.get("videoIdShort") or v.get("videoId") or v.get("id", ""),
        title=v.get("title", ""),
        channelName=v.get("author"),
        channelId=v.get("authorId"),
        viewCount=v.get("viewCount") or v.get("viewCountText"),
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

# -------- endpoints --------
@app.get("/health")
async def health():
    return {"ok": True, "invidious": INVIDIOUS_BASE}

@app.get("/v1/suggest", response_model=SuggestResponse)
async def suggest(q: str = Query(..., min_length=1)):
    # invidious doesn’t provide suggestions; use a simple heuristic or return last words
    # here we just echo a few variants to keep quota-free
    base = q.strip()
    variants = list({base, f"{base} news", f"{base} tutorial", f"{base} live", f"{base} shorts"})
    return SuggestResponse(suggestions=variants[:8])

@app.get("/v1/search", response_model=SearchResponse)
async def search(q: str, page: int = 1):
    url = f"{INVIDIOUS_BASE}/api/v1/search"
    params = {"q": q, "page": page, "type": "all"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

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
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    items = [{"type": "channel", "data": map_invidious_channel(it).model_dump()} for it in data]
    return SearchResponse(items=items, nextPage=(page + 1 if items else None))

@app.get("/v1/trending", response_model=SearchResponse)
async def trending(page: int = 1, region: str = "US"):
    # Invidious trending has no paging, so emulate by slicing
    url = f"{INVIDIOUS_BASE}/api/v1/trending"
    params = {"region": region}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    per_page = 20
    start = (page - 1) * per_page
    end = start + per_page
    chunk = data[start:end]
    items = [{"type": "video", "data": map_invidious_video(v).model_dump()} for v in chunk]
    next_page = page + 1 if end < len(data) else None
    return SearchResponse(items=items, nextPage=next_page)

@app.get("/v1/shorts", response_model=ShortItemsResponse)
async def shorts(q: str = "#shorts", page: int = 1):
    url = f"{INVIDIOUS_BASE}/api/v1/search"
    params = {"q": q, "page": page, "type": "video", "shorts": "true"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    vids = [map_invidious_video(v).model_dump() for v in data if v.get("type") == "video"]
    next_page = page + 1 if len(vids) > 0 else None
    return ShortItemsResponse(items=vids, nextPage=next_page)

@app.get("/v1/videoStats")
async def video_stats(ids: str = Query(..., description="comma-separated videoIds")) -> Dict[str, Dict[str, int]]:
    """
    Returns a map: { videoId: { views, likes, comments } }
    Invidious exposes /api/v1/videos/:id (likes, viewCount, commentCount* if available).
    """
    out: Dict[str, Dict[str, int]] = {}
    id_list = [x.strip() for x in ids.split(",") if x.strip()]
    async with httpx.AsyncClient(timeout=20.0) as client:
        for vid in id_list:
            try:
                r = await client.get(f"{INVIDIOUS_BASE}/api/v1/videos/{vid}")
                r.raise_for_status()
                j = r.json()
                out[vid] = {
                    "views": int(j.get("viewCount") or 0),
                    "likes": int(j.get("likeCount") or 0),
                    "comments": int(j.get("commentCount") or 0),
                }
            except Exception:
                out[vid] = {"views": 0, "likes": 0, "comments": 0}
    return out
