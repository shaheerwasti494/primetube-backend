# models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

# ---------- client DTOs (match Android) ----------

class VideoData(BaseModel):
    id: str
    title: str
    channelName: Optional[str] = None
    channelId: Optional[str] = None
    viewCount: Optional[int] = 0
    publishedText: Optional[str] = None
    thumbnail: Optional[str] = None  # not used by client but handy

class ChannelData(BaseModel):
    id: str
    title: Optional[str] = None
    avatar: Optional[str] = None

class ItemDto(BaseModel):
    type: Literal["video", "channel"]
    data: dict  # polymorphic payload (VideoData or ChannelData). Use dict to keep Moshi happy.

class SuggestResponse(BaseModel):
    suggestions: List[str] = []

class BackendSearchResponse(BaseModel):
    items: List[ItemDto] = []
    nextPage: Optional[int] = None

# ---- Shorts (approximate YouTube v3 shape minimal subset your app reads) ----

class Thumb(BaseModel):
    url: str

class Thumbnails(BaseModel):
    high: Thumb

class Snippet(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    channelTitle: Optional[str] = None
    channelId: Optional[str] = None
    publishedAt: Optional[str] = None
    thumbnails: Optional[Thumbnails] = None

class IdWrap(BaseModel):
    videoId: str

class YouTubeShortItem(BaseModel):
    id: IdWrap
    snippet: Snippet

class ShortItemsResponse(BaseModel):
    items: List[YouTubeShortItem] = []
    nextPage: Optional[int] = None

# ---- ChannelInfo minimal (YouTube-ish) ----

class ChannelThHigh(BaseModel):
    url: str

class ChannelThumbnails(BaseModel):
    high: ChannelThHigh

class ChannelSnippet(BaseModel):
    title: Optional[str] = None
    thumbnails: Optional[ChannelThumbnails] = None

class ChannelStatistics(BaseModel):
    subscriberCount: Optional[str] = None

class BrandingSettings(BaseModel):
    # Placeholder – extend if you need.
    pass

class ChannelInfoItem(BaseModel):
    id: str
    snippet: Optional[ChannelSnippet] = None
    statistics: Optional[ChannelStatistics] = None
    brandingSettings: Optional[BrandingSettings] = None

class ChannelInfoResponse(BaseModel):
    item: Optional[ChannelInfoItem] = None
