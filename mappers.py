# mappers.py
from typing import Any, Dict, List
from models import (
    ItemDto, VideoData, ChannelData,
    YouTubeShortItem, IdWrap, Snippet, Thumbnails, Thumb,
    ChannelInfoItem, ChannelSnippet, ChannelThumbnails, ChannelThHigh, ChannelStatistics
)

def fmt_views(v: Any) -> int:
    try:
        if isinstance(v, str) and v.endswith("K"):
            return int(float(v[:-1]) * 1_000)
        if isinstance(v, str) and v.endswith("M"):
            return int(float(v[:-1]) * 1_000_000)
        if isinstance(v, str) and v.endswith("B"):
            return int(float(v[:-1]) * 1_000_000_000)
        return int(v)
    except Exception:
        return 0

def map_search_items_to_itemdto(raw: List[Dict[str, Any]]) -> List[ItemDto]:
    items: List[ItemDto] = []
    for it in raw:
        t = it.get("type")
        if t == "video":
            v = VideoData(
                id = it.get("videoId") or it.get("videoId", ""),
                title = it.get("title") or "",
                channelName = it.get("author") or it.get("authorId"),
                channelId = it.get("authorId"),
                viewCount = fmt_views(it.get("viewCount") or it.get("viewCountText") or 0),
                publishedText = it.get("publishedText"),
                thumbnail = (it.get("videoThumbnails") or [{"url": ""}])[-1].get("url")
            )
            items.append(ItemDto(type="video", data=v.model_dump()))
        elif t == "channel":
            c = ChannelData(
                id = it.get("channelId") or it.get("authorId") or it.get("ucid") or "",
                title = it.get("author") or it.get("title"),
                avatar = (it.get("authorThumbnails") or [{"url": ""}])[-1].get("url")
            )
            items.append(ItemDto(type="channel", data=c.model_dump()))
    return items

def map_trending_to_itemdto(raw: List[Dict[str, Any]]) -> List[ItemDto]:
    out: List[ItemDto] = []
    for it in raw:
        v = VideoData(
            id = it.get("videoId", ""),
            title = it.get("title") or "",
            channelName = it.get("author"),
            channelId = it.get("authorId"),
            viewCount = fmt_views(it.get("viewCount") or it.get("viewCountText") or 0),
            publishedText = it.get("publishedText"),
            thumbnail = (it.get("videoThumbnails") or [{"url": ""}])[-1].get("url")
        )
        out.append(ItemDto(type="video", data=v.model_dump()))
    return out

def map_channel_info(raw: Dict[str, Any]) -> ChannelInfoItem:
    # Invidious channel json contains: author, authorId, authorThumbnails, subCount, etc.
    thumb_url = (raw.get("authorThumbnails") or [{"url": ""}])[-1].get("url", "")
    return ChannelInfoItem(
        id = raw.get("authorId") or raw.get("ucid") or "",
        snippet = ChannelSnippet(
            title = raw.get("author"),
            thumbnails = ChannelThumbnails(high=ChannelThHigh(url=thumb_url))
        ),
        statistics = ChannelStatistics(
            subscriberCount = str(raw.get("subCount") or 0)
        )
    )

def map_videos_to_shorts(raw: List[Dict[str, Any]]) -> List[YouTubeShortItem]:
    out: List[YouTubeShortItem] = []
    for it in raw:
        if it.get("type") != "video":
            continue
        # Filter to short videos (<= 60s) if lengthSeconds is present
        seconds = it.get("lengthSeconds")
        try:
            if seconds is not None and int(seconds) > 60:
                continue
        except Exception:
            pass

        thumb = (it.get("videoThumbnails") or [{"url": ""}])[-1].get("url", "")
        out.append(
            YouTubeShortItem(
                id=IdWrap(videoId=it.get("videoId", "")),
                snippet=Snippet(
                    title=it.get("title"),
                    description=it.get("description"),
                    channelTitle=it.get("author"),
                    channelId=it.get("authorId"),
                    publishedAt=it.get("publishedText"),
                    thumbnails=Thumbnails(high=Thumb(url=thumb))
                )
            )
        )
    return out
