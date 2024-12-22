from youtube_transcript_api import YouTubeTranscriptApi
from pytube import Playlist
import re
from typing import Optional, List

def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats."""
    match = re.search(
        r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([A-Za-z0-9_-]+)",
        url
    )
    return match.group(1) if match else None

def get_transcript(video_id: str) -> str:
    """Get transcript from YouTube video ID."""
    try:
        transcript = YouTubeTranscriptApi.list_transcripts(video_id).find_transcript(["en"])
        return " ".join(entry["text"] for entry in transcript.fetch())
    except Exception as e:
        print(f"Error fetching transcript for {video_id}: {str(e)}")
        return ""

def get_playlist_video_ids(playlist_url: str) -> List[str]:
    """Get all video IDs from a YouTube playlist."""
    playlist = Playlist(playlist_url)
    return [url.split("watch?v=")[1] for url in playlist.video_urls] 