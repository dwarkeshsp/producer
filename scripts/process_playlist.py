import asyncio
from pathlib import Path
import sys
import time
from typing import List

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.youtube_utils import get_transcript, get_playlist_video_ids
from utils.content_generator import ContentGenerator, ContentRequest

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLd7-bHaQwnthaNDpZ32TtYONGVk95-fhF"
MAX_CONCURRENT = 3  # Limit concurrent requests
RETRY_DELAY = 65  # Seconds to wait before retrying after rate limit

async def process_video(video_id: str, generator: ContentGenerator, retry_count: int = 0) -> str:
    """Process a single video and return the formatted result."""
    try:
        print(f"Processing video {video_id}...")
        
        # Get transcript
        transcript = get_transcript(video_id)
        if not transcript:
            print(f"No transcript available for {video_id}")
            return ""
            
        # Generate suggestions
        request = ContentRequest("titles_and_thumbnails", temperature=0.7)
        result = await generator.generate_content(request, transcript)
        return f"Video ID: {video_id}\n{result}\n{'='*50}\n"
        
    except Exception as e:
        if "rate_limit_error" in str(e) and retry_count < 3:
            print(f"Rate limit hit for {video_id}, waiting {RETRY_DELAY}s before retry {retry_count + 1}")
            await asyncio.sleep(RETRY_DELAY)
            return await process_video(video_id, generator, retry_count + 1)
        print(f"Error processing {video_id}: {e}")
        return ""

async def process_batch(video_ids: List[str], generator: ContentGenerator) -> List[str]:
    """Process a batch of videos with rate limiting."""
    tasks = [process_video(video_id, generator) for video_id in video_ids]
    return await asyncio.gather(*tasks)

async def process_playlist():
    """Process all videos in playlist with batching."""
    generator = ContentGenerator()
    output_file = Path("output/playlist-titles-thumbnails.txt")
    
    # Get videos from playlist
    print("Getting videos from playlist...")
    video_ids = get_playlist_video_ids(PLAYLIST_URL)
    print(f"Found {len(video_ids)} videos")
    
    # Process videos in batches
    results = []
    for i in range(0, len(video_ids), MAX_CONCURRENT):
        batch = video_ids[i:i + MAX_CONCURRENT]
        print(f"\nProcessing batch {i//MAX_CONCURRENT + 1}")
        batch_results = await process_batch(batch, generator)
        results.extend(batch_results)
        
        # Add delay between batches to avoid rate limits
        if i + MAX_CONCURRENT < len(video_ids):
            delay = 5  # Short delay between successful batches
            print(f"Waiting {delay}s before next batch...")
            await asyncio.sleep(delay)
    
    # Filter out empty results and save
    results = [r for r in results if r]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(results))
    print(f"\nResults written to {output_file}")

if __name__ == "__main__":
    asyncio.run(process_playlist())