import argparse
import assemblyai as aai
from google import generativeai
import os
from pydub import AudioSegment
import concurrent.futures
import io
import time
import asyncio
from typing import List, Tuple
import json
import hashlib
from pathlib import Path

# Suppress gRPC shutdown warnings
os.environ["GRPC_PYTHON_LOG_LEVEL"] = "error"

# Initialize API clients
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

aai.settings.api_key = ASSEMBLYAI_API_KEY
generativeai.configure(api_key=GOOGLE_API_KEY)
model = generativeai.GenerativeModel("gemini-exp-1206")

# Define the prompt template
prompt = """You are an expert transcript editor. Your task is to enhance this transcript for maximum readability while maintaining the core message.

IMPORTANT: Respond ONLY with the enhanced transcript. Do not include any explanations, headers, or phrases like "Here is the transcript."

Note: Below you'll find an auto-generated transcript that may help with speaker identification, but focus on creating your own high-quality transcript from the audio.

Please:
1. Fix speaker attribution errors, especially at segment boundaries. Watch for incomplete thoughts that were likely from the previous speaker.

2. Optimize AGGRESSIVELY for readability over verbatim accuracy:
   - Readability is the most important thing!!
   - Remove ALL conversational artifacts (yeah, so, I mean, etc.)
   - Remove ALL filler words (um, uh, like, you know)
   - Remove false starts and self-corrections completely
   - Remove redundant phrases and hesitations
   - Convert any indirect or rambling responses into direct statements
   - Break up run-on sentences into clear, concise statements
   - Maintain natural conversation flow while prioritizing clarity and directness

3. Format the output consistently:
   - Keep the "Speaker X 00:00:00" format (no brackets, no other formatting)
   - Add TWO line breaks between speaker/timestamp and the text
   - Use proper punctuation and capitalization
   - Add paragraph breaks for topic changes
   - When you add paragraph breaks between the same speaker's remarks, no need to restate the speaker attribution
   - Preserve distinct speaker turns

Example input:
Speaker A 00:01:15

Um, yeah, so like, what I was thinking was, you know, when we look at the data, the data shows us that, uh, there's this pattern, this pattern that keeps coming up again and again in the results.

Example output:
Speaker A 00:01:15

When we look at the data, we see a consistent pattern in the results.

When we examine the second part of the analysis, it reveals a completely different finding.

Enhance the following transcript, starting directly with the speaker format:
"""


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_transcript(audio_path):
    """Get transcript from AssemblyAI with speaker diarization"""
    config = aai.TranscriptionConfig(speaker_labels=True, language_code="en")

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path, config=config)

    return transcript.utterances


def format_transcript(utterances):
    """Format transcript into readable text with speaker labels"""
    formatted_sections = []
    current_speaker = None
    current_text = []
    current_start = None

    for utterance in utterances:
        # If this is a new speaker
        if current_speaker != utterance.speaker:
            # Write out the previous section if it exists
            if current_text:
                # Convert milliseconds to seconds for timestamp
                timestamp = format_timestamp(float(current_start) / 1000)
                section = f"Speaker {current_speaker} {timestamp}\n\n{' '.join(current_text).strip()}"
                formatted_sections.append(section)
                current_text = []

            # Start new section
            current_speaker = utterance.speaker
            current_start = utterance.start

        current_text.append(utterance.text.strip())

    # Add the final section
    if current_text:
        # Convert milliseconds to seconds for timestamp
        timestamp = format_timestamp(float(current_start) / 1000)
        section = (
            f"Speaker {current_speaker} {timestamp}\n\n{' '.join(current_text).strip()}"
        )
        formatted_sections.append(section)

    return "\n\n".join(formatted_sections)


async def enhance_transcript_async(chunk_text: str, audio_segment: io.BytesIO) -> str:
    """Enhance transcript using Gemini AI asynchronously"""
    audio_segment.seek(0)  # Ensure we're at the start of the buffer
    response = await model.generate_content_async(
        [
            prompt,
            chunk_text,
            {
                "mime_type": "audio/mp3",
                "data": audio_segment.read(),
            },
        ]
    )
    return response.text


async def process_chunks_async(
    prepared_chunks: List[Tuple[str, io.BytesIO]]
) -> List[str]:
    """Process all chunks in parallel using async API"""
    enhancement_tasks = []
    for chunk_text, audio_segment in prepared_chunks:
        task = enhance_transcript_async(chunk_text, audio_segment)
        enhancement_tasks.append(task)

    print(f"Processing {len(enhancement_tasks)} chunks in parallel...")
    start_time = time.time()

    enhanced_chunks = []
    for i, future in enumerate(asyncio.as_completed(enhancement_tasks), 1):
        try:
            result = await future
            processing_time = time.time() - start_time
            print(
                f"Completed chunk {i}/{len(enhancement_tasks)} in {processing_time:.2f} seconds"
            )
            enhanced_chunks.append(result)
        except Exception as e:
            print(f"Error processing chunk {i}: {str(e)}")
            enhanced_chunks.append(None)

    total_time = time.time() - start_time
    print(f"\nTotal enhancement time: {total_time:.2f} seconds")
    print(f"Average time per chunk: {total_time/len(enhancement_tasks):.2f} seconds")

    return enhanced_chunks


def create_chunks(utterances, target_tokens=2000):
    """Create chunks of utterances that fit within token limits"""
    chunks = []
    current_chunk = []
    current_start = None
    current_end = None

    for utterance in utterances:
        # Start new chunk if this is first utterance
        if not current_chunk:
            current_start = float(utterance.start) / 1000  # Convert ms to seconds
            current_chunk = [utterance]
            current_end = float(utterance.end) / 1000  # Convert ms to seconds
        # Check if adding this utterance would exceed token limit
        elif (
            len(" ".join(u.text for u in current_chunk)) + len(utterance.text)
        ) / 4 > target_tokens:
            # Save current chunk and start new one
            chunks.append(
                {
                    "utterances": current_chunk,
                    "start": current_start,
                    "end": current_end,
                }
            )
            current_chunk = [utterance]
            current_start = float(utterance.start) / 1000
            current_end = float(utterance.end) / 1000
        else:
            # Add to current chunk
            current_chunk.append(utterance)
            current_end = float(utterance.end) / 1000

    # Add final chunk
    if current_chunk:
        chunks.append(
            {"utterances": current_chunk, "start": current_start, "end": current_end}
        )

    return chunks


def get_audio_segment(audio_path, start_time, end_time):
    """Extract audio segment between start and end times and return bytes"""
    audio = AudioSegment.from_file(audio_path)
    start_ms = int(float(start_time) * 1000)
    end_ms = int(float(end_time) * 1000)
    buffer = io.BytesIO()
    audio[start_ms:end_ms].export(buffer, format="mp3")
    buffer.seek(0)
    return buffer


def prepare_chunks(audio_path, transcript_data):
    """Prepare chunks with their audio segments upfront"""
    chunks = create_chunks(transcript_data)
    prepared_chunks = []

    print(f"Preparing {len(chunks)} audio segments...")
    start_time = time.time()
    for i, chunk in enumerate(chunks, 1):
        chunk_text = format_transcript(chunk["utterances"])
        audio_segment = get_audio_segment(audio_path, chunk["start"], chunk["end"])
        # Ensure the buffer is at the start for each use
        audio_segment.seek(0)
        prepared_chunks.append((chunk_text, audio_segment))
        print(f"Prepared audio segment {i}/{len(chunks)}")

    print(f"Audio preparation took {time.time() - start_time:.2f} seconds")
    return prepared_chunks


def get_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_cached_transcript(audio_path: str) -> List[dict]:
    """Get transcript from cache if available and valid"""
    audio_hash = get_file_hash(audio_path)
    cache_dir = Path("transcripts/.cache")
    cache_file = cache_dir / f"{Path(audio_path).stem}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            cached_data = json.load(f)
            if cached_data.get("hash") == audio_hash:
                print("Using cached AssemblyAI transcript...")
                return cached_data["utterances"]

    return None


def save_transcript_cache(audio_path: str, utterances: List) -> None:
    """Save transcript data to cache"""
    audio_hash = get_file_hash(audio_path)
    cache_dir = Path("transcripts/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Convert utterances to JSON-serializable format
    utterances_data = [
        {"speaker": u.speaker, "text": u.text, "start": u.start, "end": u.end}
        for u in utterances
    ]

    cache_data = {"hash": audio_hash, "utterances": utterances_data}

    cache_file = cache_dir / f"{Path(audio_path).stem}.json"
    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)


def process_audio(audio_path):
    """Main processing pipeline"""
    print("Stage 1: Getting transcript from AssemblyAI...")

    # Try to get cached transcript first
    cached_utterances = get_cached_transcript(audio_path)

    if cached_utterances:
        # Convert cached data back to utterance-like objects
        class Utterance:
            def __init__(self, data):
                self.speaker = data["speaker"]
                self.text = data["text"]
                self.start = data["start"]
                self.end = data["end"]

        transcript_data = [Utterance(u) for u in cached_utterances]
    else:
        # Get new transcript from AssemblyAI
        config = aai.TranscriptionConfig(speaker_labels=True, language_code="en")
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path, config=config)
        transcript_data = transcript.utterances

        # Save to cache
        save_transcript_cache(audio_path, transcript_data)

    print("Preparing audio segments...")
    chunks = create_chunks(transcript_data)
    prepared_chunks = prepare_chunks(audio_path, transcript_data)

    # Get original transcript for saving
    original_transcript = "\n".join(
        format_transcript(chunk["utterances"]) for chunk in chunks
    )

    os.makedirs("transcripts", exist_ok=True)

    print("\nStage 2: Enhancing chunks with Gemini...")
    # Run async enhancement in an event loop
    enhanced_chunks = asyncio.run(process_chunks_async(prepared_chunks))

    # Filter out any failed chunks
    enhanced_chunks = [chunk for chunk in enhanced_chunks if chunk is not None]

    # Write transcripts to files
    with open("transcripts/autogenerated-transcript.md", "w", encoding="utf-8") as f:
        f.write(original_transcript)

    with open("transcripts/transcript.md", "w", encoding="utf-8") as f:
        f.write("\n".join(enhanced_chunks))

    print("\nTranscripts have been saved to:")
    print("- transcripts/autogenerated-transcript.md")
    print("- transcripts/transcript.md")


def main():
    parser = argparse.ArgumentParser(
        description="Generate enhanced transcripts from audio files"
    )
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: File '{args.audio_file}' not found")
        return

    try:
        process_audio(args.audio_file)
    except Exception as e:
        print(f"Error processing audio: {str(e)}")


if __name__ == "__main__":
    main()
