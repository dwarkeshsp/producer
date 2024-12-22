import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import os
from typing import List, Optional

import assemblyai as aai
from google import generativeai
from pydub import AudioSegment
import asyncio
import io


@dataclass
class Utterance:
    """A single utterance from a speaker"""

    speaker: str
    text: str
    start: int  # milliseconds
    end: int  # milliseconds

    @property
    def timestamp(self) -> str:
        """Format start time as HH:MM:SS"""
        seconds = self.start // 1000
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"


class Transcriber:
    """Handles getting and caching transcripts from AssemblyAI"""

    def __init__(self, api_key: str):
        aai.settings.api_key = api_key
        self.cache_dir = Path("transcripts/.cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_transcript(self, audio_path: Path) -> List[Utterance]:
        """Get transcript, using cache if available"""
        cached = self._get_cached(audio_path)
        if cached:
            print("Using cached AssemblyAI transcript...")
            return cached

        print("Getting new transcript from AssemblyAI...")
        return self._get_fresh(audio_path)

    def _get_cached(self, audio_path: Path) -> Optional[List[Utterance]]:
        """Try to get transcript from cache"""
        cache_file = self.cache_dir / f"{audio_path.stem}.json"
        if not cache_file.exists():
            return None

        with open(cache_file) as f:
            data = json.load(f)
            if data["hash"] != self._get_file_hash(audio_path):
                return None

            return [Utterance(**u) for u in data["utterances"]]

    def _get_fresh(self, audio_path: Path) -> List[Utterance]:
        """Get new transcript from AssemblyAI"""
        config = aai.TranscriptionConfig(speaker_labels=True, language_code="en")
        transcript = aai.Transcriber().transcribe(str(audio_path), config=config)

        utterances = [
            Utterance(speaker=u.speaker, text=u.text, start=u.start, end=u.end)
            for u in transcript.utterances
        ]

        self._save_cache(audio_path, utterances)
        return utterances

    def _save_cache(self, audio_path: Path, utterances: List[Utterance]) -> None:
        """Save transcript to cache"""
        cache_file = self.cache_dir / f"{audio_path.stem}.json"
        data = {
            "hash": self._get_file_hash(audio_path),
            "utterances": [vars(u) for u in utterances],
        }
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class Enhancer:
    """Handles enhancing transcripts using Gemini"""

    def __init__(self, api_key: str):
        generativeai.configure(api_key=api_key)
        self.model = generativeai.GenerativeModel("gemini-exp-1206")

        # Update prompt path
        prompt_path = Path("prompts/enhance.txt")
        self.prompt = prompt_path.read_text()

    async def enhance_chunks(self, chunks: List[tuple[str, io.BytesIO]]) -> List[str]:
        """Enhance multiple transcript chunks in parallel"""
        tasks = [self._enhance_chunk(text, audio) for text, audio in chunks]

        print(f"Enhancing {len(tasks)} chunks in parallel...")
        results = []
        for i, future in enumerate(asyncio.as_completed(tasks), 1):
            try:
                result = await future
                results.append(result)
                print(f"Completed chunk {i}/{len(tasks)}")
            except Exception as e:
                print(f"Error enhancing chunk {i}: {e}")
                results.append(None)

        return [r for r in results if r is not None]

    async def _enhance_chunk(self, text: str, audio: io.BytesIO) -> str:
        """Enhance a single chunk"""
        audio.seek(0)
        response = await self.model.generate_content_async(
            [self.prompt, text, {"mime_type": "audio/mp3", "data": audio.read()}]
        )
        return response.text


def prepare_audio_chunks(
    audio_path: Path, utterances: List[Utterance]
) -> List[tuple[str, io.BytesIO]]:
    """Prepare audio chunks and their corresponding text"""
    chunks = []
    current = []
    current_text = []

    for u in utterances:
        # Start new chunk if this is first utterance or would exceed token limit
        if not current or len(" ".join(current_text)) > 8000:  # ~2000 tokens
            if current:
                chunks.append((current[0].start, current[-1].end, current))
            current = [u]
            current_text = [u.text]
        else:
            current.append(u)
            current_text.append(u.text)

    # Add final chunk
    if current:
        chunks.append((current[0].start, current[-1].end, current))

    # Prepare audio segments and format text
    audio = AudioSegment.from_file(audio_path)
    prepared = []

    print(f"Preparing {len(chunks)} audio segments...")
    for start_ms, end_ms, utterances in chunks:
        # Get audio segment
        segment = audio[start_ms:end_ms]
        buffer = io.BytesIO()
        segment.export(buffer, format="mp3")

        # Format text
        text = format_transcript(utterances)

        prepared.append((text, buffer))

    return prepared


def format_transcript(utterances: List[Utterance]) -> str:
    """Format utterances into readable text"""
    sections = []
    current_speaker = None
    current_text = []

    for u in utterances:
        if current_speaker != u.speaker and current_text:
            sections.append(
                f"Speaker {current_speaker} {utterances[0].timestamp}\n\n{' '.join(current_text)}"
            )
            current_text = []
        current_speaker = u.speaker
        current_text.append(u.text)

    if current_text:
        sections.append(
            f"Speaker {current_speaker} {utterances[0].timestamp}\n\n{' '.join(current_text)}"
        )

    return "\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", help="Audio file to transcribe")
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        return

    # Initialize services
    transcriber = Transcriber(os.getenv("ASSEMBLYAI_API_KEY"))
    enhancer = Enhancer(os.getenv("GOOGLE_API_KEY"))

    # Create output directory
    out_dir = Path("transcripts")
    out_dir.mkdir(exist_ok=True)

    # Get transcript
    utterances = transcriber.get_transcript(audio_path)

    # Save original transcript
    original = format_transcript(utterances)
    (out_dir / "autogenerated-transcript.md").write_text(original)

    # Prepare and enhance chunks
    chunks = prepare_audio_chunks(audio_path, utterances)
    enhanced = asyncio.run(enhancer.enhance_chunks(chunks))

    # Save enhanced transcript
    (out_dir / "transcript.md").write_text("\n".join(enhanced))

    print("\nTranscripts saved to:")
    print("- transcripts/autogenerated-transcript.md")
    print("- transcripts/transcript.md")


if __name__ == "__main__":
    main()
