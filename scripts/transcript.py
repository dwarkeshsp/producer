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
    start: int  # timestamp in ms from AssemblyAI
    end: int    # timestamp in ms from AssemblyAI

    @property
    def timestamp(self) -> str:
        """Format start time as HH:MM:SS"""
        seconds = int(self.start // 1000)  # Convert ms to seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Transcriber:
    """Handles getting and caching transcripts from AssemblyAI"""

    def __init__(self, api_key: str):
        aai.settings.api_key = api_key
        self.cache_dir = Path("output/transcripts/.cache")
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
        print(f"Enhancing {len(chunks)} chunks...")
        
        async def process_chunk(chunk, index):
            text, audio = chunk
            try:
                result = await self._enhance_chunk_with_retry(text, audio)
                print(f"Completed chunk {index + 1}/{len(chunks)}")
                if result == text:  # Check if output matches input exactly
                    print("WARNING: Enhanced text matches input exactly!")
                return result
            except Exception as e:
                print(f"Error in chunk {index + 1}: {e}")
                return None

        # Create all tasks at once and wait for them all to complete
        tasks = [process_chunk(chunk, i) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        
        # Filter out failed chunks
        return [r for r in results if r is not None]

    async def _enhance_chunk_with_retry(self, text: str, audio: io.BytesIO, max_retries: int = 3) -> Optional[str]:
        """Enhance a single chunk with retries"""
        for attempt in range(max_retries):
            try:
                return await self._enhance_chunk(text, audio)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts: {e}")
                    return None
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def _enhance_chunk(self, text: str, audio: io.BytesIO) -> str:
        """Enhance a single chunk"""
        audio.seek(0)
        
        response = await self.model.generate_content_async(
            [self.prompt, text, {"mime_type": "audio/mp3", "data": audio.read()}]
        )
        
        return response.text


def prepare_audio_chunks(audio_path: Path, utterances: List[Utterance]) -> List[tuple[str, io.BytesIO]]:
    """Prepare audio chunks and their corresponding text"""
    def chunk_utterances(utterances: List[Utterance]) -> List[List[Utterance]]:
        chunks = []
        current = []
        text_length = 0
        
        for u in utterances:
            # Check if adding this utterance would exceed token limit
            new_length = text_length + len(u.text)
            if not current or new_length > 8000:  # ~2000 tokens
                if current:
                    chunks.append(current)
                current = [u]
                text_length = len(u.text)
            else:
                current.append(u)
                text_length = new_length
                
        if current:
            chunks.append(current)
            
        return chunks

    # Split utterances into chunks
    chunks = chunk_utterances(utterances)
    
    # Load audio file once
    audio = AudioSegment.from_file(audio_path)
    
    # Prepare segments
    print(f"Preparing {len(chunks)} audio segments...")
    prepared = []
    for chunk in chunks:
        # Extract audio segment
        start_ms = chunk[0].start
        end_ms = chunk[-1].end
        segment = audio[start_ms:end_ms]
        
        # Export to buffer
        buffer = io.BytesIO()
        segment.export(buffer, format="mp3")
        
        # Format text
        text = format_transcript(chunk)
        
        prepared.append((text, buffer))

    return prepared


def format_transcript(utterances: List[Utterance]) -> str:
    """Format utterances into readable text"""
    sections = []
    current_speaker = None
    current_texts = []
    
    for u in utterances:
        # When speaker changes, output the accumulated text
        if current_speaker != u.speaker:
            if current_texts:  # Don't output empty sections
                sections.append(f"Speaker {current_speaker} {utterances[len(sections)].timestamp}\n\n{''.join(current_texts)}")
            current_speaker = u.speaker
            current_texts = []
        current_texts.append(u.text)
    
    # Don't forget the last section
    if current_texts:
        sections.append(f"Speaker {current_speaker} {utterances[len(sections)].timestamp}\n\n{''.join(current_texts)}")
    
    return "\n\n".join(sections)


def main():
    def setup_args() -> Path:
        parser = argparse.ArgumentParser()
        parser.add_argument("audio_file", help="Audio file to transcribe")
        args = parser.parse_args()
        
        audio_path = Path(args.audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"File not found: {audio_path}")
        return audio_path
    
    def setup_output_dir() -> Path:
        out_dir = Path("output/transcripts")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    
    try:
        # Setup
        audio_path = setup_args()
        out_dir = setup_output_dir()
        
        # Initialize services
        transcriber = Transcriber(os.getenv("ASSEMBLYAI_API_KEY"))
        enhancer = Enhancer(os.getenv("GOOGLE_API_KEY"))
        
        # Process
        utterances = transcriber.get_transcript(audio_path)
        chunks = prepare_audio_chunks(audio_path, utterances)
        
        # Save original transcript
        original = format_transcript(utterances)
        (out_dir / "autogenerated-transcript.md").write_text(original)
        
        # Enhance and save
        enhanced = asyncio.run(enhancer.enhance_chunks(chunks))
        merged_transcript = "\n\n".join(chunk.strip() for chunk in enhanced)
        (out_dir / "transcript.md").write_text(merged_transcript)
        
        print("\nTranscripts saved to:")
        print("- output/transcripts/autogenerated-transcript.md")
        print("- output/transcripts/transcript.md")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
