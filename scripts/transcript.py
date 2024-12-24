import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import os
from typing import List, Tuple, Iterator
import assemblyai as aai
from google import generativeai
from pydub import AudioSegment
import asyncio
import io
from multiprocessing import Pool
from functools import partial
from itertools import groupby


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
        seconds = int(self.start // 1000)
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
        cache_file = self.cache_dir / f"{audio_path.stem}.json"
        
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
                if data["hash"] == self._get_file_hash(audio_path):
                    print("Using cached AssemblyAI transcript...")
                    # Create proper Utterance objects from cached data
                    return [
                        Utterance(
                            speaker=u["speaker"],
                            text=u["text"],
                            start=u["start"],
                            end=u["end"]
                        )
                        for u in data["utterances"]
                    ]

        print("Getting new transcript from AssemblyAI...")
        config = aai.TranscriptionConfig(speaker_labels=True, language_code="en")
        transcript = aai.Transcriber().transcribe(str(audio_path), config=config)
        
        utterances = [
            Utterance(
                speaker=u.speaker,
                text=u.text,
                start=u.start,
                end=u.end
            )
            for u in transcript.utterances
        ]
        
        # Cache the raw utterance data
        cache_data = {
            "hash": self._get_file_hash(audio_path),
            "utterances": [
                {
                    "speaker": u.speaker,
                    "text": u.text,
                    "start": u.start,
                    "end": u.end
                }
                for u in utterances
            ]
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
            
        return utterances

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
        self.prompt = Path("prompts/enhance.txt").read_text()

    async def enhance_chunks(self, chunks: List[Tuple[str, io.BytesIO]]) -> List[str]:
        """Enhance multiple transcript chunks concurrently with concurrency control"""
        print(f"Enhancing {len(chunks)} chunks...")
        
        # Create a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(3)  # Allow up to 3 concurrent requests
        
        async def process_chunk(i: int, chunk: Tuple[str, io.BytesIO]) -> str:
            text, audio = chunk
            async with semaphore:
                audio.seek(0)
                response = await self.model.generate_content_async(
                    [self.prompt, text, {"mime_type": "audio/mp3", "data": audio.read()}]
                )
                print(f"Completed chunk {i+1}/{len(chunks)}")
                return response.text

        # Create tasks for all chunks and run them concurrently
        tasks = [
            process_chunk(i, chunk) 
            for i, chunk in enumerate(chunks)
        ]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        return results


@dataclass
class SpeakerDialogue:
    """Represents a continuous section of speech from a single speaker"""
    speaker: str
    utterances: List[Utterance]
    
    @property
    def start(self) -> int:
        """Start time of first utterance"""
        return self.utterances[0].start
    
    @property
    def end(self) -> int:
        """End time of last utterance"""
        return self.utterances[-1].end
    
    @property
    def timestamp(self) -> str:
        """Format start time as HH:MM:SS"""
        return self.utterances[0].timestamp
    
    def format(self) -> str:
        """Format this dialogue as text with newlines between utterances"""
        texts = [u.text + "\n\n" for u in self.utterances]  # Add two newlines after each utterance
        combined_text = ''.join(texts).rstrip()  # Remove trailing whitespace at the end
        return f"Speaker {self.speaker} {self.timestamp}\n\n{combined_text}"


def group_utterances_by_speaker(utterances: List[Utterance]) -> Iterator[SpeakerDialogue]:
    """Group consecutive utterances by the same speaker"""
    for speaker, group in groupby(utterances, key=lambda u: u.speaker):
        yield SpeakerDialogue(speaker=speaker, utterances=list(group))


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """
    Estimate number of tokens in text
    Args:
        text: The text to estimate tokens for
        chars_per_token: Estimated characters per token (default 4)
    """
    return (len(text) + chars_per_token - 1) // chars_per_token


def chunk_dialogues(
    dialogues: Iterator[SpeakerDialogue], 
    max_tokens: int = 2000, 
    chars_per_token: int = 4
) -> List[List[SpeakerDialogue]]:
    """
    Split dialogues into chunks that fit within token limit
    Args:
        dialogues: Iterator of SpeakerDialogues
        max_tokens: Maximum tokens per chunk
        chars_per_token: Estimated characters per token (default 4)
    """
    chunks = []
    current_chunk = []
    current_text = ""
    
    for dialogue in dialogues:
        # Format this dialogue
        formatted = dialogue.format()
        
        # If adding this dialogue would exceed token limit, start new chunk
        new_text = current_text + "\n\n" + formatted if current_text else formatted
        if current_chunk and estimate_tokens(new_text, chars_per_token) > max_tokens:
            chunks.append(current_chunk)
            current_chunk = [dialogue]
            current_text = formatted
        else:
            current_chunk.append(dialogue)
            current_text = new_text
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def format_chunk(dialogues: List[SpeakerDialogue]) -> str:
    """Format a chunk of dialogues into readable text"""
    return "\n\n".join(dialogue.format() for dialogue in dialogues)


def prepare_audio_chunks(audio_path: Path, utterances: List[Utterance]) -> List[Tuple[str, io.BytesIO]]:
    """Prepare audio chunks and their corresponding text"""
    # Group utterances by speaker and split into chunks
    dialogues = group_utterances_by_speaker(utterances)
    chunks = chunk_dialogues(dialogues)
    print(f"Preparing {len(chunks)} audio segments...")
    
    # Load audio once
    audio = AudioSegment.from_file(audio_path)
    
    # Process each chunk
    prepared = []
    for chunk in chunks:
        # Extract just the needed segment
        segment = audio[chunk[0].start:chunk[-1].end]
        buffer = io.BytesIO()
        # Use lower quality MP3 for faster processing
        segment.export(buffer, format="mp3", parameters=["-q:a", "9"])
        prepared.append((format_chunk(chunk), buffer))
    
    return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", help="Audio file to transcribe")
    args = parser.parse_args()
    
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")
        
    out_dir = Path("output/transcripts")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get transcript
        transcriber = Transcriber(os.getenv("ASSEMBLYAI_API_KEY"))
        utterances = transcriber.get_transcript(audio_path)
        
        # Save original transcript
        dialogues = list(group_utterances_by_speaker(utterances))  # Convert iterator to list
        original = format_chunk(dialogues)
        (out_dir / "autogenerated-transcript.md").write_text(original)
        
        # Enhance transcript
        enhancer = Enhancer(os.getenv("GOOGLE_API_KEY"))
        chunks = prepare_audio_chunks(audio_path, utterances)
        enhanced = asyncio.run(enhancer.enhance_chunks(chunks))
        
        # Save enhanced transcript
        merged = "\n\n".join(chunk.strip() for chunk in enhanced)
        (out_dir / "transcript.md").write_text(merged)
        
        print("\nTranscripts saved to:")
        print(f"- {out_dir}/autogenerated-transcript.md")
        print(f"- {out_dir}/transcript.md")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
