from dataclasses import dataclass
import os
from typing import List, Optional, Dict
import json
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import asyncio
from deepgram import Deepgram
import mimetypes

@dataclass
class TranscriptSegment:
    speaker: str
    text: str
    start: float
    end: float

@dataclass
class ProcessedChunk:
    segments: List[TranscriptSegment]
    original_text: str
    processed_text: Optional[str] = None

class TranscriptProcessor:
    def __init__(self, max_tokens: int = 6000):
        """Initialize the TranscriptProcessor with API clients.

        Environment variables required:
        - GOOGLE_API_KEY: API key for Google Gemini
        - DEEPGRAM_API_KEY: API key for Deepgram
        """
        self.max_tokens = max_tokens

        # Get API keys from environment variables
        genai_api_key = os.getenv('GOOGLE_API_KEY')
        deepgram_api_key = os.getenv('DEEPGRAM_API_KEY')

        if not genai_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        if not deepgram_api_key:
            raise ValueError("DEEPGRAM_API_KEY environment variable is not set")

        self.genai_client = genai.Client(api_key=genai_api_key)
        self.deepgram_client = Deepgram(deepgram_api_key)

        # Create search tool
        self.google_search_tool = Tool(
            google_search=GoogleSearch()
        )

        self.prompt_template = """
Your task is to improve the readability of this transcript while maintaining its core meaning. Additionally, you should add relevant links to technical terms, products, papers, people, or concepts mentioned in the transcript.

Examples of input and desired output:

Input:
"Speaker A: Yeah, so like, um, you know, I've been really diving into this whole, like, transformer architecture thing, right? And, um, what's really interesting is like, you know, how they handle this attention mechanism stuff. I mean, it's like, basically, you know, the way it processes sequential data is just, like, mind-blowing if you think about it. And, um, yeah, what I'm trying to say is that, like, the whole self-attention concept is just, you know, really revolutionary and stuff."

Output:
"Speaker A: I've been exploring the [transformer architecture](https://arxiv.org/abs/1706.03762), and it's fascinating how it implements attention mechanisms. The way it processes sequential data is revolutionary, particularly the [self-attention concept](https://distill.pub/2016/augmented-rnns/#attentional-interfaces) that fundamentally changed the field."

Input:
"Speaker B: Yeah, yeah, totally, and like, you know what's really cool is that, um, I've been working with PyTorch for this project I'm doing, and like, you know, implementing BERT has been super helpful because, um, you know, it's like pre-trained and stuff, and I mean, the whole masked language modeling thing is just, like, really powerful, you know what I mean? And then, like, there's all these other models like GPT and RoBERTa that kind of like, you know, built on top of it and made things even better, if that makes sense."

Output:
"Speaker B: I've been working with [PyTorch](https://pytorch.org/) on my project, implementing [BERT](https://arxiv.org/abs/1810.04805) with its pre-trained capabilities. The masked language modeling approach has proven powerful, leading to advancements like [GPT](https://arxiv.org/abs/2005.14165) and [RoBERTa](https://arxiv.org/abs/1907.11692)."

Input:
"Speaker A: Right, right, and you know what's really interesting is like, um, when you look at the training data requirements and stuff, it's like, you know, these large language models need just a massive amount of, like, compute and data to train properly, and I mean, that's why, you know, we're seeing all these different approaches to trying to make it more efficient, like, um, you know, quantization and pruning and stuff like that. And like, yeah, I think that's why the whole debate about compute requirements is getting so much attention nowadays."

Output:
"Speaker A: The training requirements for large language models are substantial, demanding extensive compute resources and data. This has led to efficiency innovations like [quantization](https://arxiv.org/abs/2103.13630) and [pruning](https://arxiv.org/abs/2010.13103), sparking important discussions about computational sustainability in AI."

Instructions:
* Make text highly readable while maintaining the core message and natural speech patterns
* Remove or consolidate:
  - Filler words (um, uh, you know, like, I mean)
  - False starts and self-corrections
  - Repeated phrases and stutters
  - Verbal tics and unnecessary interjections
* Improve sentence structure
* Format for clarity
* Enhance coherence
* Preserve speaker labels and technical terms exactly
* Add Markdown-style links to:
  - Technical terms and concepts
  - Products and tools mentioned
  - Research papers or articles referenced
  - Notable people or organizations
  - Only add links when you're confident about the reference
  - Prioritize official documentation, papers, or authoritative sources
  - For papers, prefer arXiv links when available

Here's the transcript to process:
{text}
"""

    async def transcribe_audio(self, audio_path: str) -> List[TranscriptSegment]:
        """Transcribe audio using Deepgram with automatic format detection"""
        # Get the mime type of the audio file
        mime_type = mimetypes.guess_type(audio_path)[0]
        if not mime_type:
            # Default to mp3 if we can't detect the type
            mime_type = 'audio/mpeg' if audio_path.lower().endswith('.mp3') else 'audio/wav'

        with open(audio_path, 'rb') as audio:
            source = {'buffer': audio, 'mimetype': mime_type}
            response = await self.deepgram_client.transcription.prerecorded(
                source,
                {
                    'smart_format': True,
                    'punctuate': True,
                    'diarize': True,
                    'utterances': True
                }
            )

        segments = []
        for utterance in response['results']['utterances']:
            segments.append(TranscriptSegment(
                speaker=f"Speaker {utterance['speaker']}",
                text=utterance['transcript'],
                start=utterance['start'],
                end=utterance['end']
            ))

        return segments

    def create_chunks(self, segments: List[TranscriptSegment]) -> List[ProcessedChunk]:
        chunks = []
        current_segments = []
        current_text = ""

        for segment in segments:
            segment_text = f"{segment.speaker}: {segment.text}\n"
            potential_text = current_text + segment_text
            potential_prompt = self.prompt_template.format(text=potential_text)

            if len(potential_prompt.split()) > (self.max_tokens * 0.75) and current_segments:
                chunks.append(ProcessedChunk(
                    segments=current_segments,
                    original_text=current_text
                ))
                current_segments = []
                current_text = ""

            current_segments.append(segment)
            current_text += segment_text

        if current_segments:
            chunks.append(ProcessedChunk(
                segments=current_segments,
                original_text=current_text
            ))

        return chunks

    async def process_chunk(self, chunk: ProcessedChunk) -> None:
        """Process a chunk using Gemini 2.0 Flash with Search enabled"""
        prompt = self.prompt_template.format(text=chunk.original_text)

        response = await self.genai_client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config=GenerateContentConfig(
                tools=[self.google_search_tool],
                response_modalities=["TEXT"],
            )
        )

        # Get the main response text
        chunk.processed_text = ""
        for part in response.candidates[0].content.parts:
            chunk.processed_text += part.text

        # Log the search metadata for debugging/verification
        if hasattr(response.candidates[0], 'grounding_metadata') and \
           hasattr(response.candidates[0].grounding_metadata, 'search_entry_point'):
            print(f"Search metadata found for chunk: {response.candidates[0].grounding_metadata.search_entry_point.rendered_content}")

    async def process_transcript(self, audio_path: str) -> str:
        """Main processing pipeline"""
        # Transcribe audio
        segments = await self.transcribe_audio(audio_path)

        # Create chunks
        chunks = self.create_chunks(segments)

        # Process each chunk with retries
        async def process_with_retry(chunk, max_retries=3):
            for attempt in range(max_retries):
                try:
                    await self.process_chunk(chunk)
                    return
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Failed to process chunk after {max_retries} attempts: {e}")
                        raise
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

        # Process chunks in parallel with retries
        tasks = [process_with_retry(chunk) for chunk in chunks]
        await asyncio.gather(*tasks)

        # Combine processed chunks
        final_text = "\n".join(chunk.processed_text for chunk in chunks if chunk.processed_text)

        return final_text


async def main():
    # Make sure to set these environment variables before running:
    # export GOOGLE_API_KEY="your_google_api_key"
    # export DEEPGRAM_API_KEY="your_deepgram_api_key"
    processor = TranscriptProcessor()

    # Example usage with either MP3 or WAV file
    final_transcript = await processor.process_transcript("audio.mp3")

    # Save as markdown file
    with open("transcript.md", "w") as f:
        f.write(final_transcript)
