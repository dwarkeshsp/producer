import gradio as gr
import anthropic
import pandas as pd
from typing import Tuple, Dict, List
from youtube_transcript_api import YouTubeTranscriptApi
import re
from pathlib import Path
import asyncio
import concurrent.futures
from dataclasses import dataclass
import time

# Initialize Anthropic client
client = anthropic.Anthropic()

@dataclass
class ContentRequest:
    prompt_key: str
    max_tokens: int = 2000
    temperature: float = 0.6

class TranscriptProcessor:
    def __init__(self):
        self.current_prompts = self._load_default_prompts()
        
    def _load_default_prompts(self) -> Dict[str, str]:
        """Load default prompts from files."""
        return {
            key: Path(f"prompts/{key}.txt").read_text()
            for key in ["clips", "description", "timestamps", "titles_and_thumbnails"]
        }

    def _load_examples(self, filename: str, columns: List[str]) -> str:
        """Load examples from CSV file."""
        try:
            df = pd.read_csv(f"data/{filename}")
            if len(columns) == 1:
                return "\n\n".join(df[columns[0]].dropna().tolist())
            
            examples = []
            for _, row in df.iterrows():
                if all(pd.notna(row[col]) for col in columns):
                    example = "\n".join(f"{col}: {row[col]}" for col in columns)
                    examples.append(example)
            return "\n\n".join(examples)
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            return ""

    async def _generate_content(self, request: ContentRequest, transcript: str) -> str:
        """Generate content using Claude asynchronously."""
        print(f"Starting {request.prompt_key} generation...")
        start_time = time.time()
        
        example_configs = {
            "clips": ("Viral Twitter Clips.csv", ["Tweet Text", "Clip Transcript"]),
            "description": ("Viral Episode Descriptions.csv", ["Tweet Text"]),
            "timestamps": ("Timestamps.csv", ["Timestamps"]),
            "titles_and_thumbnails": ("Titles & Thumbnails.csv", ["Titles", "Thumbnail"]),
        }
        
        # Build prompt with examples
        full_prompt = self.current_prompts[request.prompt_key]
        if config := example_configs.get(request.prompt_key):
            if examples := self._load_examples(*config):
                full_prompt += f"\n\nPrevious examples:\n{examples}"

        # Run API call in thread pool
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            message = await loop.run_in_executor(
                pool,
                lambda: client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    system=full_prompt,
                    messages=[{"role": "user", "content": [{"type": "text", "text": f"Process this transcript:\n\n{transcript}"}]}]
                )
            )
        result = message.content[0].text
        print(f"Finished {request.prompt_key} in {time.time() - start_time:.2f} seconds")
        return result

    def _get_youtube_transcript(self, url: str) -> str:
        """Get transcript from YouTube URL."""
        try:
            video_id = re.search(
                r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([A-Za-z0-9_-]+)",
                url
            ).group(1)
            transcript = YouTubeTranscriptApi.list_transcripts(video_id).find_transcript(["en"])
            return " ".join(entry["text"] for entry in transcript.fetch())
        except Exception as e:
            raise Exception(f"Error fetching YouTube transcript: {str(e)}")

    async def process_transcript(self, input_text: str) -> Tuple[str, str, str, str]:
        """Process input and generate all content."""
        try:
            # Get transcript from URL or use direct input
            transcript = (
                self._get_youtube_transcript(input_text)
                if any(x in input_text for x in ["youtube.com", "youtu.be"])
                else input_text
            )

            # Define content generation requests
            requests = [
                ContentRequest("clips", max_tokens=8192),
                ContentRequest("description"),
                ContentRequest("timestamps", temperature=0.4),
                ContentRequest("titles_and_thumbnails", temperature=0.7),
            ]

            # Generate all content concurrently
            results = await asyncio.gather(
                *[self._generate_content(req, transcript) for req in requests]
            )
            return tuple(results)

        except Exception as e:
            return (f"Error processing input: {str(e)}",) * 4

    def update_prompts(self, *values) -> str:
        """Update the current session's prompts."""
        keys = ["clips", "description", "timestamps", "titles_and_thumbnails"]
        self.current_prompts = dict(zip(keys, values))
        return "Prompts updated for this session! Changes will reset when you reload the page."

def create_interface():
    """Create the Gradio interface."""
    processor = TranscriptProcessor()
    
    with gr.Blocks(title="Podcast Transcript Analyzer") as app:
        with gr.Tab("Generate Content"):
            gr.Markdown("# Podcast Content Generator")
            input_text = gr.Textbox(label="Input", placeholder="YouTube URL or transcript...", lines=10)
            submit_btn = gr.Button("Generate Content")
            outputs = [
                gr.Textbox(label=label, lines=10, interactive=False)
                for label in ["Twitter Clips", "Twitter Description", "Timestamps", "Title & Thumbnail Suggestions"]
            ]
            
            async def process_wrapper(text):
                return await processor.process_transcript(text)
            
            submit_btn.click(fn=process_wrapper, inputs=[input_text], outputs=outputs)

        with gr.Tab("Experiment with Prompts"):
            gr.Markdown("# Experiment with Prompts")
            gr.Markdown(
                """
            Here you can experiment with different prompts during your session. 
            Changes will remain active until you reload the page.
            
            Tip: Copy your preferred prompts somewhere safe if you want to reuse them later!
            """
            )

            prompt_inputs = [
                gr.Textbox(
                    label="Clips Prompt", lines=10, value=processor.current_prompts["clips"]
                ),
                gr.Textbox(
                    label="Description Prompt",
                    lines=10,
                    value=processor.current_prompts["description"],
                ),
                gr.Textbox(
                    label="Timestamps Prompt",
                    lines=10,
                    value=processor.current_prompts["timestamps"],
                ),
                gr.Textbox(
                    label="Titles & Thumbnails Prompt",
                    lines=10,
                    value=processor.current_prompts["titles_and_thumbnails"],
                ),
            ]
            status = gr.Textbox(label="Status", interactive=False)

            # Update prompts when they change
            for prompt in prompt_inputs:
                prompt.change(fn=processor.update_prompts, inputs=prompt_inputs, outputs=[status])

            # Reset button
            reset_btn = gr.Button("Reset to Default Prompts")
            reset_btn.click(
                fn=lambda: (
                    processor.update_prompts(*processor.current_prompts.values()),
                    *processor.current_prompts.values(),
                ),
                outputs=[status] + prompt_inputs,
            )

    return app

if __name__ == "__main__":
    create_interface().launch()
