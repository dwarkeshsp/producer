import gradio as gr
import asyncio
from pathlib import Path
import anthropic
import os
from dataclasses import dataclass
from typing import Dict
from youtube_transcript_api import YouTubeTranscriptApi
import re
import pandas as pd

# Move relevant classes and functions into app.py
@dataclass
class ContentRequest:
    prompt_key: str

class ContentGenerator:
    def __init__(self):
        self.current_prompts = self._load_default_prompts()
        self.client = anthropic.Anthropic()
        
    def _load_default_prompts(self) -> Dict[str, str]:
        """Load default prompts and examples from files and CSVs."""
        
        # Load CSV examples
        try:
            timestamps_df = pd.read_csv("data/Timestamps.csv")
            titles_df = pd.read_csv("data/Titles & Thumbnails.csv")
            descriptions_df = pd.read_csv("data/Viral Episode Descriptions.csv")
            clips_df = pd.read_csv("data/Viral Twitter Clips.csv")
            
            # Format timestamp examples
            timestamp_examples = "\n\n".join(timestamps_df['Timestamps'].dropna().tolist())
            
            # Format title examples
            title_examples = "\n".join([
                f'Title: "{row.Titles}"\nThumbnail: "{row.Thumbnail}"'
                for _, row in titles_df.iterrows()
            ])
            
            # Format description examples
            description_examples = "\n".join([
                f'Tweet: "{row["Tweet Text"]}"'
                for _, row in descriptions_df.iterrows()
            ])
            
            # Format clip examples
            clip_examples = "\n\n".join([
                f'Tweet Text: "{row["Tweet Text"]}"\nClip Transcript: "{row["Clip Transcript"]}"'
                for _, row in clips_df.iterrows() if pd.notna(row["Tweet Text"])
            ])
            
        except Exception as e:
            print(f"Warning: Error loading CSV examples: {e}")
            timestamp_examples = ""
            title_examples = ""
            description_examples = ""
            clip_examples = ""

        # Load base prompts and inject examples
        prompts = {}
        for key in ["previews", "clips", "description", "timestamps", "titles_and_thumbnails"]:
            prompt = Path(f"prompts/{key}.txt").read_text()
            
            # Inject relevant examples
            if key == "timestamps":
                prompt = prompt.replace("{timestamps_examples}", timestamp_examples)
            elif key == "titles_and_thumbnails":
                prompt = prompt.replace("{title_examples}", title_examples)
            elif key == "description":
                prompt = prompt.replace("{description_examples}", description_examples)
            elif key == "clips":
                prompt = prompt.replace("{clip_examples}", clip_examples)
            
            prompts[key] = prompt

        return prompts

    async def generate_content(self, request: ContentRequest, transcript: str) -> str:
        """Generate content using Claude asynchronously."""
        try:
            print(f"\nFull prompt for {request.prompt_key}:")
            print("=== SYSTEM PROMPT ===")
            print(self.current_prompts[request.prompt_key])
            print("=== END SYSTEM PROMPT ===\n")
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8192,
                system=self.current_prompts[request.prompt_key],
                messages=[{"role": "user", "content": f"Process this transcript:\n\n{transcript}"}]
            )
            
            if response and hasattr(response, 'content'):
                return response.content[0].text
            else:
                return f"Error: Unexpected response structure for {request.prompt_key}"
                
        except Exception as e:
            return f"Error generating content: {str(e)}"

def extract_video_id(url: str) -> str:
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
        return f"Error fetching transcript: {str(e)}"

class TranscriptProcessor:
    def __init__(self):
        self.generator = ContentGenerator()

    def _get_youtube_transcript(self, url: str) -> str:
        """Get transcript from YouTube URL."""
        try:
            if video_id := extract_video_id(url):
                return get_transcript(video_id)
            raise Exception("Invalid YouTube URL")
        except Exception as e:
            raise Exception(f"Error fetching YouTube transcript: {str(e)}")

    async def process_transcript(self, input_text: str):
        """Process input and generate all content."""
        try:
            transcript = (
                self._get_youtube_transcript(input_text)
                if any(x in input_text for x in ["youtube.com", "youtu.be"])
                else input_text
            )

            # Process each type sequentially
            sections = {}
            for key in ["titles_and_thumbnails", "description", "previews", "clips", "timestamps"]:
                result = await self.generator.generate_content(ContentRequest(key), transcript)
                sections[key] = result

            # Combine into markdown with H2 headers
            markdown = f"""
## Titles and Thumbnails

{sections['titles_and_thumbnails']}

## Twitter Description

{sections['description']}

## Preview Clips

{sections['previews']}

## Twitter Clips

{sections['clips']}

## Timestamps

{sections['timestamps']}
"""
            return markdown

        except Exception as e:
            return f"Error processing input: {str(e)}"

    def update_prompts(self, *values) -> str:
        """Update the current session's prompts."""
        self.generator.current_prompts.update(zip(
            ["previews", "clips", "description", "timestamps", "titles_and_thumbnails"],
            values
        ))
        return "Prompts updated for this session!"

def create_interface():
    """Create the Gradio interface."""
    processor = TranscriptProcessor()
    
    with gr.Blocks(title="Podcast Content Generator") as app:
        gr.Markdown(
            """
            # Podcast Content Generator
            Generate preview clips, timestamps, descriptions and more from podcast transcripts or YouTube videos.
            
            Simply paste a YouTube URL or raw transcript text to get started!
            """
        )
        
        with gr.Tab("Generate Content"):
            input_text = gr.Textbox(
                label="Input", 
                placeholder="YouTube URL or transcript text...",
                lines=10
            )
            submit_btn = gr.Button("Generate Content")
            
            output = gr.Markdown()  # Single markdown output

            async def process_wrapper(text):
                print("Process wrapper started")
                print(f"Input text: {text[:100]}...")
                
                try:
                    result = await processor.process_transcript(text)
                    print("Process completed, got results")
                    return result
                except Exception as e:
                    print(f"Error in process_wrapper: {str(e)}")
                    return f"# Error\n\n{str(e)}"

            submit_btn.click(
                fn=process_wrapper,
                inputs=input_text,
                outputs=output,
                queue=True
            )

        with gr.Tab("Customize Prompts"):
            gr.Markdown(
                """
                ## Customize Generation Prompts
                Here you can experiment with different prompts during your session.
                Changes will remain active until you reload the page.
                
                Tip: Copy your preferred prompts somewhere safe if you want to reuse them later!
                """
            )

            prompt_inputs = [
                gr.Textbox(
                    label=f"{key.replace('_', ' ').title()} Prompt",
                    lines=10,
                    value=processor.generator.current_prompts[key]
                )
                for key in [
                    "previews",
                    "clips", 
                    "description",
                    "timestamps",
                    "titles_and_thumbnails"
                ]
            ]
            status = gr.Textbox(label="Status", interactive=False)

            # Update prompts when they change
            for prompt in prompt_inputs:
                prompt.change(
                    fn=processor.update_prompts,
                    inputs=prompt_inputs,
                    outputs=[status]
                )

            # Reset button
            reset_btn = gr.Button("Reset to Default Prompts")
            reset_btn.click(
                fn=lambda: (
                    processor.update_prompts(*processor.generator.current_prompts.values()),
                    *processor.generator.current_prompts.values(),
                ),
                outputs=[status] + prompt_inputs,
            )

    return app

if __name__ == "__main__":
    create_interface().launch() 