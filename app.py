import gradio as gr
import anthropic
import pandas as pd
from typing import Tuple, Dict
from youtube_transcript_api import YouTubeTranscriptApi
import re
import json
from pathlib import Path

# Initialize Anthropic client and settings
client = anthropic.Anthropic()
PROMPTS_FILE = Path("prompts.json")

# Default prompts
DEFAULT_PROMPTS = {
    "clips": """You are a social media expert for the Dwarkesh Podcast. Generate 10 viral-worthy clips from the transcript.
Format as:
Tweet 1
Tweet Text: [text]
Clip Transcript: [45-120 seconds of transcript]

Previous examples:
{clips_examples}""",
    "description": """Create an engaging episode description tweet (280 chars max) that:
1. Highlights compelling aspects
2. Includes topic areas and handles
3. Ends with "Links below" or "Enjoy!"

Previous examples:
{description_examples}""",
    "timestamps": """Generate timestamps (HH:MM:SS) every 3-8 minutes covering key transitions and moments.
Use 2-6 word descriptions.
Start at 00:00:00.

Previous examples:
{timestamps_examples}""",
    "titles_and_thumbnails": """Create 3-5 compelling title-thumbnail combinations that tell a story.

Title Format: "Guest Name – Key Story or Core Insight"
Thumbnail: 2-4 ALL CAPS words that create intrigue with the title

Example: "David Reich – How One Small Tribe Conquered the World 70,000 Years Ago"
Thumbnail: "LAST HUMANS STANDING"

The combination should create intellectual curiosity without clickbait.

Previous examples:
{titles_and_thumbnails_examples}""",
}


def load_prompts() -> Dict[str, str]:
    """Load prompts from file or return defaults."""
    if PROMPTS_FILE.exists():
        with open(PROMPTS_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_PROMPTS


def save_prompts(prompts: Dict[str, str]) -> str:
    """Save prompts to file."""
    try:
        with open(PROMPTS_FILE, "w") as f:
            json.dump(prompts, f, indent=2)
        return f"Prompts saved to: {PROMPTS_FILE.absolute()}"
    except Exception as e:
        return f"Error saving prompts: {str(e)}"


def load_examples(filename: str, columns: list) -> str:
    """Load examples from CSV file."""
    try:
        df = pd.read_csv(filename)
        if len(columns) == 1:
            examples = df[columns[0]].dropna().tolist()
            return "\n\n".join(examples)

        examples = []
        for _, row in df.iterrows():
            if all(pd.notna(row[col]) for col in columns):
                example = "\n".join(f"{col}: {row[col]}" for col in columns)
                examples.append(example)
        return "\n\n".join(examples)
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        return ""


def generate_content(
    prompt_key: str, transcript: str, max_tokens: int = 1000, temp: float = 0.6
) -> str:
    """Generate content using Claude."""
    examples = {
        "clips": load_examples(
            "Viral Twitter Clips.csv", ["Tweet Text", "Clip Transcript"]
        ),
        "description": load_examples("Viral Episode Descriptions.csv", ["Tweet Text"]),
        "timestamps": load_examples("Timestamps.csv", ["Timestamps"]),
        "titles_and_thumbnails": load_examples(
            "Titles & Thumbnails.csv", ["Titles", "Thumbnail"]
        ),
    }

    prompts = load_prompts()
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=max_tokens,
        temperature=temp,
        system=prompts[prompt_key].format(
            **{f"{prompt_key}_examples": examples[prompt_key]}
        ),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Process this transcript:\n\n{transcript}",
                    }
                ],
            }
        ],
    )
    return message.content[0].text


def get_youtube_transcript(url: str) -> str:
    """Get transcript from YouTube URL."""
    try:
        video_id = re.search(
            r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([A-Za-z0-9_-]+)",
            url,
        ).group(1)
        transcript = YouTubeTranscriptApi.list_transcripts(video_id).find_transcript(
            ["en"]
        )
        return " ".join(entry["text"] for entry in transcript.fetch())
    except Exception as e:
        raise Exception(f"Error fetching YouTube transcript: {str(e)}")


def process_transcript(input_text: str) -> Tuple[str, str, str, str]:
    """Process input and generate all content."""
    try:
        # Get transcript from URL or use direct input
        transcript = (
            get_youtube_transcript(input_text)
            if any(x in input_text for x in ["youtube.com", "youtu.be"])
            else input_text
        )

        # Generate all content types
        return (
            generate_content("clips", transcript, max_tokens=8192),
            generate_content("description", transcript),
            generate_content("timestamps", transcript, temp=0.4),
            generate_content("titles_and_thumbnails", transcript, temp=0.7),
        )
    except Exception as e:
        error_msg = f"Error processing input: {str(e)}"
        return (error_msg,) * 4


def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(title="Podcast Transcript Analyzer") as app:
        with gr.Tab("Generate Content"):
            gr.Markdown("# Podcast Content Generator")
            input_text = gr.Textbox(
                label="Input", placeholder="YouTube URL or transcript...", lines=10
            )
            submit_btn = gr.Button("Generate Content")
            outputs = [
                gr.Textbox(label="Twitter Clips", lines=10, interactive=False),
                gr.Textbox(label="Twitter Description", lines=3, interactive=False),
                gr.Textbox(label="Timestamps", lines=10, interactive=False),
                gr.Textbox(
                    label="Title & Thumbnail Suggestions", lines=10, interactive=False
                ),
            ]
            submit_btn.click(
                fn=process_transcript, inputs=[input_text], outputs=outputs
            )

        with gr.Tab("Configure Prompts"):
            gr.Markdown("# Edit Prompts")
            gr.Markdown(f"Prompts are saved to: `{PROMPTS_FILE.absolute()}`")

            prompts = load_prompts()
            prompt_inputs = [
                gr.Textbox(
                    label="Clips Prompt",
                    lines=10,
                    value=prompts["clips"],
                    interactive=True,
                ),
                gr.Textbox(
                    label="Description Prompt",
                    lines=10,
                    value=prompts["description"],
                    interactive=True,
                ),
                gr.Textbox(
                    label="Timestamps Prompt",
                    lines=10,
                    value=prompts["timestamps"],
                    interactive=True,
                ),
                gr.Textbox(
                    label="Titles & Thumbnails Prompt",
                    lines=10,
                    value=prompts["titles_and_thumbnails"],
                    interactive=True,
                ),
            ]
            status = gr.Textbox(label="Status", interactive=False)

            def save_prompts_callback(*values):
                new_prompts = {
                    "clips": values[0],
                    "description": values[1],
                    "timestamps": values[2],
                    "titles_and_thumbnails": values[3],
                }
                return save_prompts(new_prompts)

            for prompt in prompt_inputs:
                prompt.change(
                    fn=save_prompts_callback, inputs=prompt_inputs, outputs=[status]
                )

            reset_btn = gr.Button("Reset to Default Prompts")
            reset_btn.click(
                fn=lambda: (save_prompts(DEFAULT_PROMPTS), *DEFAULT_PROMPTS.values()),
                outputs=[status] + prompt_inputs,
            )

    return app


if __name__ == "__main__":
    create_interface().launch()
