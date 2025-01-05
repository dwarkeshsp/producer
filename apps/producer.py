import gradio as gr
import asyncio
from pathlib import Path
from ..utils.content_generator import ContentGenerator, ContentRequest
from ..utils.youtube_utils import get_transcript, extract_video_id

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
            # Get transcript from URL or use direct input
            transcript = (
                self._get_youtube_transcript(input_text)
                if any(x in input_text for x in ["youtube.com", "youtu.be"])
                else input_text
            )

            # Define content generation requests
            requests = [
                ContentRequest("previews", max_tokens=8192),
                ContentRequest("clips", max_tokens=8192),
                ContentRequest("description"),
                ContentRequest("timestamps"),
                ContentRequest("titles_and_thumbnails"),
            ]

            # Generate all content concurrently
            results = await asyncio.gather(
                *[self.generator.generate_content(req, transcript) for req in requests]
            )
            return tuple(results)

        except Exception as e:
            return (f"Error processing input: {str(e)}",) * 5

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
    
    with gr.Blocks(title="Podcast Transcript Analyzer") as app:
        with gr.Tab("Generate Content"):
            gr.Markdown("# Podcast Content Generator")
            input_text = gr.Textbox(label="Input", placeholder="YouTube URL or transcript...", lines=10)
            submit_btn = gr.Button("Generate Content")
            outputs = [
                gr.Textbox(label=label, lines=10, interactive=False)
                for label in ["Preview Clips", "Twitter Clips", "Twitter Description", "Timestamps", "Title & Thumbnail Suggestions"]
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
                    label="Preview Clips Prompt", lines=10, value=processor.generator.current_prompts["previews"]
                ),
                gr.Textbox(
                    label="Clips Prompt", lines=10, value=processor.generator.current_prompts["clips"]
                ),
                gr.Textbox(
                    label="Description Prompt",
                    lines=10,
                    value=processor.generator.current_prompts["description"],
                ),
                gr.Textbox(
                    label="Timestamps Prompt",
                    lines=10,
                    value=processor.generator.current_prompts["timestamps"],
                ),
                gr.Textbox(
                    label="Titles & Thumbnails Prompt",
                    lines=10,
                    value=processor.generator.current_prompts["titles_and_thumbnails"],
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
                    processor.update_prompts(*processor.generator.current_prompts.values()),
                    *processor.generator.current_prompts.values(),
                ),
                outputs=[status] + prompt_inputs,
            )

    return app

if __name__ == "__main__":
    create_interface().launch() 