# Podcast Content Generator

A Gradio app that helps podcast producers generate preview clips, timestamps, descriptions, and more from podcast transcripts or YouTube videos.

## Features

- Generate preview clips suggestions
- Create Twitter/social media clips
- Generate episode descriptions
- Create timestamps
- Get title and thumbnail suggestions
- Support for YouTube URLs or raw transcript text
- Customizable prompts for each type of content

## Usage

1. Paste a YouTube URL or transcript text into the input box
2. Click "Generate Content" to process
3. Get generated content in various formats
4. Optionally customize the prompts used for generation

## Environment Variables

The app requires the following environment variable:
- `ANTHROPIC_API_KEY`: Your Anthropic API key for Claude

## Credits

Built with:
- Gradio
- Claude AI (Anthropic)
- YouTube Transcript API