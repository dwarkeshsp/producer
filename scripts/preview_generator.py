import argparse
from pathlib import Path
import os
from google import generativeai
from pydub import AudioSegment


class PreviewGenerator:
    """Handles generating preview suggestions using Gemini"""

    def __init__(self, api_key: str):
        generativeai.configure(api_key=api_key)
        self.model = generativeai.GenerativeModel("gemini-exp-1206")
        self.prompt = Path("prompts/previews.txt").read_text()

    async def generate_previews(self, audio_path: Path, transcript_path: Path = None) -> str:
        """Generate preview suggestions for the given audio file and optional transcript"""
        print("Generating preview suggestions...")
        
        # Load and compress audio for Gemini
        audio = AudioSegment.from_file(audio_path)
        
        # Create a buffer for the compressed audio
        import io
        buffer = io.BytesIO()
        # Use lower quality MP3 for faster processing
        audio.export(buffer, format="mp3", parameters=["-q:a", "9"])
        buffer.seek(0)
        
        # Use the File API to upload the audio
        audio_file = generativeai.upload_file(buffer, mime_type="audio/mp3")
        
        # Prepare content for Gemini
        content = [self.prompt]
        content.append(audio_file)  # Add the uploaded file reference
        
        # Add transcript if provided
        if transcript_path and transcript_path.exists():
            print("Including transcript in analysis...")
            # Upload transcript as a file too
            transcript_file = generativeai.upload_file(transcript_path)
            content.append(transcript_file)
        
        # Generate suggestions using Gemini
        response = await self.model.generate_content_async(content)
        
        return response.text


async def main():
    parser = argparse.ArgumentParser(description="Generate podcast preview suggestions")
    parser.add_argument("audio_file", help="Audio file to analyze")
    parser.add_argument("--transcript", "-t", help="Optional transcript file")
    args = parser.parse_args()
    
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")
    
    transcript_path = Path(args.transcript) if args.transcript else None
    if transcript_path and not transcript_path.exists():
        print(f"Warning: Transcript file not found: {transcript_path}")
        transcript_path = None
    
    # Ensure output directory exists
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "previews.txt"
    
    try:
        generator = PreviewGenerator(os.getenv("GOOGLE_API_KEY"))
        suggestions = await generator.generate_previews(audio_path, transcript_path)
        
        # Save output
        output_path.write_text(suggestions)
        print(f"\nPreview suggestions saved to: {output_path}")
        
        # Also print to console
        print("\nPreview Suggestions:")
        print("-" * 40)
        print(suggestions)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())