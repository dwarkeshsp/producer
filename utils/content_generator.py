import anthropic
from dataclasses import dataclass
from pathlib import Path
import asyncio
import concurrent.futures
import time
from typing import Dict, List
import pandas as pd

client = anthropic.Anthropic()

@dataclass
class ContentRequest:
    prompt_key: str
    max_tokens: int = 2000
    temperature: float = 1.0

class ContentGenerator:
    def __init__(self):
        self.current_prompts = self._load_default_prompts()
        
    def _load_default_prompts(self) -> Dict[str, str]:
        """Load default prompts from files."""
        return {
            key: Path(f"prompts/{key}.txt").read_text()
            for key in ["previews", "clips", "description", "timestamps", "titles_and_thumbnails"]
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

    async def generate_content(self, request: ContentRequest, transcript: str) -> str:
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