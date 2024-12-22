import argparse
import assemblyai as aai
from google import generativeai
import os
from pydub import AudioSegment
import concurrent.futures

# Suppress gRPC shutdown warnings
os.environ["GRPC_PYTHON_LOG_LEVEL"] = "error"

# Initialize API clients
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

aai.settings.api_key = ASSEMBLYAI_API_KEY
generativeai.configure(api_key=GOOGLE_API_KEY)
model = generativeai.GenerativeModel("gemini-exp-1206")


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_transcript(audio_path):
    """Get transcript from AssemblyAI with speaker diarization"""
    config = aai.TranscriptionConfig(speaker_labels=True, language_code="en")

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path, config=config)

    return transcript.utterances


def format_transcript(utterances):
    """Format transcript into readable text with speaker labels"""
    formatted_sections = []
    current_speaker = None
    current_text = []
    current_start = None

    for utterance in utterances:
        # If this is a new speaker
        if current_speaker != utterance.speaker:
            # Write out the previous section if it exists
            if current_text:
                # Convert milliseconds to seconds for timestamp
                timestamp = format_timestamp(float(current_start) / 1000)
                section = f"Speaker {current_speaker} {timestamp}\n\n{' '.join(current_text).strip()}"
                formatted_sections.append(section)
                current_text = []

            # Start new section
            current_speaker = utterance.speaker
            current_start = utterance.start

        current_text.append(utterance.text.strip())

    # Add the final section
    if current_text:
        # Convert milliseconds to seconds for timestamp
        timestamp = format_timestamp(float(current_start) / 1000)
        section = (
            f"Speaker {current_speaker} {timestamp}\n\n{' '.join(current_text).strip()}"
        )
        formatted_sections.append(section)

    return "\n\n".join(formatted_sections)


def enhance_transcript(chunk_text, audio_segment):
    """Enhance transcript using Gemini AI with both text and audio"""
    prompt = """You are an expert transcript editor. Your task is to enhance this transcript for maximum readability while maintaining the core message.

IMPORTANT: Respond ONLY with the enhanced transcript. Do not include any explanations, headers, or phrases like "Here is the transcript."

Note: Below you'll find an auto-generated transcript that may help with speaker identification, but focus on creating your own high-quality transcript from the audio.

Please:
1. Fix speaker attribution errors, especially at segment boundaries. Watch for incomplete thoughts that were likely from the previous speaker.

2. Optimize AGGRESSIVELY for readability over verbatim accuracy:
   - Readability is the most important thing!!
   - Remove ALL conversational artifacts (yeah, so, I mean, etc.)
   - Remove ALL filler words (um, uh, like, you know)
   - Remove false starts and self-corrections completely
   - Remove redundant phrases and hesitations
   - Convert any indirect or rambling responses into direct statements
   - Break up run-on sentences into clear, concise statements
   - Maintain natural conversation flow while prioritizing clarity and directness

3. Format the output consistently:
   - Keep the "Speaker X 00:00:00" format (no brackets, no other formatting)
   - Add TWO line breaks between speaker/timestamp and the text
   - Use proper punctuation and capitalization
   - Add paragraph breaks for topic changes
   - When you add paragraph breaks between the same speaker's remarks, no need to restate the speaker attribution
   - Preserve distinct speaker turns

Example input:
Speaker A 00:01:15

Um, yeah, so like, what I was thinking was, you know, when we look at the data, the data shows us that, uh, there's this pattern, this pattern that keeps coming up again and again in the results.

Example output:
Speaker A 00:01:15

When we look at the data, we see a consistent pattern in the results.

When we examine the second part of the analysis, it reveals a completely different finding.

Enhance the following transcript, starting directly with the speaker format:
"""

    response = model.generate_content(
        [prompt, chunk_text, {"mime_type": "audio/mp3", "data": audio_segment.read()}]
    )
    return response.text


def create_chunks(utterances, target_tokens=2000):
    """Create chunks of utterances that fit within token limits"""
    chunks = []
    current_chunk = []
    current_start = None
    current_end = None

    for utterance in utterances:
        # Start new chunk if this is first utterance
        if not current_chunk:
            current_start = float(utterance.start) / 1000  # Convert ms to seconds
            current_chunk = [utterance]
            current_end = float(utterance.end) / 1000  # Convert ms to seconds
        # Check if adding this utterance would exceed token limit
        elif (
            len(" ".join(u.text for u in current_chunk)) + len(utterance.text)
        ) / 4 > target_tokens:
            # Save current chunk and start new one
            chunks.append(
                {
                    "utterances": current_chunk,
                    "start": current_start,
                    "end": current_end,
                }
            )
            current_chunk = [utterance]
            current_start = float(utterance.start) / 1000
            current_end = float(utterance.end) / 1000
        else:
            # Add to current chunk
            current_chunk.append(utterance)
            current_end = float(utterance.end) / 1000

    # Add final chunk
    if current_chunk:
        chunks.append(
            {"utterances": current_chunk, "start": current_start, "end": current_end}
        )

    return chunks


def process_chunk(chunk_data):
    """Process a single chunk with Gemini"""
    audio_path, chunk = chunk_data
    chunk_text = format_transcript(chunk["utterances"])
    audio_segment = get_audio_segment(audio_path, chunk["start"], chunk["end"])
    return enhance_transcript(chunk_text, audio_segment)


def process_audio(audio_path):
    """Main processing pipeline"""
    print("Stage 1: Getting raw transcript from AssemblyAI...")
    transcript_data = get_transcript(audio_path)

    print("Stage 2: Processing in chunks...")
    chunks = create_chunks(transcript_data)

    # Get original transcript
    original_chunks = [format_transcript(chunk["utterances"]) for chunk in chunks]
    original_transcript = "\n".join(original_chunks)

    # Process enhanced versions in parallel
    print(f"Stage 3: Enhancing {len(chunks)} chunks in parallel...")
    chunk_data = [(audio_path, chunk) for chunk in chunks]

    # Use max_workers=None to allow as many threads as needed
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        # Submit all tasks and store with their original indices
        future_to_index = {
            executor.submit(process_chunk, data): i for i, data in enumerate(chunk_data)
        }

        # Create a list to store results in order
        enhanced_chunks = [None] * len(chunks)

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            print(f"Completed chunk {index + 1}/{len(chunks)}")
            enhanced_chunks[index] = future.result()

    enhanced_transcript = "\n".join(enhanced_chunks)

    # Write transcripts to files
    with open("autogenerated-transcript.md", "w", encoding="utf-8") as f:
        f.write(original_transcript)

    with open("transcript.md", "w", encoding="utf-8") as f:
        f.write(enhanced_transcript)

    print("\nTranscripts have been saved to:")
    print("- autogenerated-transcript.md")
    print("- transcript.md")


def get_audio_segment(audio_path, start_time, end_time):
    """Extract audio segment between start and end times"""
    audio = AudioSegment.from_file(audio_path)
    start_ms = int(float(start_time) * 1000)
    end_ms = int(float(end_time) * 1000)
    return audio[start_ms:end_ms].export(format="mp3")


def main():
    parser = argparse.ArgumentParser(
        description="Generate enhanced transcripts from audio files"
    )
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: File '{args.audio_file}' not found")
        return

    try:
        process_audio(args.audio_file)
    except Exception as e:
        print(f"Error processing audio: {str(e)}")


if __name__ == "__main__":
    main()
