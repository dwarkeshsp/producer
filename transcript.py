import gradio as gr
from deepgram import DeepgramClient, PrerecordedOptions
from google import generativeai
import os
from pydub import AudioSegment

# Initialize API clients
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

dg_client = DeepgramClient(DEEPGRAM_API_KEY)
generativeai.configure(api_key=GOOGLE_API_KEY)
model = generativeai.GenerativeModel("gemini-exp-1206")


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    h = int(float(seconds)) // 3600
    m = (int(float(seconds)) % 3600) // 60
    s = int(float(seconds)) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_transcript(audio_path):
    """Get transcript from Deepgram with speaker diarization"""
    with open(audio_path, "rb") as audio:
        options = PrerecordedOptions(
            smart_format=True,
            diarize=True,
            utterances=True,
            model="nova-2",
            language="en-US",
        )
        response = dg_client.listen.rest.v("1").transcribe_file(
            {"buffer": audio, "mimetype": "audio/mp3"}, options
        )
        return response.results.utterances


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
                timestamp = format_timestamp(current_start)
                # Add double line break after speaker/timestamp
                section = f"Speaker {current_speaker + 1} {timestamp}\n\n{' '.join(current_text).strip()}"
                formatted_sections.append(section)
                current_text = []

            # Start new section
            current_speaker = utterance.speaker
            current_start = utterance.start

        current_text.append(utterance.transcript.strip())

    # Add the final section
    if current_text:
        timestamp = format_timestamp(current_start)
        section = f"Speaker {current_speaker + 1} {timestamp}\n\n{' '.join(current_text).strip()}"
        formatted_sections.append(section)

    return "\n\n".join(formatted_sections)


def enhance_transcript(chunk_text, audio_segment):
    """Enhance transcript using Gemini AI with both text and audio"""
    prompt = """You are an expert transcript editor. Your task is to enhance this transcript for maximum readability while maintaining the core message.

IMPORTANT: Respond ONLY with the enhanced transcript. Do not include any explanations, headers, or phrases like "Here is the transcript."

Please:
1. Fix speaker attribution errors, especially at segment boundaries. Watch for incomplete thoughts that were likely from the previous speaker.

2. Optimize for readability over verbatim accuracy:
   - Remove filler words (um, uh, like, you know)
   - Eliminate false starts and repetitions
   - Convert rambling sentences into clear, concise statements
   - Break up run-on sentences into shorter ones
   - Maintain natural conversation flow while improving clarity

3. Format the output consistently:
   - Keep the "Speaker X 00:00:00" format (no brackets, no other formatting)
   - Add TWO line breaks between speaker/timestamp and the text
   - Use proper punctuation and capitalization
   - Add paragraph breaks for topic changes
   - When you add paragraph breaks between the same speaker's remarks, no need to restate the speaker attribution
   - Preserve distinct speaker turns

Example input:
Speaker 1 00:01:15

Um, yeah, so like, what I was thinking was, you know, when we look at the data, the data shows us that, uh, there's this pattern, this pattern that keeps coming up again and again in the results.

Example output:
Speaker 1 00:01:15

When we look at the data, we see a consistent pattern in the results.

And when we examine the second part of the analysis, it reveals a completely different finding.

Enhance the following transcript, starting directly with the speaker format:
"""

    response = model.generate_content(
        [prompt, {"mime_type": "audio/mp3", "data": audio_segment.read()}]
    )
    return response.text


def create_chunks(utterances, target_tokens=7500):
    """Create chunks of utterances that fit within token limits"""
    chunks = []
    current_chunk = []
    current_start = None
    current_end = None

    for utterance in utterances:
        # Start new chunk if this is first utterance
        if not current_chunk:
            current_start = utterance.start
            current_chunk = [utterance]
            current_end = utterance.end
        # Check if adding this utterance would exceed token limit
        elif (float(utterance.end) - float(current_start)) * 25 > target_tokens:
            # Save current chunk and start new one
            chunks.append(
                {
                    "utterances": current_chunk,
                    "start": current_start,
                    "end": current_end,
                }
            )
            current_chunk = [utterance]
            current_start = utterance.start
            current_end = utterance.end
        else:
            # Add to current chunk
            current_chunk.append(utterance)
            current_end = utterance.end

    # Add final chunk
    if current_chunk:
        chunks.append(
            {"utterances": current_chunk, "start": current_start, "end": current_end}
        )

    return chunks


def process_audio(audio_path):
    """Main processing pipeline"""
    print("Stage 1: Getting raw transcript from Deepgram...")
    transcript_data = get_transcript(audio_path)

    print("Stage 2: Processing in chunks...")
    chunks = create_chunks(transcript_data)
    original_chunks = []
    enhanced_chunks = []

    for i, chunk in enumerate(chunks):
        # Get original chunk
        chunk_text = format_transcript(chunk["utterances"])
        original_chunks.append(chunk_text)

        # Process enhanced version
        print(f"Processing chunk {i+1} of {len(chunks)}...")
        audio_segment = get_audio_segment(audio_path, chunk["start"], chunk["end"])
        enhanced_chunk = enhance_transcript(chunk_text, audio_segment)
        enhanced_chunks.append(enhanced_chunk)

    return "\n".join(original_chunks), "\n".join(enhanced_chunks)


def handle_upload(audio):
    """Handle Gradio interface uploads"""
    if audio is None:
        return "Please upload an audio file.", "Please upload an audio file."

    try:
        original, enhanced = process_audio(audio)
        return original, enhanced
    except Exception as e:
        error_msg = f"Error processing audio: {str(e)}"
        return error_msg, error_msg


def get_audio_segment(audio_path, start_time, end_time):
    """Extract audio segment between start and end times"""
    audio = AudioSegment.from_file(audio_path)
    start_ms = int(float(start_time) * 1000)
    end_ms = int(float(end_time) * 1000)
    return audio[start_ms:end_ms].export(format="mp3")


# Create Gradio interface
iface = gr.Interface(
    fn=handle_upload,
    inputs=gr.Audio(type="filepath"),
    outputs=[
        gr.Textbox(label="Original Transcript"),
        gr.Textbox(label="Enhanced Transcript"),
    ],
    title="Audio Transcript Enhancement",
    description="Upload an MP3 file to get both the original and enhanced transcripts using Deepgram and Gemini.",
    cache_examples=False,
)

if __name__ == "__main__":
    iface.launch()
