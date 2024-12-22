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
model = generativeai.GenerativeModel("gemini-2.0-flash-exp")


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
                # Normalize spacing: single newline after timestamp, text joined with single spaces
                section = f"Speaker {current_speaker} {timestamp}\n{' '.join(current_text).strip()}"
                formatted_sections.append(section)
                current_text = []

            # Start new section
            current_speaker = utterance.speaker
            current_start = utterance.start

        current_text.append(utterance.transcript.strip())

    # Add the final section
    if current_text:
        timestamp = format_timestamp(current_start)
        section = (
            f"Speaker {current_speaker} {timestamp}\n{' '.join(current_text).strip()}"
        )
        formatted_sections.append(section)

    return "\n\n".join(formatted_sections)


def enhance_transcript(chunk_text, audio_segment):
    """Enhance transcript using Gemini AI with both text and audio"""
    prompt = """As a professional transcript editor, enhance this transcript for maximum readability while preserving accuracy. 

Key Instructions:
1. Correct transcription errors using the audio
2. Format for readability:
   - Remove filler words (e.g., "um", "like", "you know")
   - Remove repetitions and false starts
   - Break into clear paragraphs
   - Add punctuation and quotation marks
3. Maintain exact speaker names and timestamps
4. Fix speaker attribution errors by:
   - Using the audio to verify who is actually speaking
   - Moving text to the correct speaker's section if misattributed
   - Never combining multiple speakers' text into one section
   - These often happen at the end of a speaker's section or the beginning of the next speaker's section. Be aware of this!

Example:

<Original>
Dwarkesh 0:13:37
Let's let's go to World War 1 and World War 2. So I would, you know, I, I had on the, um, the
A couple of months ago, I interviewed the biographer of Churchill, Andrew Roberts, and we, as you discussed in your book, and he discusses, you know, Churchill was the sort of technological visionary, and that's the part of him that isn't talked about often. Um,
Of you maybe talk a little bit about what Churchill did and how he saw the power of oil. I think Churchill was

Daniel Yergin 0:14:04
the first Lord of the Admiralty, and he saw that if you can convert all the naval ships at that time ran on coal, which means you had to have people on board shoveling coal, and it took a long time to get the coal on board, and if you switch to oil, you would have faster, uh, the ships would be faster, they wouldn't need to take the same time. They wouldn't need to carry the same people. And so he made
The decision, obviously others like Admiral Jackie Fisher were pushing him to convert the Royal Navy to to oil and people saying this is treacherous because we'll depend upon oil from far away, from Persia, uh, rather than Welsh coal and uh he said, um, you know, he said, um this is the prize of the venture. That's where I got my title from originally it was going to be called The Prize of the Venture, because that's what he said and then I just made it the prize, but uh, he saw that.
During, uh, uh, World War 2, World War 1, he promoted another uh uh military development, um, I'm forgetting what it was called initially, but it eventually became known as the tank. I mean, so he really did kind of constantly push technology.
Why I don't know. I mean, he was actually, you know, was not, he was not educated, uh, as that he was educated and, you know, in the sort of classic I wrote so well, uh, but, uh, he understood technology and that you had a kind of constantly push for advantage.

</Original>

<Enhanced>
Dwarkesh Patel 00:13:37

Let's go to World War I and World War II. A couple months ago, I interviewed the biographer of Churchill, Andrew Roberts. As you discuss in your book, he discusses that Churchill was this sort of technological visionary and how that's a side of him that isn't talked about often. Maybe talk a little bit about what Churchill did and how he saw the power of oil.

Daniel Yergin 00:14:04 

Churchill was the First Lord of the Admiralty. All the naval ships at that time ran on coal, which means you had to have people on board shoveling coal. It took a long time to get the coal on board. If you switched to oil, the ships would be faster. They wouldn't need to take the same time. They wouldn't need to carry the same people. 

So he made the decision—obviously others like Admiral Jackie Fisher were pushing him—to convert the Royal Navy to oil. People were saying this is treacherous because we'll depend upon oil from far away, from Persia, rather than Welsh coal. He said, "This is the prize of the venture." That's where I got my title from. Originally it was going to be called "The Prize of the Venture" because that's what he said. Then I just made it The Prize. 

During World War I, he promoted another military development. I'm forgetting what it was called initially, but it eventually became known as the tank. He really did constantly push technology. Why? I don't know. He was not educated like that. He was educated in the classic sense. That's why he wrote so well. But he understood technology and that you had to constantly push for advantage.

</Enhanced>

Notice how the enhanced version:
1. Maintains exact speaker names and timestamps
2. Removes filler words and repetitions
3. Breaks long passages into logical paragraphs
4. Adds proper punctuation and quotation marks
6. Corrects speaker attribution errors.

Output only the enhanced transcript, maintaining speaker names and timestamps exactly as given.

"""

    response = model.generate_content(
        [prompt, chunk_text, {"mime_type": "audio/mp3", "data": audio_segment.read()}]
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
