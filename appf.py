import gradio as gr
import subprocess
import whisper
from googletrans import Translator
import asyncio
import edge_tts
import os

# Extract and Transcribe Audio
def extract_and_transcribe_audio(video_path):
    ffmpeg_command = f"ffmpeg -i '{video_path}' -acodec pcm_s24le -ar 48000 -q:a 0 -map a -y 'output_audio.wav'"
    subprocess.run(ffmpeg_command, shell=True)
    model = whisper.load_model("base")
    result = model.transcribe("output_audio.wav")
    return result["text"], result['language']

# Translate Text
def translate_text(whisper_text, whisper_language, target_language):
    language_mapping = {
        'English': 'en',
        'Spanish': 'es',
        # ... (other mappings)
    }
    target_language_code = language_mapping[target_language]
    translator = Translator()
    translated_text = translator.translate(whisper_text, src=whisper_language, dest=target_language_code).text
    return translated_text

# Generate Voice
async def generate_voice(translated_text, target_language):
    VOICE_MAPPING = {
        'English': 'en-GB-SoniaNeural',
        'Spanish': 'es-ES-PabloNeural',
        # ... (other mappings)
    }
    voice = VOICE_MAPPING[target_language]
    communicate = edge_tts.Communicate(translated_text, voice)
    await communicate.save("output_synth.wav")
    return "output_synth.wav"

# Generate Lip-synced Video (Placeholder)
def generate_lip_synced_video(video_path, output_audio_path):
    # Your lip-synced video generation code here
    # ...
    return "output_high_qual.mp4"

# Main function to be called by Gradio
def process_video(video, target_language):
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(video.read())

    # Step 1: Extract and Transcribe Audio
    whisper_text, whisper_language = extract_and_transcribe_audio(video_path)

    # Step 2: Translate Text
    translated_text = translate_text(whisper_text, whisper_language, target_language)

    # Step 3: Generate Voice
    loop = asyncio.get_event_loop()
    output_audio_path = loop.run_until_complete(generate_voice(translated_text, target_language))

    # Step 4: Generate Lip-synced Video
    output_video_path = generate_lip_synced_video(video_path, output_audio_path)

    return output_video_path

# Gradio Interface
iface = gr.Interface(
    fn=process_video, 
    inputs=["file", gr.Interface.Component(type="dropdown", choices=["English", "Spanish"])], 
    outputs="file",
    live=False
)
iface.launch()
