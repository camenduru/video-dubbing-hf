import gradio as gr
import subprocess
import os
from googletrans import Translator
from TTS.api import TTS
from IPython.display import Audio, display
import ffmpeg
import whisper 

def process_video(video, high_quality, target_language):
    try:
        output_filename = "resized_video.mp4"
        if high_quality:
            ffmpeg.input(video).output(output_filename, vf='scale=-1:720').run()
            video_path = output_filename
        else:
            video_path = video

        ffmpeg.input(video_path).output('output_audio.wav', acodec='pcm_s24le', ar=48000, map='a').run()

        model = whisper.load_model("base")
        result = model.transcribe("output_audio.wav")
        whisper_text = result["text"]
        whisper_language = result['language']

        language_mapping = {
            'English': 'en',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Polish': 'pl',
            'Turkish': 'tr',
            'Russian': 'ru',
            'Dutch': 'nl',
            'Czech': 'cs',
            'Arabic': 'ar',
            'Chinese (Simplified)': 'zh-cn'
        }
        target_language_code = language_mapping[target_language]
        translator = Translator()
        translated_text = translator.translate(whisper_text, src=whisper_language, dest=target_language_code).text

        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1", gpu=True)
        tts.tts_to_file(translated_text, speaker_wav='output_audio.wav', file_path="output_synth.wav", language=target_language_code)

        subprocess.run(f"python inference.py --face {video_path} --audio 'output_synth.wav' --outfile 'output_high_qual.mp4'", shell=True)

        return "output_high_qual.mp4"

    except Exception as e:
        return str(e)

iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(),
        gr.inputs.Checkbox(label="High Quality"),
        gr.inputs.Dropdown(choices=["English", "Spanish", "French", "German", "Italian", "Portuguese", "Polish", "Turkish", "Russian", "Dutch", "Czech", "Arabic", "Chinese (Simplified)"], label="Target Language for Dubbing")
    ],
    outputs=gr.outputs.File(),
    live=False
)

iface.launch(share=True)
