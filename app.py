import gradio as gr
import subprocess, os, uuid, ffmpeg, shlex, torch
from googletrans import Translator
from TTS.api import TTS
from faster_whisper import WhisperModel
import numpy as np
from zipfile import ZipFile
from tqdm import tqdm

os.environ["COQUI_TOS_AGREED"] = "1"

#Whisper
model_size = "small"
model = WhisperModel(model_size, device="cuda", compute_type="int8")

def process_video(radio, video, target_language):
    if target_language is None:
        return gr.Interface.Warnings("Please select a Target Language for Dubbing.")
        
    run_uuid = uuid.uuid4().hex[:6]
    
    output_filename = f"{run_uuid}_resized_video.mp4"
    ffmpeg.input(video).output(output_filename, vf='scale=-2:720').run()

    video_path = output_filename
    
    if not os.path.exists(video_path):
        return f"Error: {video_path} does not exist."

    ffmpeg.input(video_path).output(f"{run_uuid}_output_audio.wav", acodec='pcm_s24le', ar=48000, map='a').run()

    shell_command = f"ffmpeg -y -i {run_uuid}_output_audio.wav -af lowpass=3000,highpass=100 {run_uuid}_output_audio_final.wav".split(" ")
    subprocess.run([item for item in shell_command], capture_output=False, text=True, check=True)

    segments, info = model.transcribe(f"{run_uuid}_output_audio_final.wav", beam_size=5)
    whisper_text = " ".join(segment.text for segment in segments)
    whisper_language = info.language
    print(whisper_text)

    language_mapping = {'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Italian': 'it', 'Portuguese': 'pt', 'Polish': 'pl', 'Turkish': 'tr', 'Russian': 'ru', 'Dutch': 'nl', 'Czech': 'cs', 'Arabic': 'ar', 'Chinese (Simplified)': 'zh-cn'}
    target_language_code = language_mapping[target_language]
    translator = Translator()
    try:
        translated_text = translator.translate(whisper_text, src=whisper_language, dest=target_language_code).text
        print(translated_text)
    except AttributeError as e:
        print("Failed to translate text. Likely an issue with token extraction in the Google Translate API.")
        translated_text = "Translation failed due to API issue."

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1")
    tts.to('cuda')
    tts.tts_to_file(translated_text, speaker_wav=f"{run_uuid}_output_audio_final.wav", file_path=f"{run_uuid}_output_synth.wav", language=target_language_code)

    pad_top = 0
    pad_bottom = 15
    pad_left = 0
    pad_right = 0
    rescaleFactor = 1

    video_path_fix = video_path

    cmd = f"python Wav2Lip/inference.py --checkpoint_path 'Wav2Lip/checkpoints/wav2lip_gan.pth' --face {shlex.quote(video_path_fix)} --audio '{run_uuid}_output_synth.wav' --pads {pad_top} {pad_bottom} {pad_left} {pad_right} --resize_factor {rescaleFactor} --nosmooth --outfile '{run_uuid}_output_video.mp4'"
    subprocess.run(cmd, shell=True)

    if not os.path.exists(f"{run_uuid}_output_video.mp4"):
        raise FileNotFoundError(f"Error: {run_uuid}_output_video.mp4 was not generated.")

    output_video_path = f"{run_uuid}_output_video.mp4"

    # Cleanup: Delete all generated files except the final output video
    files_to_delete = [
        f"{run_uuid}_resized_video.mp4",
        f"{run_uuid}_output_audio.wav",
        f"{run_uuid}_output_audio_final.wav",
        f"{run_uuid}_output_synth.wav"
    ]
    for file in files_to_delete:
        try:
            os.remove(file)
        except FileNotFoundError:
            print(f"File {file} not found for deletion.")

    return output_video_path
    
    
def swap(radio):
    if(radio == "Upload"):
        return gr.update(source="upload")
    else:
        return gr.update(source="webcam")
        
video = gr.Video()
radio = gr.Radio(["Upload", "Record"], show_label=False)
iface = gr.Interface(
    fn=process_video,
    inputs=[
        radio,
        video,
        gr.Dropdown(choices=["English", "Spanish", "French", "German", "Italian", "Portuguese", "Polish", "Turkish", "Russian", "Dutch", "Czech", "Arabic", "Chinese (Simplified)"], label="Target Language for Dubbing")
    ],
    outputs=gr.Video(),
    live=False,
    title="AI Video Dubbing",
    description="""This tool was developed by [@artificialguybr](https://twitter.com/artificialguybr) using entirely open-source tools. Special thanks to Hugging Face for the GPU support. Thanks [@yeswondwer](https://twitter.com/@yeswondwerr) for original code.

    **Note:**
    - Video limit is 1 minute.
    - Generation may take up to 5 minutes.
    - The tool uses open-source models for all operations.
    - Quality can be improved but would require more processing time per video.""",
    allow_flagging=False
)
with gr.Blocks() as demo:
    iface.render()
    radio.change(swap, inputs=[radio], outputs=video)
demo.queue(concurrency_count=2, max_size=15)
demo.launch(share=True, debug=True)