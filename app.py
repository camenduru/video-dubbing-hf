import tempfile
import gradio as gr
import subprocess
import os, stat
import uuid
from googletrans import Translator
from TTS.api import TTS
import ffmpeg
from faster_whisper import WhisperModel
from scipy.signal import wiener
import soundfile as sf
from pydub import AudioSegment
import numpy as np
import librosa
from zipfile import ZipFile
import shlex
import cv2
import torch
import torchvision
from tqdm import tqdm
from numba import jit
import threading
import time
import GPUtil

os.environ["COQUI_TOS_AGREED"] = "1"

ZipFile("ffmpeg.zip").extractall()
st = os.stat('ffmpeg')
os.chmod('ffmpeg', st.st_mode | stat.S_IEXEC)

# Initialize peak usage variables
peak_gpu_usage = 0.0
peak_vram_usage = 0.0

# Monitoring function
def monitor_gpu_usage():
    global peak_gpu_usage, peak_vram_usage
    while True:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            peak_gpu_usage = max(peak_gpu_usage, gpu.load)
            peak_vram_usage = max(peak_vram_usage, gpu.memoryUsed)
        time.sleep(1)  # Check every second

# Start the monitoring thread
monitor_thread = threading.Thread(target=monitor_gpu_usage)
monitor_thread.daemon = True
monitor_thread.start()

#Whisper
model_size = "small"
model = WhisperModel(model_size, device="cuda", compute_type="int8")

def process_video(radio, video, target_language):
    # Check video duration
    video_info = ffmpeg.probe(video)
    video_duration = float(video_info['streams'][0]['duration'])
    if video_duration > 90:
        return gr.Interface.Warnings("Video duration exceeds 1 minute and 30 seconds. Please upload a shorter video.")

    run_uuid = uuid.uuid4().hex[:6]
    
    output_filename = f"{run_uuid}_resized_video.mp4"
    ffmpeg.input(video).output(output_filename, vf='scale=-1:720').run()
    video_path = output_filename
    
    #Time tracking
    start_time = time.time()
    if not os.path.exists(video_path):
        return f"Error: {video_path} does not exist."

    ffmpeg.input(video_path).output(f"{run_uuid}_output_audio.wav", acodec='pcm_s24le', ar=48000, map='a').run()

    #y, sr = sf.read(f"{run_uuid}_output_audio.wav")
    #y = y.astype(np.float32)
    #y_denoised = wiener(y)
    #sf.write(f"{run_uuid}_output_audio_denoised.wav", y_denoised, sr)

    #sound = AudioSegment.from_file(f"{run_uuid}_output_audio_denoised.wav", format="wav")
    #sound = sound.apply_gain(0)
    #sound = sound.low_pass_filter(3000).high_pass_filter(100)
    #sound.export(f"{run_uuid}_output_audio_processed.wav", format="wav")

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
        f"{run_uuid}_output_audio_denoised.wav",
        f"{run_uuid}_output_audio_processed.wav",
        f"{run_uuid}_output_audio_final.wav",
        f"{run_uuid}_output_synth.wav"
    ]

    for file in files_to_delete:
        try:
            os.remove(file)
        except FileNotFoundError:
            print(f"File {file} not found for deletion.")

    # Stop the timer
    end_time = time.time()
    
    # Calculate and print the time taken
    time_taken = end_time - start_time
    print(f"Time taken to process video: {time_taken:.2f} seconds")

    # Display peak usages at the end
    print(f"Peak GPU usage: {peak_gpu_usage * 100}%")
    print(f"Peak VRAM usage: {peak_vram_usage}MB")
    
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
demo.launch()