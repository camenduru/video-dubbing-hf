import tempfile
import gradio as gr
import subprocess
import os, stat
from googletrans import Translator
from TTS.api import TTS
import ffmpeg
import whisper
from scipy.signal import wiener
import soundfile as sf
from pydub import AudioSegment
import numpy as np
import librosa
from zipfile import ZipFile

os.environ["COQUI_TOS_AGREED"] = "1"

ZipFile("ffmpeg.zip").extractall()
st = os.stat('ffmpeg')
os.chmod('ffmpeg', st.st_mode | stat.S_IEXEC)

def process_video(video, high_quality, target_language):
    with tempfile.TemporaryDirectory() as temp_dir:
        output_filename = os.path.join(temp_dir, "resized_video.mp4")
        
        if high_quality:
            ffmpeg.input(video).output(output_filename, vf='scale=-1:720').run()
            video_path = output_filename
        else:
            video_path = video

        if not os.path.exists(video_path):
            return f"Error: {video_path} does not exist."

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_output = os.path.join(temp_dir, "output_audio.wav")
        
        try:
            ffmpeg.input(video_path).output(audio_output, acodec='pcm_s24le', ar=48000, map='a').run()
        except ffmpeg.Error as e:
            return f"FFmpeg error: {e.stderr.decode('utf-8')}"


        y, sr = sf.read("output_audio.wav")
        y = y.astype(np.float32)
        y_denoised = wiener(y)
        sf.write("output_audio_denoised.wav", y_denoised, sr)
    
        sound = AudioSegment.from_file("output_audio_denoised.wav", format="wav")
        sound = sound.apply_gain(0)  # Reduce gain by 5 dB
        sound = sound.low_pass_filter(3000).high_pass_filter(100)
        sound.export("output_audio_processed.wav", format="wav")
    
        shell_command = f"ffmpeg -y -i output_audio_processed.wav -af lowpass=3000,highpass=100 output_audio_final.wav".split(" ")
        subprocess.run([item for item in shell_command], capture_output=False, text=True, check=True)
    
        model = whisper.load_model("base")
        result = model.transcribe("output_audio_final.wav")
        whisper_text = result["text"]
        whisper_language = result['language']
    
        language_mapping = {'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Italian': 'it', 'Portuguese': 'pt', 'Polish': 'pl', 'Turkish': 'tr', 'Russian': 'ru', 'Dutch': 'nl', 'Czech': 'cs', 'Arabic': 'ar', 'Chinese (Simplified)': 'zh-cn'}
        target_language_code = language_mapping[target_language]
        translator = Translator()
        translated_text = translator.translate(whisper_text, src=whisper_language, dest=target_language_code).text
    
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1")
        tts.to('cuda')  # Replacing deprecated gpu=True
        tts.tts_to_file(translated_text, speaker_wav='output_audio_final.wav', file_path="output_synth.wav", language=target_language_code)
    
        pad_top = 0
        pad_bottom = 15
        pad_left = 0
        pad_right = 0
        rescaleFactor = 1
    
        video_path_fix = video_path
    
        cmd = f"python Wav2Lip/inference.py --checkpoint_path '/Wav2Lip/checkpoints/wav2lip_gan.pth' --face {shlex.quote(video_path_fix)} --audio 'output_synth.wav' --pads {pad_top} {pad_bottom} {pad_left} {pad_right} --resize_factor {rescaleFactor} --nosmooth --outfile 'output_video.mp4'"
        subprocess.run(cmd, shell=True)
        # Debugging Step 3: Check if output video exists
        if not os.path.exists("output_video.mp4"):
            return "Error: output_video.mp4 was not generated."
    
        return "output_video.mp4"

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

iface.launch()