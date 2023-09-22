import os
import wget
import zipfile


# Clone necessary repositories
os.system("git clone https://github.com/vinthony/video-retalking.git")
os.system("git clone https://github.com/davisking/dlib.git")
os.system("git clone https://github.com/openai/whisper.git")

# Install dlib
os.system("cd dlib && python setup.py install")

# Create checkpoints directory in video-retalking
os.makedirs("./video-retalking/checkpoints", exist_ok=True)

# Download model checkpoints and other files
model_urls = [
    "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/30_net_gen.pth",
    "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/BFM.zip",
    "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/DNet.pt",
    "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ENet.pth",
    "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/expression.mat",
    "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/face3d_pretrain_epoch_20.pth",
    "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GFPGANv1.3.pth",
    "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GPEN-BFR-512.pth",
    "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/LNet.pth",
    "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ParseNet-latest.pth",
    "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/shape_predictor_68_face_landmarks.dat"
]

for url in model_urls:
    wget.download(url, out="./video-retalking/checkpoints")

# Unzip files
with zipfile.ZipFile("./video-retalking/checkpoints/BFM.zip", 'r') as zip_ref:
    zip_ref.extractall("./video-retalking/checkpoints")

# Install Python packages
#os.system("pip install basicsr==1.4.2 face-alignment==1.3.4 kornia==0.5.1 ninja==1.10.2.3 einops==0.4.1 facexlib==0.2.5 librosa==0.9.2 build")

