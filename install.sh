#!/bin/bash

# Clone necessary repositories
git clone https://github.com/vinthony/video-retalking.git
git clone https://github.com/davisking/dlib.git

# Install dlib
cd dlib && python setup.py install
cd ..

# Create checkpoints directory in video-retalking
mkdir -p ./video-retalking/checkpoints

# Download model checkpoints and other files
wget -P ./video-retalking/checkpoints https://github.com/vinthony/video-retalking/releases/download/v0.0.1/30_net_gen.pth
wget -P ./video-retalking/checkpoints https://github.com/vinthony/video-retalking/releases/download/v0.0.1/BFM.zip
wget -P ./video-retalking/checkpoints https://github.com/vinthony/video-retalking/releases/download/v0.0.1/DNet.pt
wget -P ./video-retalking/checkpoints https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ENet.pth
wget -P ./video-retalking/checkpoints https://github.com/vinthony/video-retalking/releases/download/v0.0.1/expression.mat
wget -P ./video-retalking/checkpoints https://github.com/vinthony/video-retalking/releases/download/v0.0.1/face3d_pretrain_epoch_20.pth
wget -P ./video-retalking/checkpoints https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GFPGANv1.3.pth
wget -P ./video-retalking/checkpoints https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GPEN-BFR-512.pth
wget -P ./video-retalking/checkpoints https://github.com/vinthony/video-retalking/releases/download/v0.0.1/LNet.pth
wget -P ./video-retalking/checkpoints https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ParseNet-latest.pth
wget -P ./video-retalking/checkpoints https://github.com/vinthony/video-retalking/releases/download/v0.0.1/RetinaFace-R50.pth
wget -P ./video-retalking/checkpoints https://github.com/vinthony/video-retalking/releases/download/v0.0.1/shape_predictor_68_face_landmarks.dat

# Unzip files
unzip -d ./video-retalking/checkpoints ./video-retalking/checkpoints/BFM.zip