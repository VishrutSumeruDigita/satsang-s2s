#!/bin/bash

# Speech-to-Speech Translation Project Dependencies Installation
# This script installs all required dependencies for the project

echo "🚀 Installing Speech-to-Speech Translation Dependencies..."

# Create conda environment with Python 3.11.13
echo "🐍 Creating conda environment with Python 3.11.13..."
conda create -n speech-to-speech python=3.11.13 -y

# Activate the environment
echo "🔄 Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate speech-to-speech

echo "✅ Conda environment 'speech-to-speech' created and activated!"

# Install system dependencies for audio processing
echo "📦 Installing system dependencies (ffmpeg)..."
sudo apt update
sudo apt install -y ffmpeg

# Install main project dependencies
echo "🐍 Installing main Python dependencies..."
pip install -r requirements.txt

# Install Python dependencies for YouTube audio processing
echo "📺 Installing YouTube audio processing dependencies..."
pip install yt-dlp pydub librosa

# Clone the github repository and navigate to the project directory.
echo "📋 Installing IndicTransToolkit..."
git clone https://github.com/VarunGumma/IndicTransToolkit
cd IndicTransToolkit
pip install .
cd ..

echo "✅ Installation complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Activate the environment: conda activate speech-to-speech"
echo "   2. Initialize models: python orchestrator.py"
echo "   3. Test installation: python -c \"import torch, transformers, yt_dlp; print('✅ All dependencies installed successfully!')\""
echo ""
echo "🎵 To test YouTube audio processing:"
echo "   cd stream_processing && python youtube_to_aud.py"
echo "   (Audio chunks will be saved in the audio_inpts directory)"
echo ""
echo "🔄 To use bulk translation:"
echo "   python bulk_translate.py" 