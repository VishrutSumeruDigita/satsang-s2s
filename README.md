# Hindi Speech-to-Speech Translation Pipeline

A complete speech-to-speech translation system for Indian languages using local AI models. Translate Hindi speech to other Indian languages without relying on external APIs.

## 🌟 Features

- **Speech-to-Text (ASR)**: Hindi audio transcription using Whisper
- **Translation**: Hindi to Indian and international languages using NLLB-200
- **Text-to-Speech (TTS)**: Generate speech in target languages
- **Local Processing**: All models run locally (no external APIs)
- **GPU Support**: Optimized for CUDA-enabled GPUs
- **Modern UI**: Beautiful Streamlit interface
- **Audio Recording**: Built-in microphone recording
- **File Upload**: Support for various audio formats

## 🗣️ Supported Languages

### 🇮🇳 Indian Languages
- Tamil (ta)
- Telugu (te)
- Kannada (kn)
- Malayalam (ml)
- Gujarati (gu)
- Punjabi (pa)
- Bengali (bn)
- Odia (or)
- Assamese (as)
- Marathi (mr)

### 🌍 International Languages
- English (en)
- French (fr)
- German (de)
- Russian (ru)
- Chinese Simplified (zh)
- Japanese (ja)

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space for models

### Setup

1. **Clone/Create the project directory:**
   ```bash
   cd speech-to-speech
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch with CUDA support (if you have GPU):**
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## 🚀 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Pipeline Programmatically

```python
from speech_pipeline import SpeechToSpeechPipeline

# Initialize pipeline for Tamil translation
pipeline = SpeechToSpeechPipeline(target_language="ta")

# Process audio file
hindi_text, tamil_text, output_audio = pipeline.process_speech_to_speech("input.wav")

print(f"Hindi: {hindi_text}")
print(f"Tamil: {tamil_text}")
print(f"Output audio: {output_audio}")
```

## 🎯 How It Works

### 1. Speech-to-Text (ASR)
- Uses OpenAI's Whisper model for Hindi speech recognition
- Converts Hindi audio to text with high accuracy
- Supports various audio formats

### 2. Translation
- Uses Meta's NLLB-200 model for multilingual translation
- Translates Hindi text to target Indian language
- Supports 10+ Indian languages

### 3. Text-to-Speech (TTS)
- Uses Microsoft's SpeechT5 for speech synthesis
- Generates natural-sounding speech in target language
- Outputs high-quality audio files

## 📁 Project Structure

```
speech-to-speech/
├── app.py                 # Streamlit frontend
├── speech_pipeline.py     # Main pipeline implementation
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── venv/                 # Virtual environment
```

## 🖥️ System Requirements

### Minimum Requirements
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB free space
- Python 3.8+

### Recommended Requirements
- GPU: NVIDIA RTX 3060 or better
- RAM: 16GB+
- Storage: 20GB+ SSD
- CUDA 11.8+

## 🔧 Configuration

### Language Codes

**Indian Languages:**
- Tamil: `ta`
- Telugu: `te`
- Kannada: `kn`
- Malayalam: `ml`
- Gujarati: `gu`
- Punjabi: `pa`
- Bengali: `bn`
- Odia: `or`
- Assamese: `as`
- Marathi: `mr`

**International Languages:**
- English: `en`
- French: `fr`
- German: `de`
- Russian: `ru`
- Chinese (Simplified): `zh`
- Japanese: `ja`

### Model Configuration
You can modify the models used in `speech_pipeline.py`:

```python
# ASR Model
model_name = "openai/whisper-small"  # or whisper-medium, whisper-large

# Translation Model
model_name = "facebook/nllb-200-distilled-600M"  # or nllb-200-1.3B

# TTS Model
model_name = "microsoft/speecht5_tts"
```

## 🎮 Web Interface Usage

1. **Load Models**: Click "Load Models" to initialize AI models (one-time setup)
2. **Input Audio**: 
   - Record audio using the microphone, or
   - Upload an audio file (WAV, MP3, M4A, OGG)
3. **Select Language**: Choose target Indian language from sidebar
4. **Process**: Click "Process Speech" to get results
5. **Results**: View transcription, translation, and download generated audio

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use smaller models or reduce batch size
   # In speech_pipeline.py, change to whisper-tiny or whisper-base
   ```

2. **Model Download Issues**
   ```bash
   # Ensure stable internet connection for initial model download
   # Models are cached locally after first download
   ```

3. **Audio Recording Issues**
   ```bash
   # Check microphone permissions in browser
   # Use HTTPS for microphone access in production
   ```

### Performance Optimization

1. **GPU Memory**: Use smaller models if running out of GPU memory
2. **CPU Mode**: Set `device="cpu"` if no GPU available
3. **Model Caching**: Models are automatically cached after first download

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- OpenAI for Whisper ASR models
- Meta for NLLB translation models
- Microsoft for SpeechT5 TTS models
- Hugging Face for the Transformers library
- Streamlit for the web framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed description

---

**Note**: First run will download models (~2-5GB) and may take several minutes. Subsequent runs will be much faster as models are cached locally. 