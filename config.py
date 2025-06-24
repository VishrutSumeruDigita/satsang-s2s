"""
Configuration file for Speech-to-Speech Pipeline
"""

# Model configurations
MODELS = {
    "asr": {
        "model_name": "openai/whisper-small",  # Options: whisper-tiny, whisper-base, whisper-small, whisper-medium, whisper-large
        "device": "auto",  # auto, cuda, cpu
        "language": "hi",  # Hindi
        "task": "transcribe"
    },
    
    "translation": {
        "model_name": "facebook/nllb-200-distilled-600M",  # Options: nllb-200-distilled-600M, nllb-200-1.3B
        "device": "auto",  # auto, cuda, cpu
        "max_length": 512,
        "num_beams": 5
    },
    
    "tts": {
        "model_name": "microsoft/speecht5_tts",
        "device": "auto",  # auto, cuda, cpu
        "vocoder": "microsoft/speecht5_hifigan"
    }
}

# Audio settings
AUDIO = {
    "sample_rate": 16000,
    "max_duration": 30,  # seconds
    "supported_formats": ["wav", "mp3", "m4a", "ogg", "flac"]
}

# Language mappings for NLLB model
LANGUAGE_CODES = {
    # Source language
    "hi": "hin_Deva",  # Hindi
    
    # Indian Languages
    "ta": "tam_Taml",  # Tamil
    "te": "tel_Telu",  # Telugu
    "kn": "kan_Knda",  # Kannada
    "ml": "mal_Mlym",  # Malayalam
    "gu": "guj_Gujr",  # Gujarati
    "pa": "pan_Guru",  # Punjabi
    "bn": "ben_Beng",  # Bengali
    "or": "ory_Orya",  # Odia
    "as": "asm_Beng",  # Assamese
    "mr": "mar_Deva",  # Marathi
    "ur": "urd_Arab",  # Urdu
    "ne": "npi_Deva",  # Nepali
    "si": "sin_Sinh",  # Sinhala
    
    # International Languages
    "en": "eng_Latn",  # English
    "fr": "fra_Latn",  # French
    "de": "deu_Latn",  # German
    "ru": "rus_Cyrl",  # Russian
    "zh": "zho_Hans",  # Chinese (Simplified)
    "ja": "jpn_Jpan",  # Japanese
}

# UI settings
UI = {
    "title": "Hindi Speech-to-Speech Translation",
    "description": "Translate Hindi speech to other Indian languages using local AI models",
    "theme": {
        "primary_color": "#1e3a8a",
        "background_color": "#ffffff",
        "secondary_color": "#3b82f6"
    }
}

# Performance settings
PERFORMANCE = {
    "batch_size": 1,
    "num_workers": 2,
    "pin_memory": True,
    "max_memory_gb": 8  # Maximum GPU memory to use
}

# Default settings for different hardware configurations
HARDWARE_CONFIGS = {
    "low_end": {
        "asr_model": "openai/whisper-tiny",
        "translation_model": "facebook/nllb-200-distilled-600M",
        "device": "cpu",
        "max_memory_gb": 4
    },
    
    "mid_range": {
        "asr_model": "openai/whisper-small",
        "translation_model": "facebook/nllb-200-distilled-600M", 
        "device": "cuda",
        "max_memory_gb": 8
    },
    
    "high_end": {
        "asr_model": "openai/whisper-medium",
        "translation_model": "facebook/nllb-200-1.3B",
        "device": "cuda",
        "max_memory_gb": 16
    }
}

# Error messages
ERROR_MESSAGES = {
    "model_load_failed": "Failed to load model. Please check your internet connection and try again.",
    "audio_process_failed": "Failed to process audio. Please check the audio file format.",
    "translation_failed": "Translation failed. Using original text.",
    "tts_failed": "Text-to-speech generation failed. Using placeholder audio.",
    "gpu_memory_error": "GPU memory insufficient. Try using smaller models or CPU mode.",
    "audio_too_long": "Audio file is too long. Maximum duration is 30 seconds.",
    "unsupported_format": "Unsupported audio format. Please use WAV, MP3, M4A, OGG, or FLAC."
}

# Success messages
SUCCESS_MESSAGES = {
    "model_loaded": "Model loaded successfully!",
    "audio_processed": "Audio processed successfully!",
    "pipeline_ready": "Pipeline is ready for processing!"
} 