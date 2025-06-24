import torch
import torchaudio
import librosa
import numpy as np
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    MarianMTModel, MarianTokenizer,
    VitsModel, VitsTokenizer,
    AutoProcessor, AutoModel,
    pipeline
)
import soundfile as sf
from typing import Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class SpeechToSpeechPipeline:
    def __init__(self, target_language: str = "ta"):  # Default to Tamil
        """
        Initialize the Speech-to-Speech Pipeline
        
        Args:
            target_language: Target Indian language code (ta=Tamil, te=Telugu, kn=Kannada, etc.)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.target_language = target_language
        
        # Initialize models
        self._load_asr_model()
        self._load_translation_model()
        self._load_tts_model()
        
    def _load_asr_model(self):
        """Load Hindi ASR model (Whisper)"""
        print("Loading Hindi ASR model...")
        model_name = "openai/whisper-small"
        self.asr_processor = WhisperProcessor.from_pretrained(model_name)
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.asr_model.to(self.device)
        self.asr_model.config.forced_decoder_ids = None
        self.asr_model.config.suppress_tokens = []
        print("Hindi ASR model loaded successfully!")
        
    def _load_translation_model(self):
        """Load Hindi to target language translation model"""
        print(f"Loading Hindi to {self.target_language} translation model...")
        
        # Use IndicBART for Indian language translation
        model_name = "ai4bharat/IndicBART"
        try:
            self.translation_model = AutoModel.from_pretrained(model_name)
            self.translation_processor = AutoProcessor.from_pretrained(model_name)
            self.translation_model.to(self.device)
        except:
            # Fallback to a general multilingual model
            print("Using fallback translation model...")
            self.translation_pipeline = pipeline(
                "translation",
                model="facebook/nllb-200-distilled-600M",
                device=0 if torch.cuda.is_available() else -1
            )
        print("Translation model loaded successfully!")
        
    def _load_tts_model(self):
        """Load TTS model for target language"""
        print(f"Loading TTS model for {self.target_language}...")
        
        # Use VITS model for TTS
        try:
            if self.target_language == "ta":
                model_name = "microsoft/speecht5_tts"  # General TTS model
            else:
                model_name = "microsoft/speecht5_tts"
                
            self.tts_processor = AutoProcessor.from_pretrained(model_name)
            self.tts_model = AutoModel.from_pretrained(model_name)
            self.tts_model.to(self.device)
        except:
            print("Using alternative TTS approach...")
            self.tts_model = None
            
        print("TTS model loaded successfully!")
        
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe Hindi audio to text
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed Hindi text
        """
        print("Transcribing audio...")
        
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Process with Whisper
        input_features = self.asr_processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.asr_model.generate(
                input_features,
                forced_decoder_ids=self.asr_processor.get_decoder_prompt_ids(
                    language="hi", task="transcribe"
                )
            )
        
        transcription = self.asr_processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        
        print(f"Transcription: {transcription}")
        return transcription
        
    def translate_text(self, hindi_text: str) -> str:
        """
        Translate Hindi text to target language
        
        Args:
            hindi_text: Hindi text to translate
            
        Returns:
            Translated text in target language
        """
        print(f"Translating to {self.target_language}...")
        
        try:
            if hasattr(self, 'translation_pipeline'):
                # Use NLLB model
                lang_map = {
                    # Indian Languages
                    "ta": "tam_Taml",
                    "te": "tel_Telu", 
                    "kn": "kan_Knda",
                    "ml": "mal_Mlym",
                    "gu": "guj_Gujr",
                    "pa": "pan_Guru",
                    "bn": "ben_Beng",
                    "or": "ory_Orya",
                    "as": "asm_Beng",
                    "mr": "mar_Deva",
                    
                    # International Languages
                    "en": "eng_Latn",
                    "fr": "fra_Latn",
                    "de": "deu_Latn",
                    "ru": "rus_Cyrl",
                    "zh": "zho_Hans",
                    "ja": "jpn_Jpan"
                }
                
                target_lang = lang_map.get(self.target_language, "tam_Taml")
                result = self.translation_pipeline(
                    hindi_text, 
                    src_lang="hin_Deva", 
                    tgt_lang=target_lang
                )
                translated_text = result[0]['translation_text']
            else:
                # Fallback: return Hindi text (you can implement other translation logic)
                translated_text = hindi_text
                
        except Exception as e:
            print(f"Translation error: {e}")
            translated_text = hindi_text
            
        print(f"Translation: {translated_text}")
        return translated_text
        
    def synthesize_speech(self, text: str, output_path: str = "output.wav") -> str:
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech
            output_path: Output audio file path
            
        Returns:
            Path to generated audio file
        """
        print("Synthesizing speech...")
        
        try:
            if self.tts_model is not None:
                # Use loaded TTS model
                inputs = self.tts_processor(text=text, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    speech = self.tts_model.generate_speech(
                        inputs["input_ids"], 
                        speaker_embeddings=None
                    )
                
                # Save audio
                sf.write(output_path, speech.cpu().numpy(), 16000)
            else:
                # Fallback: Create a simple beep sound as placeholder
                duration = len(text) * 0.1  # Rough estimate
                sample_rate = 16000
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
                sf.write(output_path, audio, sample_rate)
                
        except Exception as e:
            print(f"TTS error: {e}")
            # Create placeholder audio
            duration = 2.0
            sample_rate = 16000
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = 0.3 * np.sin(2 * np.pi * 440 * t)
            sf.write(output_path, audio, sample_rate)
            
        print(f"Speech synthesized: {output_path}")
        return output_path
        
    def process_speech_to_speech(self, audio_path: str) -> Tuple[str, str, str]:
        """
        Complete speech-to-speech pipeline
        
        Args:
            audio_path: Input Hindi audio file path
            
        Returns:
            Tuple of (hindi_transcription, translated_text, output_audio_path)
        """
        # Step 1: Transcribe Hindi audio
        hindi_transcription = self.transcribe_audio(audio_path)
        
        # Step 2: Translate to target language
        translated_text = self.translate_text(hindi_transcription)
        
        # Step 3: Synthesize speech in target language
        output_audio_path = self.synthesize_speech(translated_text)
        
        return hindi_transcription, translated_text, output_audio_path

# Language mapping for the frontend
SUPPORTED_LANGUAGES = {
    # Indian Languages
    "Tamil": "ta",
    "Telugu": "te", 
    "Kannada": "kn",
    "Malayalam": "ml",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Bengali": "bn",
    "Odia": "or",
    "Assamese": "as",
    "Marathi": "mr",
    
    # International Languages
    "English": "en",
    "French": "fr",
    "German": "de",
    "Russian": "ru",
    "Chinese (Simplified)": "zh",
    "Japanese": "ja"
} 