#!/usr/bin/env python3
"""
Whisper Translate - Audio Translation Module
Uses OpenAI Whisper model for Hindi/English audio to English text translation

Usage:
    # Import and use
    from whisper_translate import WhisperTranslator, translate_audio_to_english
    
    # Class-based usage (recommended for multiple files)
    translator = WhisperTranslator()
    result = translator.translate("audio.wav")
    
    # Function-based usage (simple one-off translations)
    result = translate_audio_to_english("audio.wav")
    
    # Batch translations
    results = translator.translate_batch(["audio1.wav", "audio2.wav"])
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import numpy as np
import os
import time
import difflib
from typing import Dict, List, Optional, Tuple, Union

class WhisperTranslator:
    def __init__(self, model_name="openai/whisper-medium", device=None):
        """Initialize the Whisper translator"""
        self.model_name = model_name
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 Loading {model_name} on {self.device}...")
        load_start = time.time()
        
        # Load Whisper model
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        load_time = time.time() - load_start
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Supported input languages
        self.supported_languages = {
            "hindi": "hi",
            "english": "en"
        }
    
    def detect_language_content(self, text: str) -> Tuple[str, float]:
        """Detect if text contains Hindi or English content"""
        if not text.strip():
            return "empty", 0
        
        # Count Hindi/Devanagari characters
        hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        # Count English/Latin characters
        english_chars = sum(1 for char in text if char.isascii() and char.isalpha())
        
        total_chars = hindi_chars + english_chars
        if total_chars == 0:
            return "no_letters", 0
        
        hindi_percentage = (hindi_chars / total_chars) * 100
        english_percentage = (english_chars / total_chars) * 100
        
        if hindi_percentage > english_percentage:
            return "Hindi", hindi_percentage
        else:
            return "English", english_percentage
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        return difflib.SequenceMatcher(None, text1, text2).ratio() * 100
    
    def _load_and_process_audio(self, audio_file_path: str) -> Tuple[np.ndarray, float, float]:
        """Load and preprocess audio file"""
        # Load audio
        audio_data, sample_rate = sf.read(audio_file_path)
        
        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 16kHz (Whisper requirement)
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
        
        audio_duration = len(audio_data) / target_sample_rate
        return audio_data, target_sample_rate, audio_duration
    
    def translate(self, audio_file_path: str, save_results: bool = False, output_dir: str = "translations") -> Dict:
        """
        Translate Hindi/English audio to English text
        
        Args:
            audio_file_path (str): Path to audio file
            save_results (bool): Whether to save results to files
            output_dir (str): Directory to save results
            
        Returns:
            dict: Translation results with multiple methods and metadata
        """
        try:
            total_start_time = time.time()
            
            # Load and process audio
            audio_start = time.time()
            audio_data, sample_rate, audio_duration = self._load_and_process_audio(audio_file_path)
            
            # Process audio for the model
            inputs = self.processor(
                audio_data,
                sampling_rate=sample_rate,
                return_tensors="pt",
                return_attention_mask=True
            ).to(self.device)
            
            audio_prep_time = time.time() - audio_start
            
            results = {
                "audio_file": audio_file_path,
                "audio_duration": audio_duration,
                "audio_processing_time": audio_prep_time,
                "translations": {},
                "transcription": None,
                "detected_language": None,
                "timing": {}
            }
            
            # 1. AUTO-DETECTION TRANSCRIPTION
            auto_start = time.time()
            try:
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        inputs["input_features"],
                        attention_mask=inputs.get("attention_mask"),
                        max_new_tokens=400,
                        num_beams=1,
                        do_sample=False,
                        use_cache=True
                    )
                
                auto_transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                auto_time = time.time() - auto_start
                
                # Detect input language
                detected_lang, lang_confidence = self.detect_language_content(auto_transcription)
                
                results["transcription"] = auto_transcription
                results["detected_language"] = {"language": detected_lang, "confidence": lang_confidence}
                results["timing"]["transcription"] = auto_time
                
            except Exception as e:
                print(f"❌ Auto-transcription failed: {e}")
                results["transcription"] = ""
                results["detected_language"] = {"language": "unknown", "confidence": 0}
            
            # 2. TRANSLATION METHODS
            translation_methods = [
                ("auto", None, "translate", "Auto-detection → English"),
                ("hindi", "hindi", "translate", "Hindi → English"),
                ("english", "english", "transcribe", "English transcription")
            ]
            
            for method_name, language, task, description in translation_methods:
                method_start = time.time()
                try:
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=language, task=task
                    )
                    
                    with torch.no_grad():
                        predicted_ids = self.model.generate(
                            inputs["input_features"],
                            attention_mask=inputs.get("attention_mask"),
                            forced_decoder_ids=forced_decoder_ids,
                            max_new_tokens=400,
                            num_beams=2 if task == "translate" else 1,
                            do_sample=False,
                            use_cache=True
                        )
                    
                    translation = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    method_time = time.time() - method_start
                    
                    results["translations"][method_name] = {
                        "text": translation,
                        "method": description,
                        "time": method_time
                    }
                    results["timing"][method_name] = method_time
                    
                except Exception as e:
                    print(f"❌ {description} failed: {e}")
                    results["translations"][method_name] = {
                        "text": f"Error: {str(e)}",
                        "method": description,
                        "time": 0
                    }
            
            # Calculate total time and performance metrics
            total_time = time.time() - total_start_time
            results["total_time"] = total_time
            results["real_time_factor"] = total_time / audio_duration if audio_duration > 0 else 0
            results["successful_translations"] = len([t for t in results["translations"].values() if not t["text"].startswith("Error:")])
            
            # Find best translation (simple scoring)
            best_translation = None
            best_score = 0
            for method, data in results["translations"].items():
                if not data["text"].startswith("Error:"):
                    score = len(data["text"].split()) * (100 - data["time"])
                    if score > best_score:
                        best_score = score
                        best_translation = method
            
            results["recommended"] = best_translation
            
            # Calculate similarities
            if len(results["translations"]) > 1:
                similarities = {}
                translation_items = [(k, v) for k, v in results["translations"].items() if not v["text"].startswith("Error:")]
                for i, (method1, data1) in enumerate(translation_items):
                    for method2, data2 in translation_items[i+1:]:
                        similarity = self.calculate_similarity(data1["text"], data2["text"])
                        similarities[f"{method1}_vs_{method2}"] = similarity
                results["similarities"] = similarities
            
            # Save results if requested
            if save_results:
                self._save_results(results, output_dir)
            
            return results
            
        except FileNotFoundError:
            return {"error": f"Audio file '{audio_file_path}' not found"}
        except Exception as e:
            return {"error": f"Translation failed: {str(e)}"}
    
    def _save_results(self, results: Dict, output_dir: str):
        """Save translation results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual translation files
        for method, data in results["translations"].items():
            if not data["text"].startswith("Error:"):
                filename = f"{output_dir}/{method}_translation.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"Method: {data['method']}\n")
                    f.write(f"Processing Time: {data['time']:.2f}s\n")
                    f.write(f"Text: {data['text']}\n")
        
        # Save summary report
        summary_file = f"{output_dir}/TRANSLATION_SUMMARY.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("HINDI/ENGLISH TO ENGLISH TRANSLATION REPORT\n")
            f.write("=" * 45 + "\n\n")
            f.write(f"Audio File: {results['audio_file']}\n")
            f.write(f"Duration: {results['audio_duration']:.2f}s\n")
            f.write(f"Total Processing Time: {results['total_time']:.2f}s\n")
            f.write(f"Detected Language: {results['detected_language']['language']}\n\n")
            
            f.write("ENGLISH TRANSLATIONS:\n")
            f.write("-" * 20 + "\n")
            for method, data in results["translations"].items():
                if not data["text"].startswith("Error:"):
                    f.write(f"\n{data['method']}:\n{data['text']}\n")
    
    def translate_batch(self, audio_files: List[str], save_results: bool = False, 
                       output_dir: str = "batch_translations", show_progress: bool = True) -> List[Dict]:
        """
        Translate multiple audio files in batch
        
        Args:
            audio_files (list): List of audio file paths
            save_results (bool): Whether to save results to files
            output_dir (str): Directory to save results
            show_progress (bool): Whether to show progress messages
            
        Returns:
            list: List of translation results
        """
        results = []
        total = len(audio_files)
        
        if show_progress:
            print(f"🔄 Processing {total} audio files...")
        
        for i, audio_file in enumerate(audio_files, 1):
            if show_progress:
                print(f"📝 [{i}/{total}] Processing {audio_file}...")
            
            # Create individual output directory for each file
            file_output_dir = f"{output_dir}/file_{i}_{os.path.basename(audio_file).split('.')[0]}" if save_results else output_dir
            
            result = self.translate(audio_file, save_results=save_results, output_dir=file_output_dir)
            result["batch_index"] = i
            results.append(result)
        
        if show_progress:
            print(f"✅ Batch translation completed!")
        
        return results

def translate_audio_to_english(audio_file_path: str, model_name: str = "openai/whisper-medium", 
                             save_results: bool = False, output_dir: str = "translations") -> Dict:
    """
    Simple function interface for audio translation
    
    Args:
        audio_file_path (str): Path to audio file
        model_name (str): Whisper model to use
        save_results (bool): Whether to save results to files
        output_dir (str): Directory to save results
        
    Returns:
        dict: Translation results
    """
    translator = WhisperTranslator(model_name)
    return translator.translate(audio_file_path, save_results, output_dir)

def translate_audio_batch(audio_files: List[str], model_name: str = "openai/whisper-medium",
                         save_results: bool = False, output_dir: str = "batch_translations",
                         show_progress: bool = True) -> List[Dict]:
    """
    Simple function interface for batch audio translation
    
    Args:
        audio_files (list): List of audio file paths
        model_name (str): Whisper model to use
        save_results (bool): Whether to save results to files
        output_dir (str): Directory to save results
        show_progress (bool): Whether to show progress messages
        
    Returns:
        list: List of translation results
    """
    translator = WhisperTranslator(model_name)
    return translator.translate_batch(audio_files, save_results, output_dir, show_progress)

def print_translation_results(results: Dict):
    """Pretty print translation results"""
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"\n🎯 TRANSLATION RESULTS")
    print("=" * 50)
    print(f"📁 Audio File: {results['audio_file']}")
    print(f"📊 Duration: {results['audio_duration']:.2f}s")
    print(f"⏱️  Total Time: {results['total_time']:.2f}s")
    print(f"📈 Real-time Factor: {results['real_time_factor']:.2f}x")
    
    if results['detected_language']:
        lang_info = results['detected_language']
        print(f"🔍 Detected: {lang_info['language']} ({lang_info['confidence']:.1f}% confidence)")
    
    print(f"\n📝 ENGLISH TRANSLATIONS:")
    print("-" * 30)
    
    for method, data in results["translations"].items():
        if not data["text"].startswith("Error:"):
            print(f"\n🔸 {data['method'].upper()}:")
            print(f"   Text: {data['text']}")
            print(f"   Time: {data['time']:.2f}s")
    
    if results.get("recommended"):
        recommended = results["translations"][results["recommended"]]
        print(f"\n🏆 RECOMMENDED:")
        print(f"   {recommended['text']}")

# Supported models and languages
SUPPORTED_MODELS = [
    "openai/whisper-tiny",
    "openai/whisper-base", 
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large"
]

SUPPORTED_INPUT_LANGUAGES = ["Hindi", "English"]
OUTPUT_LANGUAGE = "English"

def main():
    """Main function with sample usage"""
    print("=" * 70)
    print("🚀 WHISPER TRANSLATE - HINDI/ENGLISH TO ENGLISH")
    print("=" * 70)
    
    # Initialize translator
    translator = WhisperTranslator()
    
    # Example usage
    audio_file = "indic_tts_out.wav"  # Default audio file
    
    if os.path.exists(audio_file):
        print(f"\n🔄 Testing with audio file: {audio_file}")
        results = translator.translate(audio_file, save_results=True)
        print_translation_results(results)
    else:
        print(f"\n💡 No test audio file found. Example usage:")
        print(f"   results = translator.translate('your_audio.wav')")
        print(f"   print_translation_results(results)")
    
    print(f"\n📚 IMPORT USAGE EXAMPLES:")
    print("   from whisper_translate import WhisperTranslator, translate_audio_to_english")
    print("   from whisper_translate import translate_audio_batch, print_translation_results")
    print("   ")
    print("   # Simple usage")
    print("   result = translate_audio_to_english('audio.wav')")
    print("   ")
    print("   # Class-based usage")
    print("   translator = WhisperTranslator()")
    print("   result = translator.translate('audio.wav')")
    print("   ")
    print("   # Batch usage")
    print("   results = translate_audio_batch(['audio1.wav', 'audio2.wav'])")
    
    return translator

if __name__ == "__main__":
    # Run main function
    translator = main()
    
    print(f"\n✅ Translator ready for use!")
    print(f"📌 Use: translator.translate('your_audio.wav')")

# Module exports for easy importing
__all__ = [
    'WhisperTranslator',
    'translate_audio_to_english',
    'translate_audio_batch', 
    'print_translation_results',
    'SUPPORTED_MODELS',
    'SUPPORTED_INPUT_LANGUAGES'
]

# Version info
__version__ = "1.0.0"
__author__ = "OpenAI Whisper" 