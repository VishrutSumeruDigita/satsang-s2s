#!/usr/bin/env python3
"""
Audio Translate Wrapper - Importable Interface for Speech-to-Text Translation
Makes the speech_to_text_translate.py functionality available as importable functions

Usage:
    from audio_translate_wrapper import translate_audio_to_english, WhisperAudioTranslator
    
    # Simple usage
    result = translate_audio_to_english("audio.wav")
    print(result["best_translation"])
    
    # Class-based usage
    translator = WhisperAudioTranslator()
    result = translator.translate("audio.wav")
    
    # Batch processing
    results = translator.translate_batch(["audio1.wav", "audio2.wav"])
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import numpy as np
import os
import time
import difflib
from typing import Dict, List, Tuple, Optional

class WhisperAudioTranslator:
    """Class-based interface for Hindi/English audio to English translation"""
    
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
        self.load_time = load_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
    
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
    
    def translate(self, audio_file_path: str, save_results: bool = False, 
                 output_dir: str = "translations", verbose: bool = True) -> Dict:
        """
        Translate Hindi/English audio to English text
        
        Args:
            audio_file_path (str): Path to audio file
            save_results (bool): Whether to save results to files
            output_dir (str): Directory to save results
            verbose (bool): Whether to print progress messages
            
        Returns:
            dict: Complete translation results
        """
        try:
            total_start_time = time.time()
            
            if verbose:
                print(f"\n🌍 Translating: {audio_file_path}")
            
            # Load and process audio
            audio_start = time.time()
            audio_data, sample_rate = sf.read(audio_file_path)
            
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample to 16kHz
            target_sample_rate = 16000
            if sample_rate != target_sample_rate:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
                if verbose:
                    print(f"🔄 Resampled: {sample_rate}Hz → {target_sample_rate}Hz")
            
            audio_duration = len(audio_data) / target_sample_rate
            
            # Process audio for the model
            inputs = self.processor(
                audio_data,
                sampling_rate=target_sample_rate,
                return_tensors="pt",
                return_attention_mask=True
            ).to(self.device)
            
            audio_prep_time = time.time() - audio_start
            
            # Results container
            results = {
                "success": True,
                "audio_file": audio_file_path,
                "audio_duration": audio_duration,
                "processing_times": {
                    "model_loading": self.load_time,
                    "audio_processing": audio_prep_time
                },
                "transcription": "",
                "detected_language": {"language": "unknown", "confidence": 0},
                "translations": {},
                "best_translation": "",
                "performance": {}
            }
            
            # 1. AUTO-DETECTION TRANSCRIPTION
            if verbose:
                print("🔍 Step 1: Auto-detection transcription...")
            
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
                results["processing_times"]["transcription"] = auto_time
                
                if verbose:
                    print(f"✅ Detected {detected_lang} ({lang_confidence:.1f}%): {auto_transcription[:60]}...")
                
            except Exception as e:
                if verbose:
                    print(f"❌ Transcription failed: {e}")
                results["transcription"] = ""
            
            # 2. TRANSLATION METHODS
            if verbose:
                print("🌐 Step 2: Generating English translations...")
            
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
                    results["processing_times"][f"{method_name}_translation"] = method_time
                    
                    if verbose:
                        print(f"   ✅ {description}: {translation[:60]}...")
                    
                except Exception as e:
                    if verbose:
                        print(f"   ❌ {description} failed: {e}")
                    results["translations"][method_name] = {
                        "text": f"Error: {str(e)}",
                        "method": description,
                        "time": 0
                    }
            
            # Calculate total time and select best translation
            total_time = time.time() - total_start_time
            results["processing_times"]["total"] = total_time
            
            # Find best translation (simple scoring)
            best_translation = None
            best_score = 0
            for method, data in results["translations"].items():
                if not data["text"].startswith("Error:"):
                    # Score based on text length and processing speed
                    score = len(data["text"].split()) * (100 - data["time"])
                    if score > best_score:
                        best_score = score
                        best_translation = method
            
            if best_translation:
                results["best_translation"] = results["translations"][best_translation]["text"]
                results["recommended_method"] = results["translations"][best_translation]["method"]
            
            # Performance metrics
            results["performance"] = {
                "real_time_factor": total_time / audio_duration if audio_duration > 0 else 0,
                "successful_translations": len([t for t in results["translations"].values() if not t["text"].startswith("Error:")]),
                "total_translations": len(results["translations"])
            }
            
            # Save results if requested
            if save_results:
                self._save_results(results, output_dir)
            
            if verbose:
                print(f"✅ Translation completed in {total_time:.2f}s (RTF: {results['performance']['real_time_factor']:.2f}x)")
                print(f"🏆 Best: {results['best_translation']}")
            
            return results
            
        except FileNotFoundError:
            return {
                "success": False,
                "error": f"Audio file '{audio_file_path}' not found",
                "audio_file": audio_file_path
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Translation failed: {str(e)}",
                "audio_file": audio_file_path
            }
    
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
        
        # Save best translation
        if results["best_translation"]:
            with open(f"{output_dir}/BEST_TRANSLATION.txt", "w", encoding="utf-8") as f:
                f.write(f"Best Translation:\n{results['best_translation']}\n")
                f.write(f"\nMethod: {results.get('recommended_method', 'Unknown')}\n")
                f.write(f"Audio File: {results['audio_file']}\n")
                f.write(f"Duration: {results['audio_duration']:.2f}s\n")
                f.write(f"Processing Time: {results['processing_times']['total']:.2f}s\n")
    
    def translate_batch(self, audio_files: List[str], save_results: bool = False,
                       output_dir: str = "batch_translations", verbose: bool = True) -> List[Dict]:
        """
        Translate multiple audio files
        
        Args:
            audio_files (list): List of audio file paths
            save_results (bool): Whether to save results to files
            output_dir (str): Directory to save results
            verbose (bool): Whether to show progress
            
        Returns:
            list: List of translation results
        """
        results = []
        total = len(audio_files)
        
        if verbose:
            print(f"🔄 Processing {total} audio files...")
        
        for i, audio_file in enumerate(audio_files, 1):
            if verbose:
                print(f"\n📝 [{i}/{total}] Processing: {audio_file}")
            
            # Create individual output directory for each file
            file_output_dir = f"{output_dir}/file_{i}_{os.path.basename(audio_file).split('.')[0]}" if save_results else output_dir
            
            result = self.translate(audio_file, save_results=save_results, 
                                  output_dir=file_output_dir, verbose=False)
            result["batch_index"] = i
            result["batch_total"] = total
            results.append(result)
            
            if verbose and result.get("success"):
                print(f"   ✅ Best: {result['best_translation'][:80]}...")
            elif verbose:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
        
        if verbose:
            successful = len([r for r in results if r.get("success")])
            print(f"\n✅ Batch completed: {successful}/{total} successful")
        
        return results

def translate_audio_to_english(audio_file_path: str, model_name: str = "openai/whisper-medium",
                             save_results: bool = False, output_dir: str = "translations",
                             verbose: bool = True) -> Dict:
    """
    Simple function interface for audio translation
    
    Args:
        audio_file_path (str): Path to audio file
        model_name (str): Whisper model to use
        save_results (bool): Whether to save results to files
        output_dir (str): Directory to save results
        verbose (bool): Whether to show progress messages
        
    Returns:
        dict: Translation results
    """
    translator = WhisperAudioTranslator(model_name)
    return translator.translate(audio_file_path, save_results, output_dir, verbose)

def translate_audio_batch(audio_files: List[str], model_name: str = "openai/whisper-medium",
                         save_results: bool = False, output_dir: str = "batch_translations",
                         verbose: bool = True) -> List[Dict]:
    """
    Simple function interface for batch audio translation
    
    Args:
        audio_files (list): List of audio file paths
        model_name (str): Whisper model to use
        save_results (bool): Whether to save results to files
        output_dir (str): Directory to save results
        verbose (bool): Whether to show progress messages
        
    Returns:
        list: List of translation results
    """
    translator = WhisperAudioTranslator(model_name)
    return translator.translate_batch(audio_files, save_results, output_dir, verbose)

def get_best_translation(result: Dict) -> str:
    """Extract the best translation text from result"""
    if result.get("success") and result.get("best_translation"):
        return result["best_translation"]
    return result.get("error", "Translation failed")

def print_results_summary(results: List[Dict]):
    """Print a summary of batch translation results"""
    total = len(results)
    successful = len([r for r in results if r.get("success")])
    
    print(f"\n📊 BATCH TRANSLATION SUMMARY")
    print("=" * 40)
    print(f"Total files: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    
    if successful > 0:
        print(f"\n📝 SUCCESSFUL TRANSLATIONS:")
        for i, result in enumerate([r for r in results if r.get("success")], 1):
            print(f"{i:2d}. {os.path.basename(result['audio_file'])}")
            print(f"    {result['best_translation'][:80]}...")

# Constants
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
    """Main function with examples"""
    print("=" * 60)
    print("🚀 AUDIO TRANSLATE WRAPPER - IMPORTABLE INTERFACE")
    print("=" * 60)
    
    # Test with existing audio file if available
    test_file = "indic_tts_out.wav"
    
    if os.path.exists(test_file):
        print(f"\n🔄 Testing with: {test_file}")
        
        # Simple function usage
        result = translate_audio_to_english(test_file, verbose=True, save_results=True)
        
        if result.get("success"):
            print(f"\n🎯 RESULT:")
            print(f"Best Translation: {result['best_translation']}")
            print(f"Detected Language: {result['detected_language']['language']}")
            print(f"Processing Time: {result['processing_times']['total']:.2f}s")
        else:
            print(f"\n❌ Translation failed: {result.get('error')}")
    
    else:
        print(f"\n💡 No test audio file found. Here's how to use:")
        print(f"")
        print(f"📚 USAGE EXAMPLES:")
        print(f"")
        print(f"# Simple function usage")
        print(f"from audio_translate_wrapper import translate_audio_to_english")
        print(f"result = translate_audio_to_english('your_audio.wav')")
        print(f"print(result['best_translation'])")
        print(f"")
        print(f"# Class-based usage (recommended for multiple files)")
        print(f"from audio_translate_wrapper import WhisperAudioTranslator")
        print(f"translator = WhisperAudioTranslator()")
        print(f"result = translator.translate('your_audio.wav')")
        print(f"")
        print(f"# Batch processing")
        print(f"results = translator.translate_batch(['audio1.wav', 'audio2.wav'])")
        print(f"for result in results:")
        print(f"    print(result['best_translation'])")

if __name__ == "__main__":
    main()

# Module exports
__all__ = [
    'WhisperAudioTranslator',
    'translate_audio_to_english',
    'translate_audio_batch',
    'get_best_translation',
    'print_results_summary',
    'SUPPORTED_MODELS',
    'SUPPORTED_INPUT_LANGUAGES'
]

__version__ = "1.0.0" 