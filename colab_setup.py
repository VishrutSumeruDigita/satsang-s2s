# ========================================
# GOOGLE COLAB SETUP FOR HINDI/ENGLISH TO ENGLISH TRANSLATION
# ========================================

# First, install required packages
import subprocess
import sys

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        print(f"✅ {package} installed successfully")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")

print("🔧 Installing required packages for speech translation...")

# Install packages
packages = [
    "torch",
    "transformers>=4.21.0", 
    "soundfile",
    "librosa",
    "numpy"
]

for package in packages:
    install_package(package)

print("\n" + "="*50)
print("📦 INSTALLATION COMPLETE!")
print("="*50)

# ========================================
# MAIN TRANSLATION SCRIPT
# ========================================

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import numpy as np
import os
import time
import difflib

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Using device: {device}")

# Load Whisper model
model_name = "openai/whisper-medium"
print(f"📥 Loading {model_name} model...")
load_start = time.time()

processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

load_time = time.time() - load_start
print(f"✅ Model loaded in {load_time:.2f} seconds")

def detect_language_content(text):
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

def calculate_similarity(text1, text2):
    """Calculate similarity between two texts"""
    return difflib.SequenceMatcher(None, text1, text2).ratio() * 100

def translate_audio_to_english(audio_file_path):
    """Main function to translate Hindi/English audio to English text"""
    
    try:
        total_start_time = time.time()
        
        print("\n" + "="*60)
        print("🌍 HINDI/ENGLISH TO ENGLISH TRANSLATION")
        print("="*60)
        
        # Load audio
        audio_start = time.time()
        audio_data, sample_rate = sf.read(audio_file_path)
        
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 16kHz
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
            print(f"🔄 Resampled: {sample_rate}Hz → {target_sample_rate}Hz")
        
        audio_duration = len(audio_data) / target_sample_rate
        print(f"📊 Audio duration: {audio_duration:.2f} seconds")
        
        # Process audio
        inputs = processor(
            audio_data,
            sampling_rate=target_sample_rate,
            return_tensors="pt",
            return_attention_mask=True
        )
        inputs = inputs.to(device)
        
        audio_prep_time = time.time() - audio_start
        print(f"⏱️  Audio processing: {audio_prep_time:.2f}s")
        
        translation_results = {}
        
        # 1. Auto-detection transcription first
        print(f"\n🔍 STEP 1: AUTO-DETECTION & TRANSCRIPTION")
        print("-" * 40)
        
        auto_start = time.time()
        try:
            with torch.no_grad():
                predicted_ids = model.generate(
                    inputs["input_features"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=400,
                    num_beams=1,
                    do_sample=False,
                    use_cache=True
                )
            
            auto_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            auto_time = time.time() - auto_start
            
            detected_lang, confidence = detect_language_content(auto_transcription)
            
            print(f"✅ TRANSCRIBED: {auto_transcription}")
            print(f"📝 Detected: {detected_lang} ({confidence:.1f}% confidence)")
            print(f"⏱️  Time: {auto_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            auto_transcription = ""
            detected_lang = "unknown"
        
        # 2. Translation attempts
        print(f"\n🌐 STEP 2: TRANSLATION TO ENGLISH")
        print("-" * 40)
        
        # Try auto-detection translation
        print(f"🔄 Auto-detection translation... ", end="")
        try:
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=None, task="translate")
            
            with torch.no_grad():
                predicted_ids = model.generate(
                    inputs["input_features"],
                    attention_mask=inputs.get("attention_mask"),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=400,
                    num_beams=2,
                    do_sample=False,
                    use_cache=True
                )
            
            auto_translation = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print(f"✅ {auto_translation}")
            
            translation_results["auto"] = {
                "text": auto_translation,
                "method": "Auto-detection → English"
            }
            
        except Exception as e:
            print(f"❌ Failed: {str(e)[:40]}...")
        
        # Try Hindi to English
        print(f"🇮🇳 Hindi → English... ", end="")
        try:
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="hindi", task="translate")
            
            with torch.no_grad():
                predicted_ids = model.generate(
                    inputs["input_features"],
                    attention_mask=inputs.get("attention_mask"),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=400,
                    num_beams=2,
                    do_sample=False,
                    use_cache=True
                )
            
            hindi_translation = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print(f"✅ {hindi_translation}")
            
            translation_results["hindi"] = {
                "text": hindi_translation,
                "method": "Hindi → English"
            }
            
        except Exception as e:
            print(f"❌ Failed: {str(e)[:40]}...")
        
        # Try English transcription
        print(f"🇺🇸 English transcription... ", end="")
        try:
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
            
            with torch.no_grad():
                predicted_ids = model.generate(
                    inputs["input_features"],
                    attention_mask=inputs.get("attention_mask"),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=400,
                    num_beams=1,
                    do_sample=False,
                    use_cache=True
                )
            
            english_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print(f"✅ {english_transcription}")
            
            translation_results["english"] = {
                "text": english_transcription,
                "method": "English transcription"
            }
            
        except Exception as e:
            print(f"❌ Failed: {str(e)[:40]}...")
        
        total_time = time.time() - total_start_time
        
        # Results
        print(f"\n📊 RESULTS SUMMARY")
        print("="*40)
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📈 Real-time factor: {total_time/audio_duration:.2f}x")
        print(f"✅ Successful translations: {len(translation_results)}")
        
        print(f"\n📝 ALL ENGLISH OUTPUTS:")
        print("-" * 40)
        
        for method, data in translation_results.items():
            print(f"\n🔸 {data['method'].upper()}:")
            print(f"   {data['text']}")
        
        # Return results
        return translation_results, auto_transcription, detected_lang
        
    except FileNotFoundError:
        print(f"❌ Audio file '{audio_file_path}' not found")
        return {}, "", "unknown"
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}, "", "unknown"

# ========================================
# READY TO USE!
# ========================================

print(f"\n🎉 SETUP COMPLETE! Ready to translate audio.")
print(f"📝 Usage:")
print(f"   results, transcription, detected_lang = translate_audio_to_english('your_audio_file.wav')")
print(f"\n💡 Upload your audio file to Colab and run the function!")

# Example usage (uncomment when you have an audio file):
# results, transcription, detected_lang = translate_audio_to_english("your_audio_file.wav") 