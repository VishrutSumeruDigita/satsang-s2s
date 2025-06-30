import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import numpy as np
import os
import time
from collections import Counter
import difflib

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Whisper model using transformers library
model_name = "openai/whisper-medium"
print(f"Loading {model_name} model...")
load_start = time.time()
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
load_time = time.time() - load_start
print(f"✅ Model loaded in {load_time:.2f} seconds")

# Define supported languages for translation (input languages)
supported_languages = {
    "hindi": "hi",
    "english": "en"
}

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

# Input audio file path
audio_file_path = "indic_tts_out.wav"

try:
    total_start_time = time.time()
    
    # Load and process audio
    print("\n" + "="*70)
    print("🌍 HINDI/ENGLISH TO ENGLISH TRANSLATION")
    print("="*70)
    
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
        print(f"🔄 Resampled: {sample_rate}Hz → {target_sample_rate}Hz")
    
    audio_duration = len(audio_data) / target_sample_rate
    print(f"📊 Audio duration: {audio_duration:.2f} seconds")
    
    # Process audio for the model
    inputs = processor(
        audio_data, 
        sampling_rate=target_sample_rate, 
        return_tensors="pt",
        return_attention_mask=True
    )
    inputs = inputs.to(device)
    audio_prep_time = time.time() - audio_start
    print(f"⏱️  Audio processing time: {audio_prep_time:.2f} seconds")
    
    # Create output directory
    output_dir = "translations"
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    timing_data = {}
    
    # 1. AUTO-DETECTION (TRANSCRIPTION) FIRST
    print(f"\n🔍 STEP 1: AUTO-DETECTION & TRANSCRIPTION")
    print("-" * 50)
    
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
        timing_data["auto_transcription"] = auto_time
        
        # Detect input language
        detected_lang, lang_confidence = detect_language_content(auto_transcription)
        
        print(f"✅ TRANSCRIBED: {auto_transcription[:100]}...")
        print(f"📝 Detected Language: {detected_lang} ({lang_confidence:.1f}% confidence)")
        print(f"⏱️  Time taken: {auto_time:.2f} seconds")
        
        all_results["transcription"] = {
            "text": auto_transcription,
            "language": detected_lang,
            "confidence": lang_confidence,
            "time": auto_time
        }
        
    except Exception as e:
        print(f"❌ Auto-transcription failed: {e}")
        auto_transcription = ""
        detected_lang = "unknown"
    
    # 2. TRANSLATION TO ENGLISH
    print(f"\n🌐 STEP 2: TRANSLATION TO ENGLISH")
    print("-" * 50)
    
    translation_results = {}
    
    # Try auto-detection translation (no language specified)
    print(f"\n🔄 Auto-detection translation... ", end="")
    auto_trans_start = time.time()
    try:
        # Use translation task
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=None, task="translate")
        
        with torch.no_grad():
            predicted_ids = model.generate(
                inputs["input_features"],
                attention_mask=inputs.get("attention_mask"),
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=400,
                num_beams=2,  # Slightly higher for better translation quality
                do_sample=False,
                use_cache=True
            )
        
        auto_translation = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        auto_trans_time = time.time() - auto_trans_start
        timing_data["auto_translation"] = auto_trans_time
        
        print(f"✅ {auto_translation[:80]}... ({auto_trans_time:.2f}s)")
        
        translation_results["auto"] = {
            "text": auto_translation,
            "method": "Auto-detection → English",
            "time": auto_trans_time
        }
        
    except Exception as e:
        print(f"❌ Failed: {str(e)[:40]}...")
    
    # Try Hindi to English translation
    print(f"🇮🇳 Hindi → English translation... ", end="")
    hindi_trans_start = time.time()
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
        hindi_trans_time = time.time() - hindi_trans_start
        timing_data["hindi_translation"] = hindi_trans_time
        
        print(f"✅ {hindi_translation[:80]}... ({hindi_trans_time:.2f}s)")
        
        translation_results["hindi"] = {
            "text": hindi_translation,
            "method": "Hindi → English", 
            "time": hindi_trans_time
        }
        
    except Exception as e:
        print(f"❌ Failed: {str(e)[:40]}...")
    
    # Try English transcription (if input is English)
    print(f"🇺🇸 English transcription... ", end="")
    english_start = time.time()
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
        english_time = time.time() - english_start
        timing_data["english_transcription"] = english_time
        
        print(f"✅ {english_transcription[:80]}... ({english_time:.2f}s)")
        
        translation_results["english"] = {
            "text": english_transcription,
            "method": "English transcription",
            "time": english_time
        }
        
    except Exception as e:
        print(f"❌ Failed: {str(e)[:40]}...")
    
    total_time = time.time() - total_start_time
    
    # 3. ANALYSIS & RESULTS
    print(f"\n📊 PERFORMANCE METRICS & RESULTS")
    print("="*70)
    
    print(f"⏱️  TIMING BREAKDOWN:")
    print(f"   Model Loading:     {load_time:.2f}s")
    print(f"   Audio Processing:  {audio_prep_time:.2f}s")
    print(f"   Transcription:     {timing_data.get('auto_transcription', 0):.2f}s")
    print(f"   Translations:      {sum(v for k, v in timing_data.items() if 'translation' in k or 'english' in k):.2f}s")
    print(f"   TOTAL TIME:        {total_time:.2f}s")
    
    print(f"\n📈 PROCESSING STATS:")
    print(f"   Audio Duration:    {audio_duration:.2f}s")
    print(f"   Real-time Factor:  {total_time/audio_duration:.2f}x")
    print(f"   Successful Tasks:  {len(translation_results)}/3")
    
    # Compare translations for quality
    if len(translation_results) > 1:
        print(f"\n🔍 TRANSLATION COMPARISON:")
        results_list = list(translation_results.items())
        for i, (method1, data1) in enumerate(results_list):
            for method2, data2 in results_list[i+1:]:
                similarity = calculate_similarity(data1["text"], data2["text"])
                print(f"   {method1.title()} vs {method2.title()}: {similarity:.1f}% similarity")
    
    # Show all results
    print(f"\n📝 ALL ENGLISH OUTPUTS:")
    print("-" * 50)
    
    best_translation = None
    best_score = 0
    
    for method, data in translation_results.items():
        print(f"\n🔸 {data['method'].upper()}:")
        print(f"   Text: {data['text']}")
        print(f"   Time: {data['time']:.2f}s")
        
        # Simple scoring based on length and reasonable English content
        score = len(data['text'].split()) * (100 - data['time'])  # Prefer longer, faster results
        if score > best_score:
            best_score = score
            best_translation = method
    
    # Save all results
    for method, data in translation_results.items():
        filename = f"{output_dir}/{method}_translation.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Method: {data['method']}\n")
            f.write(f"Processing Time: {data['time']:.2f}s\n")
            f.write(f"Text: {data['text']}\n")
    
    # Save summary
    summary_file = f"{output_dir}/TRANSLATION_SUMMARY.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("HINDI/ENGLISH TO ENGLISH TRANSLATION REPORT\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Audio File: {audio_file_path}\n")
        f.write(f"Duration: {audio_duration:.2f}s\n")
        f.write(f"Total Processing Time: {total_time:.2f}s\n")
        f.write(f"Detected Input Language: {detected_lang}\n\n")
        
        f.write("ENGLISH TRANSLATIONS:\n")
        f.write("-" * 20 + "\n")
        for method, data in translation_results.items():
            f.write(f"\n{data['method']}:\n{data['text']}\n")
    
    print(f"\n✅ TRANSLATION COMPLETE!")
    print(f"📁 Results saved in: {output_dir}/")
    print(f"📊 Summary report: {summary_file}")
    
    # Recommendation
    if best_translation and translation_results:
        recommended = translation_results[best_translation]
        print(f"\n🏆 RECOMMENDED OUTPUT:")
        print(f"   Method: {recommended['method']}")
        print(f"   Text: {recommended['text']}")
        print(f"   ({recommended['time']:.2f}s processing time)")

except FileNotFoundError:
    print(f"❌ Audio file '{audio_file_path}' not found.")
except Exception as e:
    print(f"❌ Error: {e}")
    print(f"⏱️  Time elapsed: {time.time() - total_start_time:.2f}s") 





    #NOTES -> use Python 3.11.13

