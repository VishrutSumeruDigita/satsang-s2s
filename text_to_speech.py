import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import numpy as np
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Whisper model using transformers library (more reliable)
# Using medium model for better accuracy on Hindi/Indic languages
model_name = "openai/whisper-medium"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

# Define Indian languages supported by Whisper
indian_languages = {
    "hindi": "hi",
    "bengali": "bn", 
    "gujarati": "gu",
    "tamil": "ta",
    "telugu": "te",
    "marathi": "mr",
    "kannada": "kn",
    "malayalam": "ml",
    "punjabi": "pa",
    "urdu": "ur",
    "assamese": "as",
    "nepali": "ne"
}

# Input audio file path (you can change this to your audio file)
audio_file_path = "indic_tts_out.wav"  # Using the output from the previous TTS as example

try:
    # Load and process audio
    audio_data, sample_rate = sf.read(audio_file_path)
    
    # Ensure audio is mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Resample to 16kHz (Whisper expects 16kHz)
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
        print(f"Resampled audio from {sample_rate}Hz to {target_sample_rate}Hz")
    
    # Process audio for the model
    inputs = processor(
        audio_data, 
        sampling_rate=target_sample_rate, 
        return_tensors="pt",
        return_attention_mask=True  # Explicitly return attention mask
    )
    inputs = inputs.to(device)
    
    print("\n" + "="*60)
    print("TRANSCRIBING AUDIO IN MULTIPLE INDIAN LANGUAGES")
    print("="*60)
    
    # Create output directory for transcriptions
    output_dir = "transcriptions"
    os.makedirs(output_dir, exist_ok=True)
    
    all_transcriptions = {}
    
    # First, try automatic language detection
    print("\n1. AUTO-DETECTION (No language specified):")
    print("-" * 50)
    try:
        with torch.no_grad():
            predicted_ids = model.generate(
                inputs["input_features"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=400,  # Reduced to be safe with token limits
                num_beams=1,
                do_sample=False,
                use_cache=True
            )
        
        auto_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(f"AUTO-DETECTED: {auto_transcription}")
        
        # Save auto-detection result
        with open(f"{output_dir}/auto_detection.txt", "w", encoding="utf-8") as f:
            f.write(auto_transcription)
        
        all_transcriptions["auto_detection"] = auto_transcription
        
    except Exception as e:
        print(f"Auto-detection failed: {e}")
    
    # Now try each Indian language specifically
    print(f"\n2. LANGUAGE-SPECIFIC TRANSCRIPTIONS:")
    print("-" * 50)
    
    for lang_name, lang_code in indian_languages.items():
        try:
            print(f"\nTrying {lang_name.upper():<12} ({lang_code})... ", end="")
            
                        # Set language-specific decoder prompts
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang_name, task="transcribe")
            
            # Generate transcription for this language
            with torch.no_grad():
                predicted_ids = model.generate(
                    inputs["input_features"],
                    attention_mask=inputs.get("attention_mask"),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=400,  # Reduced to be safe with token limits
                    num_beams=1,
                    do_sample=False,
                    use_cache=True
                )
            
            # Decode the transcription
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Save to individual file
            filename = f"{output_dir}/{lang_name}_{lang_code}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(transcription)
            
            all_transcriptions[lang_name] = transcription
            
            # Show result (truncated if too long)
            display_text = transcription[:100] + "..." if len(transcription) > 100 else transcription
            print(f"✓ {display_text}")
            
        except Exception as e:
            if "max_target_positions" in str(e):
                # Try with even fewer tokens if we hit the limit
                try:
                    print(f"Retrying with fewer tokens... ", end="")
                    with torch.no_grad():
                        predicted_ids = model.generate(
                            inputs["input_features"],
                            attention_mask=inputs.get("attention_mask"),
                            forced_decoder_ids=forced_decoder_ids,
                            max_new_tokens=300,  # Even more conservative
                            num_beams=1,
                            do_sample=False,
                            use_cache=True
                        )
                    
                    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    filename = f"{output_dir}/{lang_name}_{lang_code}.txt"
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(transcription)
                    
                    all_transcriptions[lang_name] = transcription
                    display_text = transcription[:100] + "..." if len(transcription) > 100 else transcription
                    print(f"✓ {display_text}")
                except Exception as e2:
                    print(f"✗ Still failed: {str(e2)[:50]}...")
            else:
                print(f"✗ Error: {str(e)[:50]}...")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - ALL TRANSCRIPTIONS:")
    print("="*60)
    
    for lang, text in all_transcriptions.items():
        print(f"\n{lang.upper().replace('_', ' '):<15}: {text}")
    
    # Save combined results
    combined_file = f"{output_dir}/all_languages_combined.txt"
    with open(combined_file, "w", encoding="utf-8") as f:
        f.write("SPEECH-TO-TEXT TRANSCRIPTIONS IN MULTIPLE INDIAN LANGUAGES\n")
        f.write("=" * 60 + "\n\n")
        for lang, text in all_transcriptions.items():
            f.write(f"{lang.upper().replace('_', ' ')}:\n{text}\n\n")
            f.write("-" * 40 + "\n\n")
    
    print(f"\n✅ All transcriptions saved in '{output_dir}/' directory")
    print(f"✅ Combined file: {combined_file}")
    print(f"✅ Total languages processed: {len(all_transcriptions)}")
    
except FileNotFoundError:
    print(f"Audio file '{audio_file_path}' not found.")
    print("Please provide a valid audio file path.")
except Exception as e:
    print(f"Error processing audio: {e}")
    print("\nTip: For best results with Indian languages:")
    print("- Ensure you have 'transformers', 'torch', 'soundfile' and 'librosa' installed")
    print("- Use good quality audio files")
    print("- Check that your audio file exists and is in a supported format")
