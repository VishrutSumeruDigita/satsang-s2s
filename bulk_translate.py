#!/usr/bin/env python3
"""
Bulk Translation Script
- Takes input from audio_inpts directory 
- Translates using orchestrator
- Saves as {audio_file_name}_en.txt in translation directory
- Translation directory has subdirectories for each language: en gu hi kn ta te
"""

import os
import glob
import time
from pathlib import Path
from orchestrator import TranslationOrchestrator

def main():
    print("🔄 BULK AUDIO TRANSLATION PROCESSOR")
    print("=" * 50)
    
    # Get audio files from audio_inpts directory
    audio_files = glob.glob("audio_inpts/*.wav")
    if not audio_files:
        print("❌ No audio files found in 'audio_inpts' directory!")
        print("💡 Run YouTube downloader first: cd stream_processing && python youtube_to_aud.py")
        return
    
    print(f"🎵 Found {len(audio_files)} audio files")
    
    # Create translation directory structure
    languages = ["en", "gu", "hi", "kn", "ta", "te"]
    for lang in languages:
        os.makedirs(f"translations/{lang}", exist_ok=True)
        print(f"📁 Created: translations/{lang}/")
    
    # Ask for target language
    print("\n🌍 Available languages: Tamil (ta), Hindi (hi), Gujarati (gu), Kannada (kn), Telugu (te)")
    target_lang = input("Enter target language (default: Tamil): ").strip() or "Tamil"
    
    # Initialize orchestrator
    print(f"\n🤖 Loading translation models for {target_lang}...")
    orchestrator = TranslationOrchestrator()
    
    print(f"\n🚀 Processing {len(audio_files)} files...")
    print("=" * 50)
    
    # Process each audio file
    successful = 0
    failed = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        filename = Path(audio_file).stem
        print(f"\n📝 [{i}/{len(audio_files)}] Processing: {filename}.wav")
        
        try:
            # Translate using orchestrator
            result = orchestrator.process_audio(
                audio_file,
                target_language=target_lang,
                save_results=False
            )
            
            if result["success"]:
                # Save English transcription
                en_file = f"translations/en/{filename}_en.txt"
                with open(en_file, "w", encoding="utf-8") as f:
                    f.write(f"Audio: {filename}.wav\n")
                    f.write(f"Time: {result['processing_time']:.2f}s\n\n")
                    f.write(result["english_text"])
                
                # Save target language translation  
                lang_codes = {"Tamil": "ta", "Hindi": "hi", "Gujarati": "gu", 
                             "Kannada": "kn", "Telugu": "te"}
                lang_code = lang_codes.get(target_lang, "ta")
                
                target_file = f"translations/{lang_code}/{filename}_{lang_code}.txt"
                with open(target_file, "w", encoding="utf-8") as f:
                    f.write(f"Audio: {filename}.wav\n")
                    f.write(f"Language: {target_lang}\n")
                    f.write(f"Time: {result['processing_time']:.2f}s\n\n")
                    f.write(f"English: {result['english_text']}\n\n")
                    f.write(f"{target_lang}: {result['final_translation']}")
                
                print(f"   ✅ Saved: {filename}_en.txt & {filename}_{lang_code}.txt")
                successful += 1
                
            else:
                print(f"   ❌ Failed: {result['error']}")
                failed += 1
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:80]}...")
            failed += 1
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎉 BULK TRANSLATION COMPLETE!")
    print(f"📊 Successful: {successful}/{len(audio_files)}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Results saved in translations/ directory")

if __name__ == "__main__":
    main() 