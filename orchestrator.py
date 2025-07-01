#!/usr/bin/env python3
"""
Translation Orchestrator - Complete Translation Pipeline
Combines audio-to-text (Whisper) and text-to-text (Sarvam) translation

Workflow:
1. Audio (Hindi/English) → English text (Whisper)
2. English text → Target language (Sarvam)
3. Complete pipeline for multilingual audio translation

Usage:
    python orchestrator.py
    
    # Or import and use:
    from orchestrator import TranslationOrchestrator
    orchestrator = TranslationOrchestrator()
    result = orchestrator.process_audio("audio.wav", target_language="Hindi")
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
import soundfile as sf
import numpy as np
import os
import time

class TranslationOrchestrator:
    """Complete translation pipeline: Audio → English → Target Language"""
    
    def __init__(self, whisper_model="openai/whisper-medium", sarvam_model="sarvamai/sarvam-translate"):
        """Initialize both Whisper and Sarvam models"""
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {self.device}")
        
        # Load Whisper model (for audio → English)
        print(f"📥 Loading Whisper model: {whisper_model}")
        start_time = time.time()
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model).to(self.device)
        whisper_time = time.time() - start_time
        print(f"✅ Whisper loaded in {whisper_time:.2f}s")
        
        # Load Sarvam model (for English → target language)
        print(f"📥 Loading Sarvam model: {sarvam_model}")
        start_time = time.time()
        self.sarvam_tokenizer = AutoTokenizer.from_pretrained(sarvam_model)
        self.sarvam_model = AutoModelForCausalLM.from_pretrained(sarvam_model).to(self.device)
        sarvam_time = time.time() - start_time
        print(f"✅ Sarvam loaded in {sarvam_time:.2f}s")
        
        print(f"🎉 All models ready!")
    
    def audio_to_english(self, audio_file_path):
        """Convert Hindi/English audio to English text using Whisper"""
        try:
            print(f"🎵 Processing audio: {audio_file_path}")
            
            # Load and preprocess audio
            audio_data, sample_rate = sf.read(audio_file_path)
            
            # Ensure mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample to 16kHz
            if sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            # Process for Whisper
            inputs = self.whisper_processor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True
            ).to(self.device)
            
            # Generate English translation
            forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(task="translate")
            
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(
                    inputs["input_features"],
                    attention_mask=inputs.get("attention_mask"),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=400,
                    num_beams=2,
                    do_sample=False,
                    use_cache=True
                )
            
            english_text = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print(f"🔤 English text: {english_text}")
            
            return {
                "success": True,
                "english_text": english_text,
                "audio_file": audio_file_path
            }
            
        except Exception as e:
            print(f"❌ Audio processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "audio_file": audio_file_path
            }
    
    def english_to_target(self, english_text, target_language):
        """Translate English text to target language using Sarvam"""
        try:
            print(f"🌐 Translating to {target_language}: {english_text[:50]}...")
            
            # Create chat messages
            messages = [
                {"role": "system", "content": f"Translate the text below to {target_language}."},
                {"role": "user", "content": english_text}
            ]
            
            # Apply chat template
            text = self.sarvam_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            model_inputs = self.sarvam_tokenizer([text], return_tensors="pt").to(self.device)
            
            # Generate translation
            with torch.no_grad():
                generated_ids = self.sarvam_model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.01,
                    num_return_sequences=1,
                    pad_token_id=self.sarvam_tokenizer.eos_token_id
                )
            
            # Extract translated text
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            translated_text = self.sarvam_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            print(f"📝 {target_language} translation: {translated_text}")
            
            return {
                "success": True,
                "translated_text": translated_text,
                "target_language": target_language,
                "english_text": english_text
            }
            
        except Exception as e:
            print(f"❌ Text translation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "english_text": english_text,
                "target_language": target_language
            }
    
    def process_audio(self, audio_file_path, target_language="Hindi", save_results=True, output_dir="orchestrator_results"):
        """
        Complete pipeline: Audio → English → Target Language
        
        Args:
            audio_file_path (str): Path to audio file
            target_language (str): Target language for final translation
            save_results (bool): Whether to save results to files
            output_dir (str): Directory to save results
            
        Returns:
            dict: Complete processing results
        """
        start_time = time.time()
        
        print(f"\n🚀 TRANSLATION PIPELINE: {audio_file_path} → {target_language}")
        print("=" * 60)
        
        # Step 1: Audio → English
        print(f"\n📍 STEP 1: Audio → English")
        audio_result = self.audio_to_english(audio_file_path)
        
        if not audio_result["success"]:
            return {
                "success": False,
                "error": f"Audio processing failed: {audio_result['error']}",
                "audio_file": audio_file_path,
                "target_language": target_language
            }
        
        english_text = audio_result["english_text"]
        
        # Step 2: English → Target Language
        print(f"\n📍 STEP 2: English → {target_language}")
        text_result = self.english_to_target(english_text, target_language)
        
        if not text_result["success"]:
            return {
                "success": False,
                "error": f"Text translation failed: {text_result['error']}",
                "audio_file": audio_file_path,
                "english_text": english_text,
                "target_language": target_language
            }
        
        # Combine results
        total_time = time.time() - start_time
        
        final_result = {
            "success": True,
            "audio_file": audio_file_path,
            "target_language": target_language,
            "english_text": english_text,
            "final_translation": text_result["translated_text"],
            "processing_time": total_time,
            "pipeline": "Audio → English → Target Language"
        }
        
        # Save results if requested
        if save_results:
            self._save_results(final_result, output_dir)
        
        print(f"\n✅ PIPELINE COMPLETE!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"🎯 Final result: {text_result['translated_text']}")
        
        return final_result
    
    def _save_results(self, result, output_dir):
        """Save pipeline results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save final translation
        with open(f"{output_dir}/FINAL_TRANSLATION.txt", "w", encoding="utf-8") as f:
            f.write(f"TRANSLATION PIPELINE RESULT\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Audio File: {result['audio_file']}\n")
            f.write(f"Target Language: {result['target_language']}\n")
            f.write(f"Processing Time: {result['processing_time']:.2f}s\n")
            f.write(f"Pipeline: {result['pipeline']}\n\n")
            f.write(f"English Text:\n{result['english_text']}\n\n")
            f.write(f"Final Translation ({result['target_language']}):\n{result['final_translation']}\n")
        
        print(f"📁 Results saved to: {output_dir}/FINAL_TRANSLATION.txt")
    
    def process_batch(self, audio_files, target_language="Hindi", save_results=True, output_dir="batch_orchestrator"):
        """Process multiple audio files through the complete pipeline"""
        results = []
        total = len(audio_files)
        
        print(f"🔄 Processing {total} audio files → {target_language}")
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n📝 [{i}/{total}] Processing: {audio_file}")
            
            file_output_dir = f"{output_dir}/file_{i}" if save_results else output_dir
            result = self.process_audio(audio_file, target_language, save_results, file_output_dir)
            result["batch_index"] = i
            results.append(result)
            
            if result["success"]:
                print(f"   ✅ Success: {result['final_translation'][:60]}...")
            else:
                print(f"   ❌ Failed: {result['error']}")
        
        successful = len([r for r in results if r["success"]])
        print(f"\n🎉 Batch completed: {successful}/{total} successful")
        
        return results

def translate_audio_file(audio_file_path, target_language="Hindi", save_results=True):
    """
    Simple function interface for complete audio translation pipeline
    
    Args:
        audio_file_path (str): Path to audio file
        target_language (str): Target language for translation
        save_results (bool): Whether to save results
        
    Returns:
        dict: Translation results
    """
    orchestrator = TranslationOrchestrator()
    return orchestrator.process_audio(audio_file_path, target_language, save_results)

def main():
    """Main function for model initialization and testing"""
    print("=" * 70)
    print("🎼 TRANSLATION ORCHESTRATOR - MODEL SETUP")
    print("=" * 70)
    print("Initializing models for Audio → English → Target Language pipeline")
    
    try:
        print("\n🚀 INITIALIZING TRANSLATION ORCHESTRATOR...")
        print("This will download and load all required models:")
        print("  • Whisper (for audio → English)")
        print("  • Sarvam AI (for English → target language)")
        print()
        
        # Initialize orchestrator (this downloads/loads models)
        start_time = time.time()
        orchestrator = TranslationOrchestrator()
        init_time = time.time() - start_time
        
        print(f"\n✅ MODEL INITIALIZATION COMPLETE!")
        print(f"⏱️  Total initialization time: {init_time:.2f} seconds")
        
        # Check if we have any audio files to test with
        test_files = []
        
        # Look for test files in different locations
        possible_test_files = [
            "indic_tts_out.wav",
            "test_audio.wav", 
            "audio_inpts/*.wav"
        ]
        
        import glob
        for pattern in possible_test_files:
            if "*" in pattern:
                found_files = glob.glob(pattern)
                if found_files:
                    test_files.extend(found_files[:1])  # Take first file
            else:
                if os.path.exists(pattern):
                    test_files.append(pattern)
        
        if test_files:
            test_file = test_files[0]
            print(f"\n🧪 TESTING WITH: {test_file}")
            print("Testing translation pipeline...")
            
            # Quick test with Tamil
            result = orchestrator.process_audio(
                test_file,
                target_language="Tamil", 
                save_results=False
            )
            
            if result["success"]:
                print(f"\n🎯 TEST SUCCESSFUL!")
                print(f"English: {result['english_text'][:80]}...")
                print(f"Tamil: {result['final_translation'][:80]}...")
                print(f"Processing time: {result['processing_time']:.2f}s")
            else:
                print(f"\n⚠️  Test failed: {result['error']}")
                print("Models are loaded but test audio couldn't be processed.")
        
        else:
            print(f"\n💡 No test audio files found.")
            print("Models are ready! You can now:")
            print("  • Run bulk_translate.py to process audio files")
            print("  • Use the Streamlit app: streamlit run app.py")
        
        print(f"\n📚 READY FOR USE:")
        print("=" * 30)
        print("• Bulk translation: python bulk_translate.py")
        print("• Web interface: streamlit run app.py")
        print("• Direct usage:")
        print("  from orchestrator import TranslationOrchestrator")
        print("  orchestrator = TranslationOrchestrator()")
        print("  result = orchestrator.process_audio('audio.wav', 'Tamil')")
        
        print(f"\n🌟 Supported target languages:")
        print("Hindi, Tamil, Telugu, Bengali, Gujarati, Marathi, English, etc.")
        
    except Exception as e:
        print(f"\n❌ ERROR DURING INITIALIZATION:")
        print(f"Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("• Make sure all dependencies are installed: pip install -r requirements.txt")
        print("• Install yt-dlp and pydub: pip install yt-dlp pydub")
        print("• Check internet connection (models need to be downloaded)")
        print("• Ensure sufficient disk space for model files")

if __name__ == "__main__":
    main()

# Module exports
__all__ = [
    'TranslationOrchestrator',
    'translate_audio_file'
]

__version__ = "1.0.0"
