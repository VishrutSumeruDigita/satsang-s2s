#!/usr/bin/env python3
"""
Test script for the Speech-to-Speech Pipeline
"""

import torch
import numpy as np
import soundfile as sf
from speech_pipeline import SpeechToSpeechPipeline
import tempfile
import os

def create_test_audio():
    """Create a simple test audio file"""
    # Create a 3-second sine wave at 440 Hz (A4 note)
    duration = 3.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        sf.write(tmp_file.name, audio, sample_rate)
        return tmp_file.name

def test_system_info():
    """Test and display system information"""
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("No CUDA GPUs available - will use CPU")
    
    print()

def test_pipeline_initialization():
    """Test pipeline initialization"""
    print("=" * 50)
    print("TESTING PIPELINE INITIALIZATION")
    print("=" * 50)
    
    try:
        print("Initializing pipeline for Tamil translation...")
        pipeline = SpeechToSpeechPipeline(target_language="ta")
        print("✅ Pipeline initialized successfully!")
        return pipeline
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {str(e)}")
        return None

def test_individual_components(pipeline):
    """Test individual pipeline components"""
    if pipeline is None:
        return
        
    print("=" * 50)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("=" * 50)
    
    # Create test audio
    print("Creating test audio...")
    test_audio_path = create_test_audio()
    
    try:
        # Test ASR
        print("Testing Speech-to-Text (ASR)...")
        transcription = pipeline.transcribe_audio(test_audio_path)
        print(f"Transcription: {transcription}")
        
        # Test Translation
        print("Testing Translation...")
        test_text = "नमस्ते, कैसे हैं आप?"  # Hello, how are you?
        translated = pipeline.translate_text(test_text)
        print(f"Original: {test_text}")
        print(f"Translated: {translated}")
        
        # Test TTS
        print("Testing Text-to-Speech (TTS)...")
        output_audio = pipeline.synthesize_speech("Hello world test")
        print(f"Audio generated: {output_audio}")
        
        print("✅ All components tested successfully!")
        
    except Exception as e:
        print(f"❌ Component testing failed: {str(e)}")
    
    finally:
        # Cleanup
        try:
            os.unlink(test_audio_path)
        except:
            pass

def test_full_pipeline(pipeline):
    """Test the complete speech-to-speech pipeline"""
    if pipeline is None:
        return
        
    print("=" * 50)
    print("TESTING FULL PIPELINE")
    print("=" * 50)
    
    # Create test audio
    test_audio_path = create_test_audio()
    
    try:
        print("Running full speech-to-speech pipeline...")
        hindi_text, translated_text, output_audio = pipeline.process_speech_to_speech(test_audio_path)
        
        print(f"Hindi Transcription: {hindi_text}")
        print(f"Translated Text: {translated_text}")
        print(f"Output Audio: {output_audio}")
        
        print("✅ Full pipeline test completed successfully!")
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {str(e)}")
    
    finally:
        # Cleanup
        try:
            os.unlink(test_audio_path)
        except:
            pass

def main():
    """Main test function"""
    print("🚀 Starting Speech-to-Speech Pipeline Tests")
    print()
    
    # Test system info
    test_system_info()
    
    # Test pipeline initialization
    pipeline = test_pipeline_initialization()
    
    # Test individual components
    test_individual_components(pipeline)
    
    # Test full pipeline
    test_full_pipeline(pipeline)
    
    print("=" * 50)
    print("TESTS COMPLETED")
    print("=" * 50)
    print()
    print("If all tests passed, you can now run the Streamlit app:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main() 