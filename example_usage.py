#!/usr/bin/env python3
"""
Example usage of the sarvam_translate module
Demonstrates different ways to import and use the translation functionality
"""

# Import the translation module
from sarvam_translate import SarvamTranslator, translate_text, translate_batch, SUPPORTED_LANGUAGES

def example_simple_usage():
    """Example of simple function-based usage"""
    print("🔸 SIMPLE FUNCTION USAGE")
    print("-" * 30)
    
    # Single translation
    result = translate_text("Hello, how are you?", "Hindi")
    print(f"English → Hindi: {result}")
    
    result = translate_text("नमस्ते, आप कैसे हैं?", "English")
    print(f"Hindi → English: {result}")

def example_class_usage():
    """Example of class-based usage (recommended for multiple translations)"""
    print("\n🔸 CLASS-BASED USAGE")
    print("-" * 30)
    
    # Initialize translator once
    translator = SarvamTranslator()
    
    # Multiple translations (model stays loaded)
    texts_and_targets = [
        ("Good morning!", "Hindi"),
        ("Thank you very much!", "Tamil"),
        ("How is the weather today?", "Bengali"),
        ("Technology is amazing!", "Telugu")
    ]
    
    for text, target in texts_and_targets:
        result = translator.translate(text, target)
        print(f"{target}: {result}")

def example_batch_usage():
    """Example of batch translation"""
    print("\n🔸 BATCH TRANSLATION USAGE")
    print("-" * 30)
    
    # Prepare batch data
    batch_data = [
        ("Hello world!", "Hindi"),
        ("Good night!", "Tamil"),
        ("See you tomorrow!", "Bengali"),
        ("Have a great day!", "Telugu"),
        ("Welcome to India!", "Gujarati")
    ]
    
    # Method 1: Using function
    results = translate_batch(batch_data, show_progress=True)
    
    print("\nBatch Results:")
    for result in results:
        print(f"  {result['input']} → {result['target_language']}: {result['translation']}")

def example_class_batch_usage():
    """Example of class-based batch translation"""
    print("\n🔸 CLASS-BASED BATCH USAGE")
    print("-" * 30)
    
    translator = SarvamTranslator()
    
    batch_data = [
        ("Machine learning is fascinating!", "Hindi"),
        ("Artificial intelligence helps everyone!", "Tamil")
    ]
    
    results = translator.translate_batch(batch_data, show_progress=False)
    
    for result in results:
        print(f"  {result['translation']}")

def show_supported_languages():
    """Show supported languages"""
    print("\n🔸 SUPPORTED LANGUAGES")
    print("-" * 30)
    print(f"Total: {len(SUPPORTED_LANGUAGES)} languages")
    print("Languages:", ", ".join(SUPPORTED_LANGUAGES))

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 SARVAM TRANSLATE MODULE USAGE EXAMPLES")
    print("=" * 60)
    
    # Show supported languages first
    show_supported_languages()
    
    # Run examples
    example_simple_usage()
    example_class_usage() 
    example_batch_usage()
    example_class_batch_usage()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60) 