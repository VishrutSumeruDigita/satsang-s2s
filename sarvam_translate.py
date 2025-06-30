#!/usr/bin/env python3
"""
Sarvam Translate - Multi-language Translation Module
Uses sarvamai/sarvam-translate model for high-quality translations

Installation Requirements:
# Clone the github repository and navigate to the project directory.
git clone https://github.com/AI4Bharat/IndicTrans2
cd IndicTrans2

# Install all the dependencies and requirements associated with the project.
source install.sh

Usage:
    # Import and use
    from sarvam_translate import SarvamTranslator, translate_text
    
    # Class-based usage (recommended for multiple translations)
    translator = SarvamTranslator()
    result = translator.translate("Hello world!", "Hindi")
    
    # Function-based usage (simple one-off translations)
    result = translate_text("Hello world!", "Hindi")
    
    # Batch translations
    results = translator.translate_batch([
        ("Hello", "Hindi"),
        ("Good morning", "Tamil"),
        ("How are you?", "Bengali")
    ])
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class SarvamTranslator:
    def __init__(self, model_name="sarvamai/sarvam-translate", device=None):
        """Initialize the Sarvam translator"""
        self.model_name = model_name
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 Loading {model_name} on {self.device}...")
        load_start = time.time()
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        load_time = time.time() - load_start
        print(f"✅ Model loaded in {load_time:.2f} seconds")
    
    def translate(self, input_text, target_language, temperature=0.01, max_new_tokens=1024):
        """
        Translate text to target language
        
        Args:
            input_text (str): Text to translate
            target_language (str): Target language (e.g., "Hindi", "English", "Tamil", etc.)
            temperature (float): Sampling temperature for generation
            max_new_tokens (int): Maximum tokens to generate
            
        Returns:
            str: Translated text
        """
        try:
            start_time = time.time()
            
            # Chat-style message prompt
            messages = [
                {"role": "system", "content": f"Translate the text below to {target_language}."},
                {"role": "user", "content": input_text}
            ]
            
            # Apply chat template to structure the conversation
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize and move input to model device
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # Generate the output
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract only the generated part (remove input tokens)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            translation_time = time.time() - start_time
            
            print(f"🌍 {target_language} translation completed in {translation_time:.2f}s")
            return output_text
            
        except Exception as e:
            print(f"❌ Translation failed: {e}")
            return f"Translation error: {str(e)}"
    
    def translate_batch(self, text_language_pairs, show_progress=True):
        """
        Translate multiple texts to different languages in batch
        
        Args:
            text_language_pairs (list): List of (text, target_language) tuples
            show_progress (bool): Whether to show progress messages
            
        Returns:
            list: List of translation results
        """
        results = []
        total = len(text_language_pairs)
        
        if show_progress:
            print(f"🔄 Processing {total} translations...")
        
        for i, (text, target_lang) in enumerate(text_language_pairs, 1):
            if show_progress:
                print(f"📝 [{i}/{total}] Translating to {target_lang}...")
            
            result = self.translate(text, target_lang)
            results.append({
                "input": text,
                "target_language": target_lang,
                "translation": result,
                "index": i
            })
        
        if show_progress:
            print(f"✅ Batch translation completed!")
        
        return results

def translate_text(input_text, target_language, model_name="sarvamai/sarvam-translate"):
    """
    Simple function interface for translation
    
    Args:
        input_text (str): Text to translate
        target_language (str): Target language name
        model_name (str): Model to use for translation
        
    Returns:
        str: Translated text
    """
    translator = SarvamTranslator(model_name)
    return translator.translate(input_text, target_language)

def main():
    """Main function with sample translations"""
    print("=" * 70)
    print("🚀 SARVAM TRANSLATE - MULTI-LANGUAGE TRANSLATION")
    print("=" * 70)
    
    # Initialize translator
    translator = SarvamTranslator()
    
    # Sample translations
    test_cases = [
        {
            "text": "Be the change you wish to see in the world.",
            "target": "Hindi",
            "description": "English to Hindi"
        },
        {
            "text": "अरे, तुम आज कैसे हो?",
            "target": "English", 
            "description": "Hindi to English"
        },
        {
            "text": "Technology has revolutionized the way we communicate.",
            "target": "Tamil",
            "description": "English to Tamil"
        },
        {
            "text": "Machine learning is transforming industries.",
            "target": "Telugu",
            "description": "English to Telugu"
        },
        {
            "text": "Good morning! How are you today?",
            "target": "Bengali",
            "description": "English to Bengali"
        }
    ]
    
    print(f"\n🔄 Running {len(test_cases)} sample translations...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"📝 Test {i}: {test_case['description']}")
        print(f"   Input:  {test_case['text']}")
        print(f"   Target: {test_case['target']}")
        
        # Perform translation
        translation = translator.translate(
            test_case['text'], 
            test_case['target']
        )
        
        print(f"   Output: {translation}")
        print("-" * 50)
    
    # Interactive mode
    print(f"\n🎯 INTERACTIVE MODE")
    print("=" * 30)
    print("You can now use the translator with:")
    print("translator.translate('your text', 'target language')")
    print("\nSupported languages: Hindi, English, Tamil, Telugu, Bengali, Gujarati, Marathi, etc.")
    
    # Example of direct function usage
    print(f"\n💡 DIRECT FUNCTION USAGE EXAMPLE:")
    sample_translation = translate_text(
        "Hello, this is a test message!", 
        "Hindi"
    )
    print(f"Direct call result: {sample_translation}")
    
    return translator

if __name__ == "__main__":
    # Run main function
    translator = main()
    
    # Keep translator available for interactive use
    print(f"\n✅ Translator ready for use!")
    print(f"📌 Use: translator.translate('your text', 'target language')") 