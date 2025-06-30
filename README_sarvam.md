# Sarvam Translate Module

A Python module for multi-language translation using the Sarvam AI translation model.

## Installation

```bash
# Clone the github repository and navigate to the project directory
git clone https://github.com/AI4Bharat/IndicTrans2
cd IndicTrans2

# Install all the dependencies and requirements associated with the project
source install.sh

# Install additional requirements
pip install torch transformers
```

## Quick Start

### Import the module

```python
from sarvam_translate import SarvamTranslator, translate_text, translate_batch, SUPPORTED_LANGUAGES
```

### Simple Usage (Function-based)

```python
# Single translation
result = translate_text("Hello, how are you?", "Hindi")
print(result)  # नमस्ते, आप कैसे हैं?

# Reverse translation
result = translate_text("नमस्ते, आप कैसे हैं?", "English")
print(result)  # Hello, how are you?
```

### Class-based Usage (Recommended for multiple translations)

```python
# Initialize translator once (model stays loaded)
translator = SarvamTranslator()

# Multiple translations
result1 = translator.translate("Good morning!", "Hindi")
result2 = translator.translate("Thank you!", "Tamil")
result3 = translator.translate("How are you?", "Bengali")

print(result1)  # सुप्रभात!
print(result2)  # நன்றி!
print(result3)  # তুমি কেমন আছো?
```

### Batch Translation

```python
# Function-based batch translation
batch_data = [
    ("Hello world!", "Hindi"),
    ("Good night!", "Tamil"),
    ("See you tomorrow!", "Bengali")
]

results = translate_batch(batch_data)

for result in results:
    print(f"{result['input']} → {result['translation']}")

# Class-based batch translation
translator = SarvamTranslator()
results = translator.translate_batch(batch_data, show_progress=False)
```

## API Reference

### SarvamTranslator Class

```python
translator = SarvamTranslator(model_name="sarvamai/sarvam-translate", device=None)
```

#### Methods

- `translate(input_text, target_language, temperature=0.01, max_new_tokens=1024)`
  - Translate text to target language
  - Returns: `str` - Translated text

- `translate_batch(text_language_pairs, show_progress=True)`
  - Translate multiple texts in batch
  - Args: List of `(text, target_language)` tuples
  - Returns: List of result dictionaries

### Functions

- `translate_text(input_text, target_language, model_name="sarvamai/sarvam-translate")`
  - Simple function for single translation
  - Returns: `str` - Translated text

- `translate_batch(text_language_pairs, model_name="sarvamai/sarvam-translate", show_progress=True)`
  - Simple function for batch translation
  - Returns: List of result dictionaries

### Constants

- `SUPPORTED_LANGUAGES` - List of supported language names

## Supported Languages

Hindi, English, Tamil, Telugu, Bengali, Gujarati, Marathi, Kannada, Malayalam, Punjabi, Urdu, Assamese, Oriya, Nepali, Sanskrit

## Examples

See `example_usage.py` for comprehensive examples.

```python
# Run examples
python example_usage.py

# Run the main module (includes test translations)
python sarvam_translate.py
```

## Performance Tips

1. **Use class-based approach** for multiple translations to avoid reloading the model
2. **Use batch translation** for processing multiple texts efficiently
3. **GPU acceleration** is automatically used if available
4. **Adjust temperature** (0.01-1.0) for translation creativity vs accuracy

## Error Handling

The module includes comprehensive error handling:

```python
try:
    result = translate_text("Hello", "Hindi")
    print(result)
except Exception as e:
    print(f"Translation failed: {e}")
```

All functions return error messages instead of raising exceptions for robustness. 