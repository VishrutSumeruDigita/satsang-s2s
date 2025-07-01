import streamlit as st
import tempfile
import os
from orchestrator import TranslationOrchestrator
import time
from audio_recorder_streamlit import audio_recorder
import base64

# Define supported languages for the orchestrator
SUPPORTED_LANGUAGES = {
    "Tamil": "Tamil",
    "Telugu": "Telugu", 
    "Kannada": "Kannada",
    "Malayalam": "Malayalam",
    "Gujarati": "Gujarati",
    "Punjabi": "Punjabi",
    "Bengali": "Bengali",
    "Odia": "Odia",
    "Assamese": "Assamese", 
    "Marathi": "Marathi",
    "English": "English",
    "French": "French",
    "German": "German",
    "Russian": "Russian",
    "Chinese (Simplified)": "Chinese",
    "Japanese": "Japanese",
    "Hindi": "Hindi"
}

# Page configuration
st.set_page_config(
    page_title="Audio Translation Pipeline",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #374151;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #f0fdf4;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bbf7d0;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #fef2f2;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #fecaca;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Header
st.markdown('<div class="main-header">🗣️ Audio Translation Pipeline</div>', 
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Translate Hindi/English audio to text in multiple languages using AI</div>', 
            unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Language selection with categories
    st.markdown("**Target Language:**")
    
    # Create language categories
    indian_languages = [
        "Tamil", "Telugu", "Kannada", "Malayalam", "Gujarati", 
        "Punjabi", "Bengali", "Odia", "Assamese", "Marathi"
    ]
    
    international_languages = [
        "English", "French", "German", "Russian", 
        "Chinese (Simplified)", "Japanese"
    ]
    
    # Language category selection
    language_category = st.radio(
        "Choose category:",
        ["🇮🇳 Indian Languages", "🌍 International Languages"],
        horizontal=True
    )
    
    if language_category == "🇮🇳 Indian Languages":
        target_language_name = st.selectbox(
            "Select Indian Language:",
            options=indian_languages,
            index=0
        )
    else:
        target_language_name = st.selectbox(
            "Select International Language:",
            options=international_languages,
            index=0
        )
    target_language_code = SUPPORTED_LANGUAGES[target_language_name]
    
    st.markdown("---")
    
    # Model information
    st.markdown("### 🤖 Model Information")
    st.markdown("""
    - **ASR**: Whisper (OpenAI)
    - **Translation**: Sarvam AI
    - **Pipeline**: Audio → English → Target Language
    - **Device**: GPU (if available)
    """)
    
    st.markdown("---")
    
    # Features
    st.markdown("### ✨ Features")
    st.markdown("""
    - ✅ Local processing
    - ✅ Audio transcription to English
    - ✅ Multi-language translation
    - ✅ Audio recording
    - ✅ File upload support
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🎤 Input Audio")
    
    # Audio input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Record Audio", "Upload Audio File"],
        horizontal=True
    )
    
    audio_data = None
    audio_file_path = None
    
    if input_method == "Record Audio":
        st.markdown("### Record your Hindi/English speech:")
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#6c757d",
            icon_name="microphone",
            icon_size="2x",
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            # Save recorded audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                audio_file_path = tmp_file.name
                
    else:  # Upload Audio File
        uploaded_file = st.file_uploader(
            "Upload Hindi/English audio file:",
            type=["wav", "mp3", "m4a", "ogg"]
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                audio_file_path = tmp_file.name

with col2:
    st.header("🔄 Results")
    
    # Load model button
    if not st.session_state.model_loaded:
        if st.button("🚀 Load Models", type="primary"):
            with st.spinner("Loading AI models... This may take a few minutes..."):
                try:
                    st.session_state.orchestrator = TranslationOrchestrator()
                    st.session_state.model_loaded = True
                    st.success("✅ Models loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading models: {str(e)}")
    else:
        st.success("✅ Models are ready!")
        
        # Process audio button
        if audio_file_path and st.button("🎯 Process Speech", type="primary"):
            with st.spinner("Processing audio translation..."):
                try:
                    # Process the audio using orchestrator
                    result = st.session_state.orchestrator.process_audio(
                        audio_file_path, 
                        target_language=target_language_name,
                        save_results=False
                    )
                    
                    if result["success"]:
                        # Display results
                        st.markdown("### 📝 Results:")
                        
                        # English transcription
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown("**English Transcription:**")
                        st.write(result["english_text"])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Translation
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown(f"**{target_language_name} Translation:**")
                        st.write(result["final_translation"])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Processing info
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown("**Processing Info:**")
                        st.write(f"⏱️ Processing time: {result['processing_time']:.2f} seconds")
                        st.write(f"🔄 Pipeline: {result['pipeline']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Download button for results
                        results_text = f"""TRANSLATION RESULTS
==================

Audio File: {result['audio_file']}
Target Language: {result['target_language']}
Processing Time: {result['processing_time']:.2f}s
Pipeline: {result['pipeline']}

English Transcription:
{result['english_text']}

{result['target_language']} Translation:
{result['final_translation']}
"""
                        st.download_button(
                            label="⬇️ Download Translation Results",
                            data=results_text,
                            file_name=f"translation_results_{target_language_code}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.markdown('<div class="error-box">', unsafe_allow_html=True)
                        st.error(f"❌ Translation failed: {result['error']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Cleanup temporary files
                    try:
                        if os.path.exists(audio_file_path):
                            os.unlink(audio_file_path)
                    except:
                        pass
                        
                except Exception as e:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.error(f"❌ Error processing audio: {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)

# Instructions section
st.markdown("---")
st.header("📋 How to Use")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown("### 1️⃣ Load Models")
    st.write("Click 'Load Models' to initialize the AI models. This is a one-time setup.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown("### 2️⃣ Input Audio")
    st.write("Record Hindi/English speech using the microphone or upload an audio file.")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown("### 3️⃣ Get Results")
    st.write("Click 'Process Speech' to get English transcription and target language translation.")
    st.markdown('</div>', unsafe_allow_html=True)

# System information
with st.expander("🖥️ System Information"):
    import torch
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Device Information:**")
        st.write(f"- CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.write(f"- GPU Count: {torch.cuda.device_count()}")
            st.write(f"- Current Device: {torch.cuda.get_device_name()}")
        st.write(f"- PyTorch Version: {torch.__version__}")
    
    with col2:
        st.write("**Supported Languages:**")
        for lang_name, lang_code in SUPPORTED_LANGUAGES.items():
            st.write(f"- {lang_name} ({lang_code})")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6b7280;'>"
    "🚀 Built with Streamlit • 🤖 Powered by Whisper & Sarvam AI • 💻 Local Translation Pipeline"
    "</div>",
    unsafe_allow_html=True
) 