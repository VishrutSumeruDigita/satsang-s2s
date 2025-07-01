#!/usr/bin/env python3
"""
Test script for YouTube audio download and chunking
Run this from the project root directory
"""

import sys
import os

# Add stream_processing to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'stream_processing'))

from youtube_to_aud import process_youtube_to_chunks

def main():
    print("🎬 YouTube Audio Download & Chunking Test")
    print("=" * 50)
    
    # Test YouTube URL
    youtube_url = "https://www.youtube.com/watch?v=qx0wR1EMjqg"
    
    try:
        print(f"📹 Processing: {youtube_url}")
        print("📁 Audio chunks will be saved in the audio_inpts directory")
        print()
        
        # Process the YouTube video
        chunks = process_youtube_to_chunks(youtube_url, output_dir="audio_inpts")
        
        print()
        print("=" * 50)
        print(f"✅ Success! Created {len(chunks)} audio chunks:")
        for i, chunk in enumerate(chunks, 1):
            filename = os.path.basename(chunk)
            print(f"   {i:2d}. {filename}")
        
        print()
        print("🎵 You can now use these audio files with your translation pipeline!")
        
    except Exception as e:
        print()
        print("=" * 50)
        print(f"❌ Error: {e}")
        print()
        print("💡 Troubleshooting:")
        print("   1. Make sure ffmpeg is installed: sudo apt install ffmpeg")
        print("   2. Install yt-dlp: pip install yt-dlp")
        print("   3. Install pydub: pip install pydub")

if __name__ == "__main__":
    main() 