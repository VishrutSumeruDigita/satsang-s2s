## This script takes in youtube audio input and splits it into audio chunks of 10 seconds each

import os
import yt_dlp

def download_youtube_audio(url, output_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(f"Error downloading YouTube audio: {e}")
        return False

def split_audio_into_chunks(audio_file_path, chunk_duration=10, output_dir="../audio_inpts"):
    """
    Split audio file into chunks of specified duration
    
    Args:
        audio_file_path (str): Path to the audio file
        chunk_duration (int): Duration of each chunk in seconds
        output_dir (str): Directory to save audio chunks (default: audio_inpts directory)
    
    Returns:
        list: List of paths to the generated audio chunks
    """
    import librosa
    from pydub import AudioSegment
    
    # Create output directory - use absolute path to avoid issues
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio file
    audio = AudioSegment.from_wav(audio_file_path)
    
    # Calculate chunk parameters
    chunk_length_ms = chunk_duration * 1000
    total_length_ms = len(audio)
    num_chunks = (total_length_ms + chunk_length_ms - 1) // chunk_length_ms
    
    chunk_paths = []
    
    for i in range(num_chunks):
        start_ms = i * chunk_length_ms
        end_ms = min((i + 1) * chunk_length_ms, total_length_ms)
        
        # Extract chunk
        chunk = audio[start_ms:end_ms]
        
        # Save chunk
        chunk_filename = f"chunk_{i+1:03d}.wav"
        chunk_path = os.path.join(output_dir, chunk_filename)
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
        
        print(f"Created chunk {i+1}/{num_chunks}: {chunk_filename}")
    
    return chunk_paths

def process_youtube_to_chunks(youtube_url, output_dir="../audio_inpts"):
    """
    Download YouTube audio and split into chunks
    
    Args:
        youtube_url (str): YouTube URL to download
        output_dir (str): Directory to save audio chunks (default: audio_inpts directory)
    
    Returns:
        list: List of paths to the generated audio chunks
    """
    # Create output directory - use absolute path to avoid issues
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Saving audio chunks to: {output_dir}")
    
    # Download YouTube audio
    temp_audio_path = os.path.join(output_dir, "temp_audio")  # Don't include .wav extension
    print(f"📥 Downloading audio to: {temp_audio_path}")
    if not download_youtube_audio(youtube_url, temp_audio_path):
        raise Exception("Failed to download YouTube audio")
    
    # yt-dlp will add .wav extension, so update the path
    temp_audio_path = temp_audio_path + ".wav"
    
    # Check if the downloaded file exists
    if not os.path.exists(temp_audio_path):
        raise Exception(f"Downloaded audio file not found at: {temp_audio_path}")
    
    # Split into chunks
    chunk_paths = split_audio_into_chunks(temp_audio_path, output_dir=output_dir)
    
    # Clean up temporary file
    try:
        os.remove(temp_audio_path)
        print(f"🗑️  Cleaned up temporary file: {temp_audio_path}")
    except Exception as e:
        print(f"⚠️  Could not remove temporary file: {e}")
    # Save chunks with YouTube video ID and sequential numbering
    import re
    
    # Extract YouTube video ID from URL
    video_id_match = re.search(r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)', youtube_url)
    if video_id_match:
        video_id = video_id_match.group(1)
    else:
        video_id = "unknown"
    
    # Rename chunks with video ID and sequential numbering
    renamed_chunks = []
    for i, chunk_path in enumerate(chunk_paths):
        # Generate new filename with video ID and sequential number
        new_filename = f"{video_id}_{i+1:03d}.wav"
        new_path = os.path.join(output_dir, new_filename)
        
        # Rename the file
        try:
            os.rename(chunk_path, new_path)
            renamed_chunks.append(new_path)
            print(f"Renamed chunk {i+1}: {new_filename}")
        except Exception as e:
            print(f"Error renaming chunk {i+1}: {e}")
            renamed_chunks.append(chunk_path)  # Keep original path if rename fails
    return renamed_chunks

if __name__ == "__main__":
    # Example usage - saves 10-second audio chunks to audio_inpts directory
    youtube_url = "https://www.youtube.com/watch?v=qx0wR1EMjqg"
    try:
        chunks = process_youtube_to_chunks(youtube_url)
        print(f"Successfully created {len(chunks)} audio chunks in audio_inpts directory")
    except Exception as e:
        print(f"Error processing YouTube audio: {e}")

