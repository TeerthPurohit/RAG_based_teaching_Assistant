"""
Convert MP3 audio files to text transcripts using Whisper
"""
import whisper
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def transcribe_audio(audio_file, model_name="turbo"):
    """
    Transcribe a single audio file
    
    Args:
        audio_file: Path to audio file
        model_name: Whisper model size (tiny, small, base, small, medium, large)
    
    Returns:
        Transcription result dictionary
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_file)
    return result

def batch_transcribe(audio_dir=None, output_dir=None):
    """
    Transcribe all MP3 files in a directory
    
    Args:
        audio_dir: Directory containing MP3 files
        output_dir: Directory to save transcripts
    """
    if audio_dir is None:
        audio_dir = os.path.join(os.path.dirname(__file__), "../../data/audios")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "../../data/chunks")
    os.makedirs(output_dir, exist_ok=True)
    model = whisper.load_model("turbo")
    
    for filename in os.listdir(audio_dir):
        if filename.endswith(".mp3"):
            audio_path = os.path.join(audio_dir, filename)
            print(f"Transcribing {filename}...")
            
            result = model.transcribe(audio_path)
            
            # Save transcription
            output_file = os.path.join(output_dir, filename.replace(".mp3", ".txt"))
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["text"])
            
            print(f"✓ Saved: {output_file}")

if __name__ == "__main__":
    # Single file example
    audio = os.path.join(os.path.dirname(__file__), "../../data/audios/001 - Introduction.mp3")
    if os.path.exists(audio):
        result = transcribe_audio(audio)
        print(result)
