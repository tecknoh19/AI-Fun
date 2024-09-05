# This is a simple Whisper POC.  I plan on doing more with this in the near future

import whisper
import os

# Load the pre-trained Whisper model (you can specify "tiny", "base", "small", "medium", or "large")
model = whisper.load_model("base")  # Change to "small", "medium", or "large" for higher accuracy

# Path to your MP3 file
mp3_file_path = "2600test.mp3"

# Transcribe the audio
print(f"Transcribing {mp3_file_path}...")

# Whisper can handle mp3, wav, m4a, mp4, etc.
result = model.transcribe(mp3_file_path)

# Output the transcription
print("Transcription:")
print(result["text"])

# Optional: Save the transcription to a text file
output_txt_file = os.path.splitext(mp3_file_path)[0] + "_transcription.txt"
with open(output_txt_file, "w") as f:
    f.write(result["text"])

print(f"Transcription saved to {output_txt_file}")
