from faster_whisper import WhisperModel

print("Loading model...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Model loaded!")

segments, info = model.transcribe("audio/Angel may 26 vox.wav")
print("\nTranscript:")

for segment in segments:
    print(segment.text)