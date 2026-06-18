from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL_NAME = "openai/whisper-small"

print(f"Loading processor: {MODEL_NAME}")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

print(f"Loading model: {MODEL_NAME}")
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME)

print("Whisper Small loaded successfully.")
print(f"Model type: {type(model).__name__}")