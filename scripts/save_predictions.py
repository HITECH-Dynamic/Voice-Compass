from pathlib import Path

import evaluate
import pandas as pd
import torch
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_DIR = "outputs/exp006-whisper-small-2000examples"
OUTPUT_PATH = Path("results/exp007_predictions.csv")
LANGUAGE = "Swahili"
TASK = "transcribe"
EVAL_SAMPLES = 100
SEED = 42

OUTPUT_PATH.parent.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")
print(f"Loading model from {MODEL_DIR}")

processor = WhisperProcessor.from_pretrained(
    MODEL_DIR,
    language=LANGUAGE,
    task=TASK,
)

model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
model.eval()

model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None

wer_metric = evaluate.load("wer")

print("Loading validation data...")
raw = load_dataset("google/fleurs", "sw_ke")
eval_ds = raw["validation"].shuffle(seed=SEED).select(range(EVAL_SAMPLES))
eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16_000))

rows = []

print("Generating predictions...")
for i, example in enumerate(eval_ds):
    audio = example["audio"]

    inputs = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
    )

    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    prediction = processor.tokenizer.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )[0]

    reference = example["transcription"]

    rows.append(
        {
            "index": i,
            "id": example["id"],
            "reference": reference,
            "prediction": prediction,
        }
    )

    if (i + 1) % 10 == 0:
        print(f"Generated {i + 1}/{EVAL_SAMPLES}")

df = pd.DataFrame(rows)

wer = wer_metric.compute(
    predictions=df["prediction"].tolist(),
    references=df["reference"].tolist(),
)

df["experiment"] = "exp007"
df["source_model"] = MODEL_DIR
df["wer_overall"] = wer

df.to_csv(OUTPUT_PATH, index=False)

print()
print(f"Saved predictions to {OUTPUT_PATH}")
print(f"Overall WER: {wer:.4f}")
print()
print(df[["index", "reference", "prediction"]].head(10))