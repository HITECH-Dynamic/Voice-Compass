"""
Experiment 001 — Smoke test

Model   : Whisper Small
Data    : FLEURS sw_KE
Steps   : 50

Goal:
Prove Trainer + W&B + checkpointing work end-to-end.
"""

import wandb
import torch
# import evaluate  # deferred until evaluation stage
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_ID = "openai/whisper-small"

LANGUAGE = "Swahili"
TASK = "transcribe"

TRAIN_SAMPLES = 100
EVAL_SAMPLES = 50

MAX_STEPS = 50

OUTPUT_DIR = "outputs/exp001-smoke-test"

WANDB_PROJECT = "afrivoices-asr"
RUN_NAME = "exp001-whisper-small-smoke"

SEED = 42


# --------------------------------------------------
# Device
# --------------------------------------------------

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")


# --------------------------------------------------
# W&B
# --------------------------------------------------

wandb.init(
    project=WANDB_PROJECT,
    name=RUN_NAME,
)

# --------------------------------------------------
# Dataset
# --------------------------------------------------

print("Loading dataset...")

raw = load_dataset(
    "google/fleurs",
    "sw_KE",
    trust_remote_code=True,
)

train_ds = raw["train"].shuffle(seed=SEED).select(range(TRAIN_SAMPLES))
eval_ds = raw["validation"].shuffle(seed=SEED).select(range(EVAL_SAMPLES))

train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16_000))
eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16_000))

print(train_ds)
print(eval_ds)