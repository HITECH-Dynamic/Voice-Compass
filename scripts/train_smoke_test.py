"""
Experiment 034 — Whisper Large-v3 LoRA audio augmentation study

Model   : Whisper Large-v3
Data    : FLEURS sw_ke
Steps   : 1000
Goal    : Test training-only speed and gain augmentation against the Exp026 champion.
"""

import wandb
import torch
import evaluate
import random
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_dataset, Audio
from peft import LoraConfig, get_peft_model

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

MODEL_ID = "openai/whisper-large-v3"
LANGUAGE = "Swahili"
TASK = "transcribe"
TRAIN_SAMPLES = 2570
EVAL_SAMPLES = 211
MAX_STEPS = 1000
OUTPUT_DIR = "outputs/exp034-whisper-large-v3-lora-audio-augmentation"
WANDB_PROJECT = "afrivoices-asr"
RUN_NAME = "exp034-whisper-large-v3-lora-audio-augmentation"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

set_seed(SEED)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")

wandb.init(
    project=WANDB_PROJECT,
    name=RUN_NAME,
    config={
        "model_id": MODEL_ID,
        "language": LANGUAGE,
        "task": TASK,
        "train_samples": TRAIN_SAMPLES,
        "eval_samples": EVAL_SAMPLES,
        "max_steps": MAX_STEPS,
        "device": DEVICE,
        "seed": SEED,
        "lora": False,
        "augmentation": True,
        "language_weighting": False,
        "notes": "Experiment 034. Training-only audio augmentation study using Exp026 champion baseline.",
    },
)

print("Loading dataset...")
raw = load_dataset("google/fleurs", "sw_ke")

train_ds = raw["train"].shuffle(seed=SEED).select(range(TRAIN_SAMPLES))
eval_ds = raw["validation"].shuffle(seed=SEED).select(range(EVAL_SAMPLES))

train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16_000))
eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16_000))

print(train_ds)
print(eval_ds)

print("Loading processor...")
processor = WhisperProcessor.from_pretrained(
    MODEL_ID,
    language=LANGUAGE,
    task=TASK,
)

def augment_audio_array(audio_array, sampling_rate):
    """
    Training-only augmentation.

    Applies:
    - speed perturbation: 0.9x, 1.0x, 1.1x
    - gain perturbation: -3 dB to +3 dB
    """
    audio_array = np.asarray(audio_array, dtype=np.float32)

    speed = random.choice([0.9, 1.0, 1.1])
    if speed != 1.0 and len(audio_array) > 1:
        old_indices = np.arange(len(audio_array))
        new_length = max(1, int(len(audio_array) / speed))
        new_indices = np.linspace(0, len(audio_array) - 1, new_length)
        audio_array = np.interp(new_indices, old_indices, audio_array).astype(np.float32)

    gain_db = random.uniform(-3.0, 3.0)
    gain = 10 ** (gain_db / 20.0)
    audio_array = audio_array * gain

    audio_array = np.clip(audio_array, -1.0, 1.0).astype(np.float32)

    return audio_array


def prepare_dataset(batch, augment=False):
    audio = batch["audio"]
    audio_array = audio["array"]

    if augment:
        audio_array = augment_audio_array(audio_array, audio["sampling_rate"])

    batch["input_features"] = processor.feature_extractor(
        audio_array,
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

print("Processing audio...")
train_ds = train_ds.map(
    lambda batch: prepare_dataset(batch, augment=True),
    remove_columns=train_ds.column_names,
    num_proc=1,
)
eval_ds = eval_ds.map(
    lambda batch: prepare_dataset(batch, augment=False),
    remove_columns=eval_ds.column_names,
    num_proc=1,
)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

print("Loading WER metric...")
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(
        pred_ids,
        skip_special_tokens=True,
    )

    label_str = processor.tokenizer.batch_decode(
        label_ids,
        skip_special_tokens=True,
    )

    wer = wer_metric.compute(
        predictions=pred_str,
        references=label_str,
    )

    return {"wer": round(wer, 4)}

print("Loading model...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None

use_fp16 = False
use_bf16 = False

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    warmup_steps=10,
    fp16=use_fp16,
    bf16=use_bf16,
    predict_with_generate=True,
    generation_max_length=225,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=2,
    logging_steps=10,
    report_to=["wandb"],
    run_name=RUN_NAME,
    seed=SEED,
    data_seed=SEED,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

print("Starting training...")
trainer.train()

print("Saving final model...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

wandb.finish()

print(f"Experiment 034 complete. Model saved to: {OUTPUT_DIR}")
