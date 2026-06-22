"""
Experiment 016 — Whisper Large-v3 scaling test

Model   : Whisper Medium
Data    : FLEURS sw_ke
Steps   : 500
Goal    : Strengthen the baseline after Experiment 001 proved the pipeline works.
"""

import wandb
import torch
import evaluate
import random
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_dataset, Audio
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
TRAIN_SAMPLES = 3070
EVAL_SAMPLES = 100
MAX_STEPS = 1000
OUTPUT_DIR = "outputs/exp016-whisper-large-v3"
WANDB_PROJECT = "afrivoices-asr"
RUN_NAME = "exp016-whisper-large-v3"
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
        "augmentation": False,
        "language_weighting": False,
        "notes": "Experiment 016. Whisper Large-v3 scaling test using winning Exp013 recipe.",
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

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

print("Processing audio...")
train_ds = train_ds.map(
    prepare_dataset,
    remove_columns=train_ds.column_names,
    num_proc=1,
)
eval_ds = eval_ds.map(
    prepare_dataset,
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
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None

use_fp16 = False
use_bf16 = DEVICE == "cuda"

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    warmup_steps=10,
    fp16=use_fp16,
    bf16=use_bf16,
    predict_with_generate=True,
    generation_max_length=225,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
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

print(f"Experiment 016 complete. Model saved to: {OUTPUT_DIR}")
