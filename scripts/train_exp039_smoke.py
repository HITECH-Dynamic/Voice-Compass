"""
Exp039 — First End-to-End Whisper Smoke Training

Purpose:
Prove the full real-data training loop:

manifest -> AfriVoicesWhisperDataset -> collator -> Whisper model -> LoRA -> Trainer -> checkpoint

This is a smoke test, not a leaderboard run.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from peft import LoraConfig, get_peft_model

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.afrivoices_whisper_dataset import AfriVoicesWhisperDataset


MODEL_NAME = "openai/whisper-large-v3"
TRAIN_MANIFEST = "data/afrivoices/manifests/manifest_train.parquet"
DEV_MANIFEST = "data/afrivoices/manifests/manifest_dev.parquet"
OUTPUT_DIR = "checkpoints/exp039_smoke"


@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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

        batch["labels"] = labels
        return batch


def main() -> None:
    print("Exp039 smoke training starting...")
    print(f"CUDA available: {torch.cuda.is_available()}")

    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language=None,
        task="transcribe",
    )

    train_dataset = AfriVoicesWhisperDataset(
        manifest_path=TRAIN_MANIFEST,
        processor_name=MODEL_NAME,
        languages=["kik", "swa"],
        max_duration=30.0,
        max_rows_per_language=1,
    )

    eval_dataset = AfriVoicesWhisperDataset(
        manifest_path=DEV_MANIFEST,
        processor_name=MODEL_NAME,
        languages=["kik", "swa"],
        max_duration=30.0,
        max_rows_per_language=1,
    )

    print(f"Train rows: {len(train_dataset)}")
    print(f"Eval rows: {len(eval_dataset)}")

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = WhisperDataCollator(processor=processor)

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=0,
        max_steps=1,
        eval_strategy="steps",
        eval_steps=1,
        save_steps=1,
        logging_steps=1,
        predict_with_generate=False,
        fp16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
        save_total_limit=1,
    )

    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print(f"Exp039 smoke training complete. Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
