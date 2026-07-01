"""
Exp043 — First Multilingual Whisper Baseline

Uses Exp042 processed train/eval parquet files.
Goal: first Kikuyu + Swahili multilingual training baseline.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model

from src.datasets.afrivoices_whisper_dataset import AfriVoicesWhisperDataset


def main():
    print("Exp043 multilingual baseline starting...")
    print(f"CUDA available: {torch.cuda.is_available()}")

    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name, language=None, task="transcribe")

    train_dataset = AfriVoicesWhisperDataset(
        manifest_path="data/processed/exp042_train.parquet",
        processor_name=model_name,
        max_duration=30.0,
        max_rows_per_language=None,
    )

    eval_dataset = AfriVoicesWhisperDataset(
        manifest_path="data/processed/exp042_eval.parquet",
        processor_name=model_name,
        max_duration=30.0,
        max_rows_per_language=None,
    )

    print(f"Train rows: {len(train_dataset)}")
    print(f"Eval rows: {len(eval_dataset)}")

    model = WhisperForConditionalGeneration.from_pretrained(model_name)
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

    training_args = Seq2SeqTrainingArguments(
        output_dir="checkpoints/exp043_multilingual_baseline",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        max_steps=5,
        eval_strategy="steps",
        eval_steps=5,
        save_steps=5,
        logging_steps=1,
        predict_with_generate=False,
        fp16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
        save_total_limit=1,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    trainer.save_model("checkpoints/exp043_multilingual_baseline")
    processor.save_pretrained("checkpoints/exp043_multilingual_baseline")

    print("Exp043 multilingual baseline complete.")


if __name__ == "__main__":
    main()
