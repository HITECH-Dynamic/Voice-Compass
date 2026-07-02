"""
Exp043 — First Multilingual Whisper Baseline
Download-efficient multilingual baseline with WER.
"""

from pathlib import Path
import sys
import argparse
import numpy as np
import torch
import evaluate

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from src.datasets.afrivoices_whisper_dataset import AfriVoicesWhisperDataset


class WhisperDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100,
        )

        batch["labels"] = labels
        return batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="exp043_multilingual_baseline")
    parser.add_argument("--train_manifest", default="data/processed/exp043_train.parquet")
    parser.add_argument("--eval_manifest", default="data/processed/exp043_eval.parquet")
    parser.add_argument("--output_dir", default="checkpoints/exp043_multilingual_baseline")
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=5)
    parser.add_argument("--report_to", default="wandb")
    parser.add_argument("--wandb_project", default="voice-compass")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"{args.experiment_name} starting...")
    print(f"Train manifest: {args.train_manifest}")
    print(f"Eval manifest: {args.eval_manifest}")
    print(f"Output dir: {args.output_dir}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if args.report_to == "wandb":
        import os
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_NAME"] = args.experiment_name

    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name, language=None, task="transcribe")
    wer_metric = evaluate.load("wer")

    train_dataset = AfriVoicesWhisperDataset(
        manifest_path=args.train_manifest,
        processor_name=model_name,
        max_duration=30.0,
    )

    eval_dataset = AfriVoicesWhisperDataset(
        manifest_path=args.eval_manifest,
        processor_name=model_name,
        max_duration=30.0,
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

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        if pred_ids.ndim == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)

        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=1,
        predict_with_generate=False,
        fp16=torch.cuda.is_available(),
        report_to=[args.report_to] if args.report_to != "none" else [],
        run_name=args.experiment_name,
        remove_unused_columns=False,
        save_total_limit=1,
        metric_for_best_model="wer",
        greater_is_better=False,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=WhisperDataCollator(processor),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print(f"{args.experiment_name} complete.")


if __name__ == "__main__":
    main()
