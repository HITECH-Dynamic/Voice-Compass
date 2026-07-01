"""
Exp043 — First Multilingual Whisper Baseline
Download-efficient multilingual baseline with WER.
"""

from pathlib import Path
import sys
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


def main():
    print("Exp043 multilingual baseline starting...")
    print(f"CUDA available: {torch.cuda.is_available()}")

    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name, language=None, task="transcribe")
    wer_metric = evaluate.load("wer")

    train_dataset = AfriVoicesWhisperDataset(
        manifest_path="data/processed/exp043_train.parquet",
        processor_name=model_name,
        max_duration=30.0,
    )

    eval_dataset = AfriVoicesWhisperDataset(
        manifest_path="data/processed/exp043_eval.parquet",
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

    trainer.save_model("checkpoints/exp043_multilingual_baseline")
    processor.save_pretrained("checkpoints/exp043_multilingual_baseline")

    print("Exp043 multilingual baseline complete.")


if __name__ == "__main__":
    main()
