"""
Exp037-ready — AfriVoices Competition Training Script

Purpose:
Train Whisper Large-v3 + LoRA using the locked Exp026 champion configuration
on unified AfriVoices manifest files.

Status:
Ready for use once Exp036 produces:
- data/afrivoices/manifests/manifest_train.parquet
- data/afrivoices/manifests/manifest_dev.parquet
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import numpy as np
import pandas as pd
import torch
import wandb

from datasets import Audio, Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)


MODEL_ID = "openai/whisper-large-v3"
TASK = "transcribe"

TRAIN_MANIFEST = Path("data/afrivoices/manifests/manifest_train.parquet")
DEV_MANIFEST = Path("data/afrivoices/manifests/manifest_dev.parquet")

OUTPUT_DIR = "outputs/exp037-afrivoices-champion-baseline"
WANDB_PROJECT = "afrivoices-asr"
RUN_NAME = "exp037-afrivoices-champion-baseline"

MAX_STEPS = 1000
SEED = 42


def setup_seed() -> None:
    set_seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_manifest(path: Path, split_name: str) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {split_name} manifest: {path}\n"
            "Run scripts/build_afrivoices_manifest.py after Hugging Face access is approved."
        )

    df = pd.read_parquet(path)

    required = {"audio_path", "transcription", "language"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{split_name} manifest missing required columns: {missing}")

    df = df.dropna(subset=["audio_path", "transcription", "language"]).copy()
    df["audio"] = df["audio_path"]

    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    return dataset


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


def main() -> None:
    setup_seed()

    device = get_device()

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    print("Loading manifests...")
    train_ds = load_manifest(TRAIN_MANIFEST, "train")
    eval_ds = load_manifest(DEV_MANIFEST, "dev")

    print(train_ds)
    print(eval_ds)

    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        MODEL_ID,
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

    print("Processing train audio...")
    train_ds = train_ds.map(
        prepare_dataset,
        remove_columns=train_ds.column_names,
        num_proc=1,
    )

    print("Processing dev audio...")
    eval_ds = eval_ds.map(
        prepare_dataset,
        remove_columns=eval_ds.column_names,
        num_proc=1,
    )

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
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    use_fp16 = False
    use_bf16 = False

    wandb.init(
        project=WANDB_PROJECT,
        name=RUN_NAME,
        config={
            "model_id": MODEL_ID,
            "task": TASK,
            "max_steps": MAX_STEPS,
            "seed": SEED,
            "phase": "Phase 4",
            "experiment": "Exp037",
            "reference": "Exp026 champion configuration",
            "dataset": "AfriVoices unified manifest",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 2e-5,
            "warmup_steps": 10,
            "effective_batch_size": 4,
            "weight_decay": 0,
            "notes": "Competition baseline using locked Phase 3 champion configuration.",
        },
    )

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

    print("Starting AfriVoices training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    wandb.finish()

    print(f"Exp037 complete. Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
