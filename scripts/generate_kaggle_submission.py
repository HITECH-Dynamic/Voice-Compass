"""
Generate a resumable Kaggle submission for the AfriVoices East Africa ASR Hackathon.

Expected output columns:
    id, language, transcription

The script:
- loads Whisper Small plus a PEFT/LoRA adapter;
- processes test Parquet files shard by shard;
- batches inference;
- saves progress periodically;
- resumes from an existing progress CSV;
- validates the final Kaggle submission.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


LANGUAGES = {"swa", "kik", "luo", "som", "mas", "kln"}
REQUIRED_TEST_COLUMNS = {"audio", "id", "language"}
SUBMISSION_COLUMNS = ["id", "language", "transcription"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-root",
        required=True,
        help="Root directory containing the six-language test Parquet files.",
    )
    parser.add_argument(
        "--adapter-dir",
        required=True,
        help="Directory containing adapter_model.safetensors and adapter_config.json.",
    )
    parser.add_argument(
        "--base-model",
        default="openai/whisper-small",
    )
    parser.add_argument(
        "--output-csv",
        default="submissions/exp062_submission.csv",
    )
    parser.add_argument(
        "--progress-csv",
        default="submissions/exp062_submission_partial.csv",
    )
    parser.add_argument(
        "--failure-csv",
        default="submissions/exp062_submission_failures.csv",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=225)
    parser.add_argument("--num-beams", type=int, default=1)

    return parser.parse_args()


def decode_audio(audio_obj: object) -> np.ndarray:
    if not isinstance(audio_obj, dict):
        raise TypeError(f"Expected audio dict, received {type(audio_obj).__name__}")

    audio_bytes = audio_obj.get("bytes")
    audio_path = audio_obj.get("path")

    if audio_bytes is not None and len(audio_bytes) > 0:
        try:
            audio, sample_rate = sf.read(BytesIO(audio_bytes), dtype="float32")
        except Exception:
            suffix = Path(str(audio_path or "audio.wav")).suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                audio, sample_rate = librosa.load(
                    tmp.name,
                    sr=None,
                    mono=True,
                )
    elif audio_path:
        audio, sample_rate = librosa.load(
            str(audio_path),
            sr=None,
            mono=True,
        )
    else:
        raise ValueError("Audio object contains neither bytes nor path.")

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sample_rate != 16_000:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16_000,
        )

    if audio.size == 0:
        raise ValueError("Decoded audio is empty.")

    return audio.astype(np.float32)


def save_progress(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    df = df.drop_duplicates(subset=["id", "language"], keep="last")
    df.to_csv(path, index=False)


def validate_submission(
    submission: pd.DataFrame,
    expected_keys: pd.DataFrame,
) -> None:
    if list(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(
            f"Invalid columns: {list(submission.columns)}; "
            f"expected {SUBMISSION_COLUMNS}"
        )

    if submission[["id", "language"]].duplicated().any():
        raise ValueError("Submission contains duplicate (id, language) rows.")

    if submission["id"].isna().any():
        raise ValueError("Submission contains missing IDs.")

    if submission["language"].isna().any():
        raise ValueError("Submission contains missing languages.")

    unexpected_languages = set(submission["language"]) - LANGUAGES
    if unexpected_languages:
        raise ValueError(
            f"Unexpected language codes: {sorted(unexpected_languages)}"
        )

    blank_predictions = (
        submission["transcription"].isna()
        | submission["transcription"].astype(str).str.strip().eq("")
    )
    if blank_predictions.any():
        bad = submission.loc[blank_predictions, ["id", "language"]]
        raise ValueError(
            f"Submission contains {len(bad)} blank transcriptions."
        )

    expected = set(
        map(tuple, expected_keys[["id", "language"]].astype(str).to_numpy())
    )
    actual = set(
        map(tuple, submission[["id", "language"]].astype(str).to_numpy())
    )

    missing = expected - actual
    extra = actual - expected

    if missing:
        raise ValueError(f"Submission is missing {len(missing)} test rows.")

    if extra:
        raise ValueError(f"Submission contains {len(extra)} unknown rows.")


def main() -> None:
    args = parse_args()

    test_root = Path(args.test_root)
    adapter_dir = Path(args.adapter_dir)
    output_csv = Path(args.output_csv)
    progress_csv = Path(args.progress_csv)
    failure_csv = Path(args.failure_csv)

    parquet_files = sorted(test_root.rglob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found beneath {test_root}"
        )

    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"LoRA adapter not found in {adapter_dir}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Device: {device}")
    print(f"Test shards: {len(parquet_files)}")
    print(f"Adapter: {adapter_dir}")

    processor_source = (
        adapter_dir
        if (adapter_dir / "processor_config.json").exists()
        else args.base_model
    )

    processor = WhisperProcessor.from_pretrained(
        processor_source,
        language=None,
        task="transcribe",
    )

    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
    ).to(device)

    model.eval()
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    completed_rows: list[dict] = []

    if progress_csv.exists():
        previous = pd.read_csv(progress_csv)

        missing_columns = set(SUBMISSION_COLUMNS) - set(previous.columns)
        if missing_columns:
            raise ValueError(
                f"Progress CSV is missing columns: {sorted(missing_columns)}"
            )

        completed_rows = previous[SUBMISSION_COLUMNS].to_dict("records")
        print(f"Resuming from {len(completed_rows)} saved predictions.")

    completed_keys = {
        (str(row["id"]), str(row["language"]))
        for row in completed_rows
    }

    expected_parts: list[pd.DataFrame] = []
    failures: list[dict] = []
    generated_since_save = 0

    for shard_number, parquet_path in enumerate(parquet_files, start=1):
        print(
            f"\n[{shard_number}/{len(parquet_files)}] "
            f"{parquet_path.relative_to(test_root)}"
        )

        shard = pd.read_parquet(parquet_path)

        missing = REQUIRED_TEST_COLUMNS - set(shard.columns)
        if missing:
            raise ValueError(
                f"{parquet_path} is missing columns: {sorted(missing)}"
            )

        shard = shard[["audio", "id", "language"]].copy()
        shard["id"] = shard["id"].astype(str)
        shard["language"] = shard["language"].astype(str)

        expected_parts.append(shard[["id", "language"]])

        pending = shard[
            ~shard.apply(
                lambda row: (row["id"], row["language"]) in completed_keys,
                axis=1,
            )
        ].reset_index(drop=True)

        print(f"Rows: {len(shard)} | Pending: {len(pending)}")

        for start in range(0, len(pending), args.batch_size):
            batch = pending.iloc[start : start + args.batch_size]

            audio_arrays: list[np.ndarray] = []
            valid_rows: list[pd.Series] = []

            for _, row in batch.iterrows():
                try:
                    audio_arrays.append(decode_audio(row["audio"]))
                    valid_rows.append(row)
                except Exception as exc:
                    failures.append(
                        {
                            "id": row["id"],
                            "language": row["language"],
                            "shard": str(parquet_path),
                            "error": repr(exc),
                        }
                    )
                    print(
                        f"WARNING: failed to decode "
                        f"{row['language']}/{row['id']}: {exc}"
                    )

            if not valid_rows:
                continue

            features = processor.feature_extractor(
                audio_arrays,
                sampling_rate=16_000,
                return_tensors="pt",
            ).input_features.to(device=device, dtype=dtype)

            with torch.inference_mode():
                predicted_ids = model.generate(
                    features,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )

            predictions = processor.tokenizer.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )

            for row, transcription in zip(valid_rows, predictions):
                result = {
                    "id": str(row["id"]),
                    "language": str(row["language"]),
                    "transcription": transcription.strip(),
                }

                completed_rows.append(result)
                completed_keys.add((result["id"], result["language"]))
                generated_since_save += 1

            if generated_since_save >= args.save_every:
                save_progress(completed_rows, progress_csv)
                generated_since_save = 0
                print(f"Saved progress: {len(completed_keys)} predictions")

        save_progress(completed_rows, progress_csv)

        if failures:
            failure_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(failures).to_csv(failure_csv, index=False)

    expected_keys = pd.concat(expected_parts, ignore_index=True)
    submission = pd.DataFrame(completed_rows, columns=SUBMISSION_COLUMNS)
    submission = submission.drop_duplicates(
        subset=["id", "language"],
        keep="last",
    )

    validate_submission(submission, expected_keys)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_csv, index=False)

    metadata = {
        "base_model": args.base_model,
        "adapter_dir": str(adapter_dir),
        "test_root": str(test_root),
        "test_shards": len(parquet_files),
        "submission_rows": len(submission),
        "batch_size": args.batch_size,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "output_csv": str(output_csv),
    }

    metadata_path = output_csv.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSubmission complete.")
    print(f"Rows: {len(submission)}")
    print(f"Output: {output_csv}")
    print(f"Metadata: {metadata_path}")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
