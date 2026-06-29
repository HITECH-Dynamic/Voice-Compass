"""
Exp036B — AfriVoices Unified Manifest Builder

Creates normalized metadata manifests for the six official AfriVoices
competition languages from the available Hugging Face source repositories.

Outputs:
- data/afrivoices/manifests/manifest_train.parquet
- data/afrivoices/manifests/manifest_dev.parquet
- data/afrivoices/manifests/manifest_test.parquet
- reports/dataset_analysis/manifest_summary.csv

This script builds metadata/audio-reference manifests only. It does not
extract all audio bytes. Audio resolution happens later in the dataset loader.
"""

from pathlib import Path
import json
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files


OUTPUT_DIR = Path("data/afrivoices/manifests")
REPORT_DIR = Path("reports/dataset_analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SWAHILI_REPO = "DigitalUmuganda/Afrivoice_Swahili"

ANV_SOURCES = {
    "kik": {"name": "Kikuyu", "repo": "Anv-ke/kikuyu"},
    "luo": {"name": "Luo / Dholuo", "repo": "Anv-ke/Dholuo"},
    "kln": {"name": "Kalenjin", "repo": "Anv-ke/Kalenjin"},
    "mas": {"name": "Maasai", "repo": "Anv-ke/Maasai"},
    "som": {"name": "Somali", "repo": "Anv-ke/Somali"},
}

KEEP_COLUMNS = [
    "id",
    "language",
    "language_name",
    "source_repo",
    "source_name",
    "split",
    "raw_split",
    "speech_type",
    "audio_source_type",
    "audio_ref",
    "audio_filename",
    "audio_parquet_candidates",
    "transcription",
    "translation",
    "duration_seconds",
    "speaker_id",
    "dialect",
    "domain",
    "gender",
    "age",
    "county",
    "metadata",
]


def read_csv_safe(path: str | Path) -> pd.DataFrame:
    """Read CSV files that may have mixed encodings or malformed lines."""
    for encoding in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                engine="python",
                on_bad_lines="skip",
            )
        except UnicodeDecodeError:
            continue

    return pd.read_csv(
        path,
        encoding="latin1",
        encoding_errors="replace",
        engine="python",
        on_bad_lines="skip",
    )


def normalize_split(raw_split: str) -> str:
    if raw_split == "dev":
        return "dev"
    if raw_split == "dev_test":
        return "test"
    if raw_split == "test":
        return "test"
    if raw_split == "train":
        return "train"
    return raw_split


def safe_series(df: pd.DataFrame, column: str, default: Any = "") -> pd.Series:
    """Return an existing column or a same-length default Series."""
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def first_existing_series(
    df: pd.DataFrame,
    columns: list[str],
    default: Any = "",
) -> pd.Series:
    for column in columns:
        if column in df.columns:
            return df[column]
    return pd.Series([default] * len(df), index=df.index)


def json_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def build_audio_map(files: list[str]) -> dict[tuple[str, str], list[str]]:
    audio_files = sorted(
        f for f in files if "/audios/" in f and f.endswith(".parquet")
    )
    audio_map: dict[tuple[str, str], list[str]] = {}

    for audio_file in audio_files:
        parts = audio_file.split("/")
        if len(parts) >= 4:
            raw_split = parts[0]
            speech_type = parts[1]
            audio_map.setdefault((raw_split, speech_type), []).append(audio_file)

    return audio_map


def normalize_anv_transcripts(
    tx: pd.DataFrame,
    iso: str,
    info: dict[str, str],
    raw_split: str,
    speech_type: str,
    audio_candidates: list[str],
) -> pd.DataFrame:
    media_path = first_existing_series(tx, ["mediaPathId", "audio_ref"], "")
    media_path_str = media_path.astype(str)

    transcription = first_existing_series(
        tx,
        ["actualSentence", "transcription", "sentence", "text"],
        "",
    )

    out = pd.DataFrame(
        {
            "id": (
                media_path_str
                .str.replace("VOICE_COLLECTION/", "", regex=False)
                .str.replace(".wav", "", regex=False)
            ),
            "language": iso,
            "language_name": info["name"],
            "source_repo": info["repo"],
            "source_name": info["repo"],
            "split": normalize_split(raw_split),
            "raw_split": raw_split,
            "speech_type": speech_type,
            "audio_source_type": "parquet_bytes",
            "audio_ref": media_path_str,
            "audio_filename": media_path_str.str.split("/").str[-1],
            "audio_parquet_candidates": json_list(audio_candidates),
            "transcription": transcription.astype(str),
            "translation": first_existing_series(tx, ["translatedText"], "").astype(str),
            "duration_seconds": pd.to_numeric(
                first_existing_series(tx, ["duration"], None),
                errors="coerce",
            ),
            "speaker_id": first_existing_series(tx, ["recorder_uuid"], "").astype(str),
            "dialect": first_existing_series(
                tx,
                ["sentenceDialect", "dialect"],
                "",
            ).astype(str),
            "domain": first_existing_series(tx, ["domain", "topic"], "").astype(str),
            "gender": first_existing_series(tx, ["gender"], "").astype(str),
            "age": first_existing_series(tx, ["ownerAge"], "").astype(str),
            "county": first_existing_series(tx, ["countyName"], "").astype(str),
            "metadata": "{}",
        }
    )

    return out[KEEP_COLUMNS]


def load_anv_source(iso: str, info: dict[str, str]) -> pd.DataFrame:
    repo = info["repo"]
    files = list_repo_files(repo_id=repo, repo_type="dataset")

    transcript_files = sorted(
        f for f in files if f.endswith("/files/transcripts.csv")
    )
    audio_map = build_audio_map(files)

    frames = []

    for tx_file in transcript_files:
        parts = tx_file.split("/")
        raw_split = parts[0]
        speech_type = parts[1] if len(parts) > 1 else "unknown"

        tx_path = hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=tx_file,
        )
        tx = read_csv_safe(tx_path)

        meta_file = tx_file.replace("transcripts.csv", "meta.csv")
        try:
            meta_path = hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename=meta_file,
            )
            meta = read_csv_safe(meta_path)

            if "recorder_uuid" in tx.columns and "recorder_uuid" in meta.columns:
                tx = tx.merge(
                    meta,
                    on="recorder_uuid",
                    how="left",
                    suffixes=("", "_speaker"),
                )
        except Exception as exc:
            print(f"WARNING: Could not merge metadata for {tx_file}: {exc}")

        audio_candidates = audio_map.get((raw_split, speech_type), [])
        normalized = normalize_anv_transcripts(
            tx=tx,
            iso=iso,
            info=info,
            raw_split=raw_split,
            speech_type=speech_type,
            audio_candidates=audio_candidates,
        )
        frames.append(normalized)

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def infer_swahili_split(folder: str) -> str:
    if "_train" in folder:
        return "train"
    if "_dev" in folder:
        return "dev"
    if "_test" in folder:
        return "test"
    return "unknown"


def load_swahili_source() -> pd.DataFrame:
    files = list_repo_files(repo_id=SWAHILI_REPO, repo_type="dataset")
    manifest_files = sorted(
        f for f in files if f.endswith(".jsonl") and "manifest" in f
    )

    rows = []

    for manifest_file in manifest_files:
        folder = manifest_file.split("/")[0]
        split = infer_swahili_split(folder)

        path = hf_hub_download(
            repo_id=SWAHILI_REPO,
            repo_type="dataset",
            filename=manifest_file,
        )

        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                item = json.loads(line)

                audio_filepath = item.get("audio_filepath", "")
                key = item.get("key") or Path(audio_filepath).stem
                transcription = (
                    item.get("normalized_transcription")
                    or item.get("transcription")
                    or ""
                )

                rows.append(
                    {
                        "id": f"swa_{key}",
                        "language": "swa",
                        "language_name": "Swahili",
                        "source_repo": SWAHILI_REPO,
                        "source_name": "Afrivoice Swahili",
                        "split": split,
                        "raw_split": folder,
                        "speech_type": "spontaneous",
                        "audio_source_type": "swahili_tar_ref",
                        "audio_ref": f"{folder}/{audio_filepath}",
                        "audio_filename": audio_filepath,
                        "audio_parquet_candidates": "[]",
                        "transcription": transcription,
                        "translation": "",
                        "duration_seconds": pd.to_numeric(
                            item.get("duration", None),
                            errors="coerce",
                        ),
                        "speaker_id": item.get("voice_creator_id", ""),
                        "dialect": item.get("locale", ""),
                        "domain": item.get("category", ""),
                        "gender": item.get("gender", ""),
                        "age": item.get("age_group", ""),
                        "county": item.get("location", ""),
                        "metadata": json.dumps(
                            {
                                "manifest_file": manifest_file,
                                "dir_path": item.get("dir_path", ""),
                                "chunk_id": item.get("chunk_id", ""),
                                "project_name": item.get("project_name", ""),
                            },
                            ensure_ascii=False,
                        ),
                    }
                )

    return pd.DataFrame(rows, columns=KEEP_COLUMNS)


def validate_manifest(df: pd.DataFrame) -> None:
    required = ["id", "language", "split", "audio_ref", "transcription"]
    missing_cols = [column for column in required if column not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    allowed_languages = {"swa", "kik", "luo", "som", "mas", "kln"}
    bad_languages = set(df["language"].dropna().unique()) - allowed_languages

    if bad_languages:
        raise ValueError(f"Unexpected language codes: {bad_languages}")

    train_dev = df[df["split"].isin(["train", "dev"])]
    missing_transcripts = (
        train_dev["transcription"].isna().sum()
        + (train_dev["transcription"].astype(str).str.strip() == "").sum()
    )

    if missing_transcripts:
        print(f"WARNING: train/dev rows with missing transcription: {missing_transcripts:,}")

    duplicate_ids = df["id"].duplicated().sum()
    if duplicate_ids:
        print(f"WARNING: duplicate IDs detected: {duplicate_ids:,}")


def write_outputs(df: pd.DataFrame) -> None:
    validate_manifest(df)

    for split, filename in [
        ("train", "manifest_train.parquet"),
        ("dev", "manifest_dev.parquet"),
        ("test", "manifest_test.parquet"),
    ]:
        split_df = df[df["split"] == split].copy()
        output_path = OUTPUT_DIR / filename
        split_df.to_parquet(output_path, index=False)
        print(f"Wrote {output_path}: {len(split_df):,} rows")

    summary = (
        df.groupby(["language", "language_name", "split", "speech_type"], dropna=False)
        .agg(
            rows=("id", "count"),
            hours=("duration_seconds", lambda x: x.dropna().sum() / 3600),
        )
        .reset_index()
        .sort_values(["language", "split", "speech_type"])
    )
    summary.to_csv(REPORT_DIR / "manifest_summary.csv", index=False)
    print(f"Wrote {REPORT_DIR / 'manifest_summary.csv'}")


def main() -> None:
    frames = []

    print("Loading Swahili manifest references...")
    frames.append(load_swahili_source())

    for iso, info in ANV_SOURCES.items():
        print(f"Loading {iso} manifest references...")
        frames.append(load_anv_source(iso, info))

    manifest = pd.concat(frames, ignore_index=True)
    write_outputs(manifest)

    print("Exp036B unified manifest generation complete.")


if __name__ == "__main__":
    main()
