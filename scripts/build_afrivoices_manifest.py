"""
Exp036B — AfriVoices Unified Manifest Builder

Purpose:
Create normalized metadata manifests for the six official AfriVoices
competition languages from seven Hugging Face source repositories.

Outputs:
- data/afrivoices/manifests/manifest_train.parquet
- data/afrivoices/manifests/manifest_dev.parquet
- data/afrivoices/manifests/manifest_test.parquet
- reports/dataset_analysis/manifest_summary.csv

Note:
This builder creates metadata/audio-reference manifests. It does not extract
all audio bytes yet. Audio resolution happens in the dataset loader.
"""

from pathlib import Path
import json
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


def read_csv_safe(path):
    for encoding in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=encoding, engine="python", on_bad_lines="warn")
        except UnicodeDecodeError:
            continue
    return pd.read_csv(
        path,
        encoding="latin1",
        encoding_errors="replace",
        engine="python",
        on_bad_lines="warn",
    )


def normalize_split(split):
    if split == "dev":
        return "dev"
    if split == "dev_test":
        return "test"
    if split == "test":
        return "test"
    if split == "train":
        return "train"
    return split


def safe_series(df, column, default=""):
    """Return a column as a Series, or a default Series when missing."""
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def load_anv_source(iso, info):
    repo = info["repo"]
    files = list_repo_files(repo_id=repo, repo_type="dataset")

    transcript_files = sorted(f for f in files if f.endswith("/files/transcripts.csv"))
    audio_parquet_files = sorted(f for f in files if "/audios/" in f and f.endswith(".parquet"))
    audio_map = {}

    for audio_file in audio_parquet_files:
        parts = audio_file.split("/")
        if len(parts) >= 4:
            key = (parts[0], parts[1])
            audio_map.setdefault(key, []).append(audio_file)

    frames = []

    for tx_file in transcript_files:
        parts = tx_file.split("/")
        raw_split = parts[0]
        speech_type = parts[1] if len(parts) > 1 else "unknown"
        split = normalize_split(raw_split)

        tx_path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=tx_file)
        tx = read_csv_safe(tx_path)

        meta_file = tx_file.replace("transcripts.csv", "meta.csv")
        try:
            meta_path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=meta_file)
            meta = read_csv_safe(meta_path)
            if "recorder_uuid" in tx.columns and "recorder_uuid" in meta.columns:
                tx = tx.merge(meta, on="recorder_uuid", how="left", suffixes=("", "_speaker"))
        except Exception:
            pass

        media_path = safe_series(tx, "mediaPathId", "")
tx["id"] = media_path.astype(str).str.replace("VOICE_COLLECTION/", "", regex=False).str.replace(".wav", "", regex=False)
        tx["language"] = iso
        tx["language_name"] = info["name"]
        tx["source_repo"] = repo
        tx["source_name"] = info["repo"]
        tx["split"] = split
        tx["raw_split"] = raw_split
        tx["speech_type"] = speech_type
        tx["audio_source_type"] = "parquet_bytes"
        tx["audio_ref"] = safe_series(tx, "mediaPathId", "")
        tx["audio_filename"] = safe_series(tx, "mediaPathId", "").astype(str).str.split("/").str[-1]
        tx["audio_parquet_candidates"] = json.dumps(audio_map.get((raw_split, speech_type), []))
        tx["transcription"] = safe_series(tx, "actualSentence", safe_series(tx, "transcription", ""))
        tx["translation"] = safe_series(tx, "translatedText", "")
        tx["duration_seconds"] = pd.to_numeric(safe_series(tx, "duration", None), errors="coerce")
        tx["speaker_id"] = safe_series(tx, "recorder_uuid", "")
        tx["dialect"] = safe_series(tx, "sentenceDialect", safe_series(tx, "dialect", ""))
        tx["domain"] = safe_series(tx, "domain", safe_series(tx, "topic", ""))
        tx["gender"] = safe_series(tx, "gender", "")
        tx["age"] = safe_series(tx, "ownerAge", "")
        tx["county"] = safe_series(tx, "countyName", "")
        tx["metadata"] = "{}"

        keep = [
            "id", "language", "language_name", "source_repo", "source_name",
            "split", "raw_split", "speech_type", "audio_source_type", "audio_ref",
            "audio_filename", "audio_parquet_candidates", "transcription",
            "translation", "duration_seconds", "speaker_id", "dialect",
            "domain", "gender", "age", "county", "metadata",
        ]

        frames.append(tx[keep])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_swahili_source():
    files = list_repo_files(repo_id=SWAHILI_REPO, repo_type="dataset")
    manifest_files = sorted(f for f in files if f.endswith(".jsonl") and "manifest" in f)
    rows = []

    for mf in manifest_files:
        folder = mf.split("/")[0]
        if "_train" in folder:
            split = "train"
        elif "_dev" in folder:
            split = "dev"
        elif "_test" in folder:
            split = "test"
        else:
            split = "unknown"

        path = hf_hub_download(repo_id=SWAHILI_REPO, repo_type="dataset", filename=mf)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)

                transcription = (
                    item.get("normalized_transcription")
                    or item.get("transcription")
                    or ""
                )

                audio_filepath = item.get("audio_filepath", "")
                key = item.get("key", Path(audio_filepath).stem)

                rows.append({
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
                    "duration_seconds": pd.to_numeric(item.get("duration", None), errors="coerce"),
                    "speaker_id": item.get("voice_creator_id", ""),
                    "dialect": item.get("locale", ""),
                    "domain": item.get("category", ""),
                    "gender": item.get("gender", ""),
                    "age": item.get("age_group", ""),
                    "county": item.get("location", ""),
                    "metadata": json.dumps({
                        "manifest_file": mf,
                        "dir_path": item.get("dir_path", ""),
                        "chunk_id": item.get("chunk_id", ""),
                        "project_name": item.get("project_name", ""),
                    }, ensure_ascii=False),
                })

    return pd.DataFrame(rows)


def validate_manifest(df):
    required = ["id", "language", "split", "audio_ref", "transcription"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    train_dev = df[df["split"].isin(["train", "dev"])]
    missing_transcripts = train_dev["transcription"].isna().sum() + (train_dev["transcription"].astype(str).str.strip() == "").sum()
    if missing_transcripts:
        print(f"WARNING: train/dev rows with missing transcription: {missing_transcripts}")

    bad_langs = set(df["language"].dropna().unique()) - {"swa", "kik", "luo", "som", "mas", "kln"}
    if bad_langs:
        raise ValueError(f"Unexpected language codes: {bad_langs}")


def write_outputs(df):
    validate_manifest(df)

    for split, out_name in [
        ("train", "manifest_train.parquet"),
        ("dev", "manifest_dev.parquet"),
        ("test", "manifest_test.parquet"),
    ]:
        part = df[df["split"] == split].copy()
        part.to_parquet(OUTPUT_DIR / out_name, index=False)
        print(f"Wrote {OUTPUT_DIR / out_name}: {len(part):,} rows")

    summary = (
        df.groupby(["language", "language_name", "split", "speech_type"], dropna=False)
        .agg(
            rows=("id", "count"),
            hours=("duration_seconds", lambda x: x.dropna().sum() / 3600),
        )
        .reset_index()
    )
    summary.to_csv(REPORT_DIR / "manifest_summary.csv", index=False)
    print(f"Wrote {REPORT_DIR / 'manifest_summary.csv'}")


def main():
    frames = []

    print("Loading Swahili manifest references...")
    frames.append(load_swahili_source())

    for iso, info in ANV_SOURCES.items():
        print(f"Loading {iso} manifest references...")
        frames.append(load_anv_source(iso, info))

    df = pd.concat(frames, ignore_index=True)
    write_outputs(df)

    print("Exp036B unified manifest generation complete.")


if __name__ == "__main__":
    main()
