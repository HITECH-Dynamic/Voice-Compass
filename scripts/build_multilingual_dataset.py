"""
Exp042 — Multilingual Dataset Builder

Builds train/eval parquet datasets from AfriVoices manifests using a YAML config.
"""

from pathlib import Path
import argparse
import yaml
import pandas as pd


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manifests(cfg):
    frames = []
    inputs = cfg["inputs"]

    for key in ["train_manifest", "dev_manifest", "test_manifest"]:
        path = Path(inputs[key])
        if path.exists():
            df = pd.read_parquet(path)
            df["manifest_source"] = key.replace("_manifest", "")
            frames.append(df)

    if not frames:
        raise FileNotFoundError("No manifest parquet files found.")

    return pd.concat(frames, ignore_index=True)


def filter_dataset(df, cfg):
    f = cfg["filters"]

    if f.get("languages"):
        df = df[df["language"].isin(f["languages"])]

    if f.get("source_splits"):
        df = df[df["split"].isin(f["source_splits"])]

    if f.get("speech_types"):
        df = df[df["speech_type"].isin(f["speech_types"])]

    if f.get("require_transcription", True):
        df = df[df["transcription"].notna()]
        df = df[df["transcription"].astype(str).str.strip() != ""]

    min_dur = f.get("min_duration_seconds")
    max_dur = f.get("max_duration_seconds")

    if min_dur is not None:
        df = df[df["duration_seconds"].fillna(-1) >= min_dur]

    if max_dur is not None:
        df = df[df["duration_seconds"].fillna(10**9) <= max_dur]

    return df.copy()


def attach_index_status(df, cfg):
    index_path = Path(cfg["inputs"].get("anv_audio_index", ""))

    df["audio_index_status"] = "not_required"
    df["indexed_parquet_file"] = ""

    if not index_path.exists():
        df.loc[df["audio_source_type"].eq("parquet_bytes"), "audio_index_status"] = "index_missing"
        return df

    index = pd.read_parquet(index_path)
    index = index[["language", "audio_filename", "parquet_file"]].drop_duplicates()

    merged = df.merge(
        index,
        on=["language", "audio_filename"],
        how="left",
        suffixes=("", "_indexed"),
    )

    is_anv = merged["audio_source_type"].eq("parquet_bytes")
    has_index = merged["parquet_file"].notna()

    merged.loc[is_anv & has_index, "audio_index_status"] = "indexed"
    merged.loc[is_anv & ~has_index, "audio_index_status"] = "fallback_required"
    merged["indexed_parquet_file"] = merged["parquet_file"].fillna("")
    merged = merged.drop(columns=["parquet_file"])

    return merged


def sample_and_split(df, cfg):
    s = cfg["sampling"]
    max_rows = s.get("max_rows_per_language")
    eval_fraction = float(s.get("eval_fraction", 0.2))
    seed = int(s.get("random_seed", 42))

    sampled = []

    for lang, part in df.groupby("language"):
        part = part.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if max_rows:
            part = part.head(int(max_rows))
        sampled.append(part)

    df = pd.concat(sampled, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    eval_parts = []
    train_parts = []

    for lang, part in df.groupby("language"):
        n_eval = max(1, int(len(part) * eval_fraction)) if len(part) > 1 else 0
        eval_parts.append(part.head(n_eval))
        train_parts.append(part.iloc[n_eval:])

    train = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    eval_df = pd.concat(eval_parts, ignore_index=True) if eval_parts else pd.DataFrame()

    return train, eval_df


def write_summary(train, eval_df, cfg):
    rows = []

    for name, df in [("train", train), ("eval", eval_df)]:
        if df.empty:
            continue

        summary = (
            df.groupby(["language", "speech_type", "audio_index_status"], dropna=False)
            .agg(
                rows=("unique_id", "count"),
                hours=("duration_seconds", lambda x: x.dropna().sum() / 3600),
            )
            .reset_index()
        )
        summary["dataset_split"] = name
        rows.append(summary)

    summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out = Path(cfg["outputs"]["summary"])
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp042_multilingual_dataset.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    df = load_manifests(cfg)
    df = filter_dataset(df, cfg)
    df = attach_index_status(df, cfg)

    train, eval_df = sample_and_split(df, cfg)

    train_path = Path(cfg["outputs"]["train"])
    eval_path = Path(cfg["outputs"]["eval"])

    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    train.to_parquet(train_path, index=False)
    eval_df.to_parquet(eval_path, index=False)

    summary = write_summary(train, eval_df, cfg)

    print(f"Wrote {train_path}: {len(train):,} rows")
    print(f"Wrote {eval_path}: {len(eval_df):,} rows")
    print(f"Wrote {cfg['outputs']['summary']}")
    print()
    print(summary)
    print()
    print("Exp042 dataset builder complete.")


if __name__ == "__main__":
    main()
