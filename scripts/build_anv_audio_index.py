"""
Exp040A — Bounded ANV Audio Index Builder

Purpose:
Build a small filename -> parquet shard lookup index first, before attempting
full-corpus indexing.

Default scope:
- language: Kikuyu only
- split/type: train/scripted only
- max shards: 2

Outputs:
- data/afrivoices/indexes/anv_audio_index_smoke.parquet
- reports/dataset_analysis/anv_audio_index_smoke_summary.csv
"""

from pathlib import Path
import argparse
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

INDEX_DIR = Path("data/afrivoices/indexes")
REPORT_DIR = Path("reports/dataset_analysis")

INDEX_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ANV_SOURCES = {
    "kik": ("Kikuyu", "Anv-ke/kikuyu"),
    "luo": ("Luo / Dholuo", "Anv-ke/Dholuo"),
    "kln": ("Kalenjin", "Anv-ke/Kalenjin"),
    "mas": ("Maasai", "Anv-ke/Maasai"),
    "som": ("Somali", "Anv-ke/Somali"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=["kik"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--speech-type", default="scripted")
    parser.add_argument("--max-shards", type=int, default=2)
    parser.add_argument("--output-name", default="anv_audio_index_smoke")
    return parser.parse_args()


def build_language_index(iso, repo, split, speech_type, max_shards):
    print(f"\nLoading {repo}")

    files = list_repo_files(repo_id=repo, repo_type="dataset")

    prefix = f"{split}/{speech_type}/audios/"
    parquet_files = sorted(
        f for f in files
        if f.startswith(prefix) and f.endswith(".parquet")
    )

    parquet_files = parquet_files[:max_shards]

    print(f"Selected {len(parquet_files)} shards:")
    for pf in parquet_files:
        print(f"  - {pf}")

    rows = []

    for parquet_file in parquet_files:
        local_path = hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=parquet_file,
        )

        df = pd.read_parquet(local_path, columns=["filename"])

        tmp = pd.DataFrame(
            {
                "language": iso,
                "repo": repo,
                "split": split,
                "speech_type": speech_type,
                "parquet_file": parquet_file,
                "audio_filename": df["filename"].astype(str),
            }
        )

        rows.append(tmp)

    if not rows:
        return pd.DataFrame(
            columns=[
                "language",
                "repo",
                "split",
                "speech_type",
                "parquet_file",
                "audio_filename",
            ]
        )

    return pd.concat(rows, ignore_index=True)


def main():
    args = parse_args()

    all_rows = []

    for iso in args.languages:
        if iso not in ANV_SOURCES:
            raise ValueError(f"Unknown language: {iso}")

        _, repo = ANV_SOURCES[iso]
        all_rows.append(
            build_language_index(
                iso=iso,
                repo=repo,
                split=args.split,
                speech_type=args.speech_type,
                max_shards=args.max_shards,
            )
        )

    index = pd.concat(all_rows, ignore_index=True)

    index_path = INDEX_DIR / f"{args.output_name}.parquet"
    summary_path = REPORT_DIR / f"{args.output_name}_summary.csv"

    index.to_parquet(index_path, index=False)

    summary = (
        index.groupby(["language", "split", "speech_type"], dropna=False)
        .agg(
            files=("audio_filename", "count"),
            shards=("parquet_file", "nunique"),
        )
        .reset_index()
    )

    summary.to_csv(summary_path, index=False)

    print("\n========================================")
    print(summary)
    print("========================================")
    print(f"Index rows: {len(index):,}")
    print(f"Unique filenames: {index['audio_filename'].nunique():,}")
    print(f"Wrote: {index_path}")
    print(f"Wrote: {summary_path}")
    print("Exp040A smoke index complete.")


if __name__ == "__main__":
    main()
