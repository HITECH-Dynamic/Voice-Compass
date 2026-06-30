"""
Exp041B — Resumable Global ANV Audio Index Builder

Builds a persistent filename -> Parquet shard lookup table for ANV datasets.

Design:
- Processes Parquet shards incrementally.
- Saves progress after every shard.
- Skips shards already indexed.
- Can resume after Colab disconnects.

Outputs:
- data/afrivoices/indexes/anv_audio_index.parquet
- data/afrivoices/indexes/anv_audio_index_progress.csv
- reports/dataset_analysis/anv_audio_index_summary.csv
"""

from pathlib import Path
import argparse
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

INDEX_DIR = Path("data/afrivoices/indexes")
REPORT_DIR = Path("reports/dataset_analysis")

INDEX_PATH = INDEX_DIR / "anv_audio_index.parquet"
PROGRESS_PATH = INDEX_DIR / "anv_audio_index_progress.csv"
SUMMARY_PATH = REPORT_DIR / "anv_audio_index_summary.csv"

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
    parser.add_argument("--languages", nargs="+", default=list(ANV_SOURCES.keys()))
    parser.add_argument("--split", default=None)
    parser.add_argument("--speech-type", default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_existing_index():
    if INDEX_PATH.exists() and not INDEX_PATH.stat().st_size == 0:
        return pd.read_parquet(INDEX_PATH)

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


def load_progress():
    if PROGRESS_PATH.exists() and not PROGRESS_PATH.stat().st_size == 0:
        return pd.read_csv(PROGRESS_PATH)

    return pd.DataFrame(
        columns=[
            "language",
            "repo",
            "split",
            "speech_type",
            "parquet_file",
            "status",
            "rows",
        ]
    )


def save_outputs(index, progress):
    index = index.drop_duplicates(
        subset=["language", "repo", "parquet_file", "audio_filename"]
    ).reset_index(drop=True)

    index.to_parquet(INDEX_PATH, index=False)
    progress.to_csv(PROGRESS_PATH, index=False)

    if len(index):
        summary = (
            index.groupby(["language", "split", "speech_type"], dropna=False)
            .agg(
                files=("audio_filename", "count"),
                shards=("parquet_file", "nunique"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame(columns=["language", "split", "speech_type", "files", "shards"])

    summary.to_csv(SUMMARY_PATH, index=False)


def parse_shard_path(parquet_file):
    parts = parquet_file.split("/")
    split = parts[0] if len(parts) > 0 else "unknown"
    speech_type = parts[1] if len(parts) > 1 else "unknown"
    return split, speech_type


def get_candidate_shards(repo, split_filter=None, speech_type_filter=None):
    files = list_repo_files(repo_id=repo, repo_type="dataset")

    shards = sorted(
        f for f in files
        if "/audios/" in f and f.endswith(".parquet")
    )

    if split_filter:
        shards = [s for s in shards if s.split("/")[0] == split_filter]

    if speech_type_filter:
        shards = [s for s in shards if len(s.split("/")) > 1 and s.split("/")[1] == speech_type_filter]

    return shards


def read_filenames_from_shard(repo, parquet_file):
    local_path = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=parquet_file,
    )

    df = pd.read_parquet(local_path, columns=["filename"])

    return df["filename"].astype(str)


def main():
    args = parse_args()

    index = load_existing_index()
    progress = load_progress()

    completed = set()
    if not args.force and len(progress):
        completed = set(
            progress.loc[progress["status"] == "completed", "parquet_file"].astype(str)
        )

    total_processed_this_run = 0

    for iso in args.languages:
        if iso not in ANV_SOURCES:
            raise ValueError(f"Unknown language: {iso}")

        _, repo = ANV_SOURCES[iso]
        shards = get_candidate_shards(repo, args.split, args.speech_type)

        if args.max_shards is not None:
            shards = shards[: args.max_shards]

        print("\n" + "=" * 80)
        print(f"{iso} — {repo}")
        print(f"Candidate shards: {len(shards)}")

        for parquet_file in shards:
            shard_key = f"{repo}::{parquet_file}"

            if shard_key in completed:
                print(f"SKIP completed: {parquet_file}")
                continue

            print(f"INDEXING: {parquet_file}")

            split, speech_type = parse_shard_path(parquet_file)
            filenames = read_filenames_from_shard(repo, parquet_file)

            rows = pd.DataFrame(
                {
                    "language": iso,
                    "repo": repo,
                    "split": split,
                    "speech_type": speech_type,
                    "parquet_file": parquet_file,
                    "audio_filename": filenames,
                }
            )

            index = pd.concat([index, rows], ignore_index=True)

            progress_row = pd.DataFrame(
                [{
                    "language": iso,
                    "repo": repo,
                    "split": split,
                    "speech_type": speech_type,
                    "parquet_file": shard_key,
                    "status": "completed",
                    "rows": len(rows),
                }]
            )
            progress = pd.concat([progress, progress_row], ignore_index=True)

            save_outputs(index, progress)

            total_processed_this_run += 1

            print(f"Saved progress after shard. Rows added: {len(rows):,}")

    save_outputs(index, progress)

    print("\n" + "=" * 80)
    print("Exp041B resumable ANV audio index complete.")
    print(f"Processed shards this run: {total_processed_this_run}")
    print(f"Total index rows: {len(index):,}")
    print(f"Total indexed shards: {progress[progress['status'] == 'completed']['parquet_file'].nunique():,}")
    print(f"Wrote: {INDEX_PATH}")
    print(f"Wrote: {PROGRESS_PATH}")
    print(f"Wrote: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
