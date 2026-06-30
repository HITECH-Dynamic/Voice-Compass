"""
Exp040A — ANV Audio Index Builder

Builds a filename -> parquet shard lookup table for all ANV datasets.

Outputs

data/afrivoices/indexes/anv_audio_index.parquet
reports/dataset_analysis/anv_audio_index_summary.csv
"""

from pathlib import Path
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


def build_language_index(iso, repo):
    print(f"\nLoading {repo}")

    files = list_repo_files(repo_id=repo, repo_type="dataset")

    parquet_files = sorted(
        f for f in files
        if "/audios/" in f and f.endswith(".parquet")
    )

    rows = []

    for parquet_file in parquet_files:

        print(" ", parquet_file)

        local_path = hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=parquet_file,
        )

        df = pd.read_parquet(
            local_path,
            columns=["filename"],
        )

        tmp = pd.DataFrame({
            "language": iso,
            "repo": repo,
            "parquet_file": parquet_file,
            "audio_filename": df["filename"].astype(str),
        })

        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)


def main():

    all_rows = []

    for iso, (_, repo) in ANV_SOURCES.items():
        all_rows.append(build_language_index(iso, repo))

    index = pd.concat(all_rows, ignore_index=True)

    index.to_parquet(
        INDEX_DIR / "anv_audio_index.parquet",
        index=False,
    )

    summary = (
        index.groupby("language")
        .agg(
            files=("audio_filename", "count"),
            shards=("parquet_file", "nunique"),
        )
        .reset_index()
    )

    summary.to_csv(
        REPORT_DIR / "anv_audio_index_summary.csv",
        index=False,
    )

    print("\n========================================")
    print(summary)
    print("========================================")
    print()
    print("Index rows:", len(index))
    print("Unique filenames:", index["audio_filename"].nunique())
    print()
    print("Exp040A complete.")


if __name__ == "__main__":
    main()
