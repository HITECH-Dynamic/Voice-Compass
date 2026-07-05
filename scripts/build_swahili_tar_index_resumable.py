"""
Exp057B — Resumable Swahili TAR Index Builder

Builds a persistent audio_filename -> TAR archive/member lookup table for Swahili.

Design:
- Processes Parquet shards incrementally.
- Saves progress after every shard.
- Skips shards already indexed.
- Can resume after Colab disconnects.

Outputs:
- data/afrivoices/indexes/swahili_tar_index.parquet
- data/afrivoices/indexes/swahili_tar_index_progress.csv
- reports/dataset_analysis/swahili_tar_index_summary.csv
"""

from pathlib import Path
import argparse
import time
import pandas as pd
import tarfile
from huggingface_hub import hf_hub_download, list_repo_files

INDEX_DIR = Path("data/afrivoices/indexes")
REPORT_DIR = Path("reports/dataset_analysis")

INDEX_PATH = INDEX_DIR / "swahili_tar_index.parquet"
PROGRESS_PATH = INDEX_DIR / "swahili_tar_index_progress.csv"
SUMMARY_PATH = REPORT_DIR / "swahili_tar_index_summary.csv"

INDEX_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SWAHILI_REPO = "DigitalUmuganda/Afrivoice_Swahili"

SWAHILI_SOURCES = {
    "agriculture": "agriculture_swahili_train",
    "education": "education_swahili_train",
    "financial": "financial_swahili_train",
    "government": "government_swahili_train",
    "health": "health_swahili_train",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=list(SWAHILI_SOURCES.keys()))
    parser.add_argument("--start-archive-index", type=int, default=0)
    parser.add_argument("--max-archives", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_existing_index():
    if INDEX_PATH.exists() and not INDEX_PATH.stat().st_size == 0:
        return pd.read_parquet(INDEX_PATH)

    return pd.DataFrame(
        columns=[
            "domain",
            "repo",
            "raw_split",
            "archive_file",
            "member_name",
            "audio_filename",
        ]
    )


def load_progress():
    if PROGRESS_PATH.exists() and not PROGRESS_PATH.stat().st_size == 0:
        return pd.read_csv(PROGRESS_PATH)

    return pd.DataFrame(
        columns=[
            "domain",
            "repo",
            "raw_split",
            "archive_file",
            "status",
            "rows",
        ]
    )


def save_outputs(index, progress):
    index = index.drop_duplicates(
        subset=["domain", "repo", "archive_file", "member_name", "audio_filename"]
    ).reset_index(drop=True)

    index.to_parquet(INDEX_PATH, index=False)
    progress.to_csv(PROGRESS_PATH, index=False)

    if len(index):
        summary = (
            index.groupby(["domain", "raw_split"], dropna=False)
            .agg(
                files=("audio_filename", "count"),
                archives=("archive_file", "nunique"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame(columns=["domain", "raw_split", "files", "archives"])

    summary.to_csv(SUMMARY_PATH, index=False)


def parse_archive_path(archive_file):
    parts = archive_file.split("/")
    raw_split = parts[0] if len(parts) > 0 else "unknown"
    return raw_split


def get_candidate_archives(raw_split):
    files = list_repo_files(repo_id=SWAHILI_REPO, repo_type="dataset")

    return sorted(
        f for f in files
        if f.startswith(f"{raw_split}/audio/")
        and (
            f.endswith(".tar")
            or f.endswith(".tar.gz")
            or f.endswith(".tar.xz")
        )
    )


def read_members_from_archive(archive_file, max_retries=3, retry_sleep=30):
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Download attempt {attempt}/{max_retries}: {archive_file}")

            local_path = hf_hub_download(
                repo_id=SWAHILI_REPO,
                repo_type="dataset",
                filename=archive_file,
            )

            rows = []
            with tarfile.open(local_path, "r:*") as tar:
                for member in tar:
                    if member.isfile():
                        rows.append(
                            {
                                "member_name": member.name,
                                "audio_filename": Path(member.name).name,
                            }
                        )

            return pd.DataFrame(rows)

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            last_error = exc
            print(f"WARNING: attempt {attempt} failed for {archive_file}: {exc}")

            if attempt < max_retries:
                print(f"Sleeping {retry_sleep}s before retry...")
                time.sleep(retry_sleep)

    raise RuntimeError(f"Failed after {max_retries} attempts: {archive_file}") from last_error


def main():
    args = parse_args()

    index = load_existing_index()
    progress = load_progress()

    completed = set()
    if not args.force and len(progress):
        completed = set(
            progress.loc[progress["status"] == "completed", "archive_file"].astype(str)
        )

    total_processed_this_run = 0

    for domain in args.domains:
        if domain not in SWAHILI_SOURCES:
            raise ValueError(f"Unknown Swahili domain: {domain}")

        raw_split = SWAHILI_SOURCES[domain]
        archives = get_candidate_archives(raw_split)

        if args.start_archive_index:
            archives = archives[args.start_archive_index:]

        if args.max_archives is not None:
            archives = archives[: args.max_archives]

        print("\n" + "=" * 80)
        print(f"{domain} — {raw_split}")
        print(f"Candidate archives: {len(archives)}")

        for archive_file in archives:
            archive_key = f"{SWAHILI_REPO}::{archive_file}"

            if archive_key in completed:
                print(f"SKIP completed: {archive_file}")
                continue

            print(f"INDEXING: {archive_file}")

            try:
                members = read_members_from_archive(
                    archive_file,
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                )

                rows = members.assign(
                    domain=domain,
                    repo=SWAHILI_REPO,
                    raw_split=raw_split,
                    archive_file=archive_file,
                )

                rows = rows[
                    [
                        "domain",
                        "repo",
                        "raw_split",
                        "archive_file",
                        "member_name",
                        "audio_filename",
                    ]
                ]

                index = pd.concat([index, rows], ignore_index=True)

                progress_row = pd.DataFrame(
                    [{
                        "domain": domain,
                        "repo": SWAHILI_REPO,
                        "raw_split": raw_split,
                        "archive_file": archive_key,
                        "status": "completed",
                        "rows": len(rows),
                    }]
                )
                progress = pd.concat([progress, progress_row], ignore_index=True)

                save_outputs(index, progress)

                total_processed_this_run += 1

                print(f"Saved progress after archive. Rows added: {len(rows):,}")

            except Exception as exc:
                print(f"ERROR: failed archive: {archive_file}")
                print(f"ERROR: {exc}")

                progress_row = pd.DataFrame(
                    [{
                        "domain": domain,
                        "repo": SWAHILI_REPO,
                        "raw_split": raw_split,
                        "archive_file": archive_key,
                        "status": "failed",
                        "rows": 0,
                    }]
                )
                progress = pd.concat([progress, progress_row], ignore_index=True)
                save_outputs(index, progress)
                print("Saved failed archive status and continuing.")
                continue

    save_outputs(index, progress)

    completed_count = progress[progress["status"] == "completed"]["archive_file"].nunique()

    print("\n" + "=" * 80)
    print("Exp057B resumable Swahili TAR index complete.")
    print(f"Processed archives this run: {total_processed_this_run}")
    print(f"Total index rows: {len(index):,}")
    print(f"Total indexed archives: {completed_count:,}")
    print(f"Wrote: {INDEX_PATH}")
    print(f"Wrote: {PROGRESS_PATH}")
    print(f"Wrote: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
