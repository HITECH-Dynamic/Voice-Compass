"""
Exp041A — ANV Index Strategy Inspection

Purpose:
Inspect whether ANV audio filename-to-shard mapping can be derived without
downloading every large audio Parquet shard.

This script lists repo structure and compares:
- transcript/meta CSV filename references
- available audio parquet shard names
- whether any lightweight metadata gives shard-level mapping
"""

from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

REPORT_DIR = Path("reports/dataset_analysis")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ANV_SOURCES = {
    "kik": ("Kikuyu", "Anv-ke/kikuyu"),
    "luo": ("Luo / Dholuo", "Anv-ke/Dholuo"),
    "kln": ("Kalenjin", "Anv-ke/Kalenjin"),
    "mas": ("Maasai", "Anv-ke/Maasai"),
    "som": ("Somali", "Anv-ke/Somali"),
}


def read_csv_safe(path):
    for encoding in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=encoding, engine="python", on_bad_lines="warn")
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1", encoding_errors="replace", engine="python", on_bad_lines="warn")


def inspect_repo(iso, name, repo):
    print("\n" + "=" * 80)
    print(f"{iso} — {name}")
    print(repo)

    files = list_repo_files(repo_id=repo, repo_type="dataset")

    transcript_files = sorted(f for f in files if f.endswith("/files/transcripts.csv"))
    meta_files = sorted(f for f in files if f.endswith("/files/meta.csv"))
    parquet_files = sorted(f for f in files if "/audios/" in f and f.endswith(".parquet"))

    rows = []

    print(f"Transcript CSVs: {len(transcript_files)}")
    print(f"Meta CSVs: {len(meta_files)}")
    print(f"Audio parquet shards: {len(parquet_files)}")

    for tx_file in transcript_files:
        parts = tx_file.split("/")
        split = parts[0] if len(parts) > 0 else "unknown"
        speech_type = parts[1] if len(parts) > 1 else "unknown"

        tx_path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=tx_file)
        tx = read_csv_safe(tx_path)

        possible_audio_cols = [
            c for c in tx.columns
            if "media" in c.lower()
            or "file" in c.lower()
            or "audio" in c.lower()
            or "path" in c.lower()
            or "filename" in c.lower()
        ]

        matching_parquets = [
            p for p in parquet_files
            if p.startswith(f"{split}/{speech_type}/audios/")
        ]

        rows.append({
            "language": iso,
            "language_name": name,
            "repo": repo,
            "split": split,
            "speech_type": speech_type,
            "transcript_file": tx_file,
            "transcript_rows": len(tx),
            "transcript_columns": "|".join(tx.columns),
            "possible_audio_columns": "|".join(possible_audio_cols),
            "matching_parquet_shards": len(matching_parquets),
            "example_parquet_shard": matching_parquets[0] if matching_parquets else "",
        })

        print(f"\n{tx_file}")
        print(f"  rows: {len(tx):,}")
        print(f"  audio-like columns: {possible_audio_cols}")
        print(f"  matching parquet shards: {len(matching_parquets)}")
        if possible_audio_cols:
            col = possible_audio_cols[0]
            print(f"  sample {col}: {tx[col].dropna().astype(str).head(3).tolist()}")

    return rows


def main():
    all_rows = []

    for iso, (name, repo) in ANV_SOURCES.items():
        all_rows.extend(inspect_repo(iso, name, repo))

    report = pd.DataFrame(all_rows)
    out = REPORT_DIR / "exp041a_anv_index_strategy_inspection.csv"
    report.to_csv(out, index=False)

    print("\n" + "=" * 80)
    print(f"Wrote {out}")
    print("Exp041A inspection complete.")


if __name__ == "__main__":
    main()
