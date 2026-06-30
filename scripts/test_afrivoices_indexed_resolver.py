"""
Exp040B — Indexed ANV Resolver Test

Tests that AfriVoicesDataset can use the bounded ANV audio index to resolve
a Kikuyu example directly from an indexed Parquet shard.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.afrivoices_dataset import AfriVoicesDataset


MANIFEST_PATH = "data/afrivoices/manifests/manifest_train.parquet"
INDEX_PATH = "data/afrivoices/indexes/anv_audio_index_smoke.parquet"


def main():
    dataset = AfriVoicesDataset(
        manifest_path=MANIFEST_PATH,
        languages=["kik"],
        max_duration=30.0,
        max_rows_per_language=50,
        anv_index_path=INDEX_PATH,
    )

    indexed_files = set(dataset.anv_index["audio_filename"].astype(str))
    subset = dataset.df[
        dataset.df["audio_filename"].astype(str).isin(indexed_files)
    ].reset_index(drop=True)

    if subset.empty:
        raise RuntimeError("No manifest rows found that match the smoke ANV index.")

    dataset.df = subset.head(1).copy()

    print(f"Dataset rows after indexed filter: {len(dataset)}")
    print(f"ANV index rows: {len(dataset.anv_index)}")

    sample = dataset[0]

    print("\n" + "=" * 80)
    print(f"Language: {sample['language']}")
    print(f"unique_id: {sample['unique_id']}")
    print(f"Sample rate: {sample['sample_rate']}")
    print(f"Audio samples: {len(sample['audio'])}")
    print(f"Duration: {sample['duration_seconds']:.2f}s")
    print(f"Resolved source: {sample['resolved_source']}")
    print(f"Transcript: {sample['transcription'][:160]}")

    print("\nExp040B indexed resolver test complete.")


if __name__ == "__main__":
    main()
