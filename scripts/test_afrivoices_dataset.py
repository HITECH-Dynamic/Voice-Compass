"""
Exp038A — Dataset Class Test

Tests one decoded example per language from the unified manifest.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.afrivoices_dataset import AfriVoicesDataset


MANIFEST_PATH = "data/afrivoices/manifests/manifest_train.parquet"


def main():
    dataset = AfriVoicesDataset(
        manifest_path=MANIFEST_PATH,
        max_duration=30.0,
        languages=["kik", "swa"],
        max_rows_per_language=1,
    )

    print(f"Dataset rows after filtering: {len(dataset):,}")
    print("Languages:", sorted(dataset.df["language"].unique()))

    for lang in sorted(dataset.df["language"].unique()):
        idx = dataset.df[dataset.df["language"] == lang].index[0]
        sample = dataset[idx]

        print("\n" + "=" * 80)
        print(f"Language: {sample['language']}")
        print(f"unique_id: {sample['unique_id']}")
        print(f"Sample rate: {sample['sample_rate']}")
        print(f"Audio samples: {len(sample['audio'])}")
        print(f"Duration: {sample['duration_seconds']:.2f}s")
        print(f"Resolved source: {sample['resolved_source']}")
        print(f"Transcript: {sample['transcription'][:160]}")

    print("\nExp038A dataset test complete.")


if __name__ == "__main__":
    main()
