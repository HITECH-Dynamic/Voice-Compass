"""
Exp038B — Whisper Dataset Test

Tests that AfriVoices examples can be converted into Whisper-ready
input_features and labels.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.afrivoices_whisper_dataset import AfriVoicesWhisperDataset


MANIFEST_PATH = "data/afrivoices/manifests/manifest_train.parquet"


def main():
    dataset = AfriVoicesWhisperDataset(
        manifest_path=MANIFEST_PATH,
        processor_name="openai/whisper-large-v3",
        languages=["kik", "swa"],
        max_duration=30.0,
        max_rows_per_language=1,
    )

    print(f"Dataset rows after filtering: {len(dataset):,}")

    for i in range(len(dataset)):
        sample = dataset[i]

        print("\n" + "=" * 80)
        print(f"Language: {sample['language']}")
        print(f"unique_id: {sample['unique_id']}")
        print(f"Duration: {sample['duration_seconds']:.2f}s")
        print(f"Resolved source: {sample['resolved_source']}")
        print(f"Input features shape: {tuple(sample['input_features'].shape)}")
        print(f"Labels shape: {tuple(sample['labels'].shape)}")
        print(f"Transcript: {sample['transcription'][:160]}")

    print("\nExp038B Whisper dataset test complete.")


if __name__ == "__main__":
    main()
