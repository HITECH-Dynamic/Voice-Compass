"""
Exp038B — Whisper Processor Dataset Wrapper

Wraps AfriVoicesDataset and converts decoded audio/transcripts into
Whisper-ready input_features and label IDs.

No model training here.
"""

from transformers import WhisperProcessor

from src.datasets.afrivoices_dataset import AfriVoicesDataset


class AfriVoicesWhisperDataset:
    def __init__(
        self,
        manifest_path,
        processor_name="openai/whisper-large-v3",
        languages=None,
        max_duration=30.0,
        max_rows_per_language=None,
    ):
        self.base_dataset = AfriVoicesDataset(
            manifest_path=manifest_path,
            languages=languages,
            max_duration=max_duration,
            max_rows_per_language=max_rows_per_language,
        )

        self.processor = WhisperProcessor.from_pretrained(
            processor_name,
            language=None,
            task="transcribe",
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]

        input_features = self.processor.feature_extractor(
            sample["audio"],
            sampling_rate=sample["sample_rate"],
            return_tensors="pt",
        ).input_features[0]

        labels = self.processor.tokenizer(
            sample["transcription"],
            return_tensors="pt",
        ).input_ids[0]

        return {
            "unique_id": sample["unique_id"],
            "language": sample["language"],
            "input_features": input_features,
            "labels": labels,
            "transcription": sample["transcription"],
            "duration_seconds": sample["duration_seconds"],
            "resolved_source": sample["resolved_source"],
        }
