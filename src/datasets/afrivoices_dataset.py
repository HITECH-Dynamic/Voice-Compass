"""
Exp038A — AfriVoices Dataset Class

Reusable dataset layer:
manifest row -> audio resolver -> waveform + transcript + metadata

No Whisper processor yet.
"""

from pathlib import Path
from io import BytesIO
import json
import tarfile
import tempfile

import librosa
import pandas as pd
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download, list_repo_files


class AfriVoicesDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        languages=None,
        max_duration=30.0,
        max_rows_per_language=None,
    ):
        self.manifest_path = Path(manifest_path)
        self.languages = languages
        self.max_duration = max_duration
        self.max_rows_per_language = max_rows_per_language
        self.df = self._load_manifest()

    def _load_manifest(self):
        df = pd.read_parquet(self.manifest_path)

        df = df[
            df["transcription"].notna()
            & (df["transcription"].astype(str).str.strip() != "")
            & df["audio_ref"].notna()
            & (df["audio_ref"].astype(str).str.strip() != "")
            & df["duration_seconds"].notna()
            & (df["duration_seconds"] > 0)
        ].copy()

        if self.max_duration is not None:
            df = df[df["duration_seconds"] <= self.max_duration].copy()

        if self.languages:
            df = df[df["language"].isin(self.languages)].copy()

        if self.max_rows_per_language:
            df = (
                df.groupby("language", group_keys=False)
                .head(self.max_rows_per_language)
                .reset_index(drop=True)
            )
        else:
            df = df.reset_index(drop=True)

        return df

    def __len__(self):
        return len(self.df)

    def _decode_audio_bytes(self, audio_bytes, suffix=".wav"):
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            audio, sr = librosa.load(tmp.name, sr=16000, mono=True)
        return audio, sr

    def _resolve_anv_parquet(self, row):
        repo_id = row["source_repo"]
        filename = row["audio_filename"]
        candidates = json.loads(row["audio_parquet_candidates"])

        for parquet_file in candidates:
            parquet_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=parquet_file,
            )

            df = pd.read_parquet(parquet_path, columns=["audio", "filename"])
            match = df[df["filename"].astype(str) == str(filename)]

            if not match.empty:
                audio_obj = match.iloc[0]["audio"]
                audio_bytes = audio_obj["bytes"]
                audio, sr = self._decode_audio_bytes(audio_bytes, suffix=".wav")
                return audio, sr, parquet_file

        raise FileNotFoundError(f"Could not find {filename} in parquet candidates")

    def _resolve_swahili(self, row):
        repo_id = row["source_repo"]
        audio_ref = row["audio_ref"]
        audio_filename = row["audio_filename"]
        folder = str(audio_ref).split("/")[0]

        files = list_repo_files(repo_id=repo_id, repo_type="dataset")

        archive_matches = [
            f for f in files
            if folder in f and (f.endswith(".tar") or f.endswith(".tar.gz") or f.endswith(".tar.xz"))
        ]

        if not archive_matches:
            raise FileNotFoundError(f"No Swahili archive found for {audio_ref}")

        for archive_file in archive_matches:
            archive_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=archive_file,
            )

            with tarfile.open(archive_path, "r:*") as tar:
                for member in tar.getmembers():
                    if Path(member.name).name == audio_filename:
                        extracted = tar.extractfile(member)
                        if extracted is None:
                            continue

                        audio_bytes = extracted.read()
                        suffix = Path(audio_filename).suffix or ".webm"
                        audio, sr = self._decode_audio_bytes(audio_bytes, suffix=suffix)
                        return audio, sr, archive_file + "::" + member.name

        raise FileNotFoundError(f"Could not resolve Swahili audio {audio_ref}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if row["audio_source_type"] == "parquet_bytes":
            audio, sr, resolved_source = self._resolve_anv_parquet(row)
        elif row["audio_source_type"] == "swahili_tar_ref":
            audio, sr, resolved_source = self._resolve_swahili(row)
        else:
            raise ValueError(f"Unknown audio_source_type: {row['audio_source_type']}")

        return {
            "unique_id": row["unique_id"],
            "language": row["language"],
            "audio": audio,
            "sample_rate": sr,
            "transcription": row["transcription"],
            "duration_seconds": len(audio) / sr,
            "resolved_source": resolved_source,
        }
