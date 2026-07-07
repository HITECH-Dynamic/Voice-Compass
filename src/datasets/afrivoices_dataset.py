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
        anv_index_path=None,
        swahili_index_path=None,
    ):
        self.manifest_path = Path(manifest_path)
        self.languages = languages
        self.max_duration = max_duration
        self.max_rows_per_language = max_rows_per_language
        self.anv_index_path = Path(anv_index_path) if anv_index_path else None
        self.swahili_index_path = Path(swahili_index_path) if swahili_index_path else None
        self.anv_index = self._load_anv_index()
        self.swahili_index = self._load_swahili_index()
        self.parquet_cache = {}
        self.tar_cache = {}
        self.df = self._load_manifest()

    def _load_anv_index(self):
        if self.anv_index_path is None:
            return None

        if not self.anv_index_path.exists():
            raise FileNotFoundError(f"ANV index not found: {self.anv_index_path}")

        index = pd.read_parquet(self.anv_index_path)
        return index


    def _load_swahili_index(self):
        if self.swahili_index_path is None:
            return None

        if not self.swahili_index_path.exists():
            raise FileNotFoundError(f"Swahili index not found: {self.swahili_index_path}")

        index = pd.read_parquet(self.swahili_index_path)
        return index

    def _lookup_indexed_parquet(self, row):
        if "indexed_parquet_file" in row.index:
            value = str(row.get("indexed_parquet_file", "")).strip()
            if value:
                return value

        if self.anv_index is None:
            return None

        filename = str(row["audio_filename"])
        language = str(row["language"])

        matches = self.anv_index[
            (self.anv_index["language"].astype(str) == language)
            & (self.anv_index["audio_filename"].astype(str) == filename)
        ]

        if matches.empty:
            return None

        return matches.iloc[0]["parquet_file"]

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

    def _get_tar_index(self, archive_path):
        archive_key = str(archive_path)

        if archive_key not in self.tar_cache:
            print(f"[TAR CACHE MISS] indexing {archive_path}")
            tar = tarfile.open(archive_path, "r:*")
            members = {m.name: m for m in tar.getmembers()}
            self.tar_cache[archive_key] = (tar, members)
        else:
            print(f"[TAR CACHE HIT] {archive_path}")

        return self.tar_cache[archive_key]


    def close(self):
        for tar, _members in self.tar_cache.values():
            try:
                tar.close()
            except Exception:
                pass
        self.tar_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


    def _resolve_anv_parquet(self, row):
        repo_id = row["source_repo"]
        filename = row["audio_filename"]

        indexed_parquet = self._lookup_indexed_parquet(row)
        if indexed_parquet:
            candidates = [indexed_parquet]
        else:
            candidates = json.loads(row["audio_parquet_candidates"])

        for parquet_file in candidates:
            parquet_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=parquet_file,
            )

            if parquet_file not in self.parquet_cache:
                self.parquet_cache[parquet_file] = pd.read_parquet(
                    parquet_path,
                    columns=["audio", "filename"],
                )

            df = self.parquet_cache[parquet_file]
            match = df[df["filename"].astype(str) == str(filename)]

            if not match.empty:
                audio_obj = match.iloc[0]["audio"]
                audio_bytes = audio_obj["bytes"]
                audio, sr = self._decode_audio_bytes(audio_bytes, suffix=".wav")
                return audio, sr, parquet_file

        raise FileNotFoundError(f"Could not find {filename} in parquet candidates")

    def _resolve_swahili(self, row):
        if self.swahili_index is None:
            raise FileNotFoundError(
                "Swahili TAR index required for swahili_tar_ref rows. "
                "Pass swahili_index_path to AfriVoicesDataset."
            )

        repo_id = row["source_repo"]
        audio_filename = str(row["audio_filename"])
        raw_split = str(row["raw_split"])

        matches = self.swahili_index[
            (self.swahili_index["raw_split"].astype(str) == raw_split)
            & (self.swahili_index["audio_filename"].astype(str) == audio_filename)
        ]

        if matches.empty:
            raise FileNotFoundError(f"Could not resolve Swahili audio {raw_split}/{audio_filename}")

        match = matches.iloc[0]
        archive_file = match["archive_file"]
        member_name = match["member_name"]

        archive_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=archive_file,
        )

        tar, members = self._get_tar_index(archive_path)

        tarinfo = members.get(member_name)
        if tarinfo is None:
            raise FileNotFoundError(f"Could not find {member_name} in {archive_file}")

        extracted = tar.extractfile(tarinfo)
        if extracted is None:
            raise FileNotFoundError(f"Could not extract {member_name} from {archive_file}")

        audio_bytes = extracted.read()

        suffix = Path(audio_filename).suffix or ".webm"
        audio, sr = self._decode_audio_bytes(audio_bytes, suffix=suffix)
        return audio, sr, archive_file + "::" + member_name


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
