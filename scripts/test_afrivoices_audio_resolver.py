"""
Exp037 — AfriVoices Audio Resolver Test

Purpose:
Verify that unified manifest rows can resolve to real audio for:
- ANV-KE parquet_bytes rows
- Swahili swahili_tar_ref rows

Run in Colab after Exp036B manifests have been generated.
"""

from pathlib import Path
from io import BytesIO
import argparse
import json
import tarfile
import tempfile

import librosa
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files


MANIFEST_DIR = Path("data/afrivoices/manifests")


def decode_audio_bytes(audio_bytes: bytes, suffix: str = ".wav"):
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        audio, sr = librosa.load(tmp.name, sr=16000, mono=True)
    return audio, sr


def decode_audio_file(path: str):
    audio, sr = librosa.load(path, sr=16000, mono=True)
    return audio, sr


def resolve_anv_parquet(row: pd.Series):
    repo_id = row["source_repo"]
    filename = row["audio_filename"]
    candidates = json.loads(row["audio_parquet_candidates"])

    if not candidates:
        raise ValueError(f"No parquet candidates for row {row['unique_id']}")

    for parquet_file in candidates:
        print(f"  Searching parquet: {parquet_file}")
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
            audio, sr = decode_audio_bytes(audio_bytes, suffix=".wav")
            return audio, sr, parquet_file

    raise FileNotFoundError(f"Could not find {filename} in parquet candidates")


def resolve_swahili(row: pd.Series):
    repo_id = row["source_repo"]
    audio_ref = row["audio_ref"]
    audio_filename = row["audio_filename"]
    folder = str(audio_ref).split("/")[0]

    files = list_repo_files(repo_id=repo_id, repo_type="dataset")

    direct_matches = [
        f for f in files
        if f.endswith(audio_filename) and folder in f
    ]

    if direct_matches:
        audio_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=direct_matches[0],
        )
        audio, sr = decode_audio_file(audio_path)
        return audio, sr, direct_matches[0]

    archive_matches = [
        f for f in files
        if folder in f and (f.endswith(".tar") or f.endswith(".tar.gz") or f.endswith(".tar.xz"))
    ]

    if not archive_matches:
        raise FileNotFoundError(
            f"No direct file or archive found for {audio_ref}"
        )

    for archive_file in archive_matches:
        print(f"  Searching archive: {archive_file}")
        archive_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=archive_file,
        )

        with tarfile.open(archive_path, "r:*") as tar:
            members = [
                m for m in tar.getmembers()
                if Path(m.name).name == audio_filename
            ]

            if members:
                extracted = tar.extractfile(members[0])
                if extracted is None:
                    continue

                audio_bytes = extracted.read()
                suffix = Path(audio_filename).suffix or ".webm"
                audio, sr = decode_audio_bytes(audio_bytes, suffix=suffix)
                return audio, sr, archive_file + "::" + members[0].name

    raise FileNotFoundError(f"Could not resolve Swahili audio {audio_ref}")


def load_sample_rows():
    train_path = MANIFEST_DIR / "manifest_train.parquet"
    dev_path = MANIFEST_DIR / "manifest_dev.parquet"

    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(
            "Manifest parquet files not found. Run scripts/build_afrivoices_manifest.py first."
        )

    df = pd.concat(
        [
            pd.read_parquet(train_path),
            pd.read_parquet(dev_path),
        ],
        ignore_index=True,
    )

    df = df[
        df["transcription"].notna()
        & (df["transcription"].astype(str).str.strip() != "")
        & df["audio_ref"].notna()
        & (df["audio_ref"].astype(str).str.strip() != "")
    ].copy()

    samples = []

    anv = df[df["audio_source_type"] == "parquet_bytes"]
    swa = df[df["audio_source_type"] == "swahili_tar_ref"]

    if anv.empty:
        raise ValueError("No ANV parquet_bytes rows found.")
    if swa.empty:
        raise ValueError("No Swahili swahili_tar_ref rows found.")

    samples.append(anv.iloc[0])
    samples.append(swa.iloc[0])

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=2)
    args = parser.parse_args()

    samples = load_sample_rows()[: args.max_samples]

    for i, row in enumerate(samples, start=1):
        print("\n" + "=" * 80)
        print(f"Sample {i}")
        print(f"unique_id: {row['unique_id']}")
        print(f"language: {row['language']}")
        print(f"audio_source_type: {row['audio_source_type']}")
        print(f"audio_ref: {row['audio_ref']}")
        print(f"transcription: {str(row['transcription'])[:120]}")

        if row["audio_source_type"] == "parquet_bytes":
            audio, sr, source = resolve_anv_parquet(row)
        elif row["audio_source_type"] == "swahili_tar_ref":
            audio, sr, source = resolve_swahili(row)
        else:
            raise ValueError(f"Unknown audio_source_type: {row['audio_source_type']}")

        print(f"Resolved source: {source}")
        print(f"Sample rate: {sr}")
        print(f"Audio samples: {len(audio)}")
        print(f"Duration seconds: {len(audio) / sr:.2f}")

    print("\nAudio resolver test complete.")


if __name__ == "__main__":
    main()
