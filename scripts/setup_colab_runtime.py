from pathlib import Path
import subprocess
import sys


def run(cmd, required=True):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if required and result.returncode != 0:
        raise SystemExit(f"Command failed: {cmd}")


def exists(path):
    return Path(path).exists()


def main():
    print("Voice-Compass Colab runtime setup starting...")

    run(f"{sys.executable} -m pip install -q evaluate jiwer wandb")

    required_manifest_files = [
        "data/afrivoices/manifests/manifest_train.parquet",
        "data/afrivoices/manifests/manifest_dev.parquet",
        "data/afrivoices/manifests/manifest_test.parquet",
    ]

    if not all(exists(p) for p in required_manifest_files):
        print("\nManifest files missing. Building AfriVoices manifests...")
        run(f"{sys.executable} -u scripts/build_afrivoices_manifest.py")
    else:
        print("\nManifest files already exist.")

    index_path = "data/afrivoices/indexes/anv_audio_index.parquet"
    if not exists(index_path):
        raise SystemExit(
            "\nMissing ANV audio index:\n"
            "data/afrivoices/indexes/anv_audio_index.parquet\n\n"
            "Upload anv_audio_index.parquet to data/afrivoices/indexes/ before continuing."
        )
    else:
        print("\nANV audio index found.")

    clean_train = "data/processed/exp048_train_clean.parquet"
    clean_eval = "data/processed/exp048_eval_clean.parquet"

    if not exists("data/processed/exp048_train.parquet") or not exists("data/processed/exp048_eval.parquet"):
        print("\nProcessed Exp048 datasets missing. Rebuilding...")
        run(
            f"{sys.executable} -u scripts/build_multilingual_dataset.py "
            "--config configs/exp048_5lang_parquet_multilingual.yaml"
        )
    else:
        print("\nProcessed Exp048 datasets already exist.")

    if not exists(clean_train):
        print("\nClean train dataset missing. Running decode filter...")
        run(
            f"{sys.executable} -u scripts/filter_decodable_dataset.py "
            "--input data/processed/exp048_train.parquet "
            "--output data/processed/exp048_train_clean.parquet "
            "--bad-rows reports/dataset_analysis/exp048_train_bad_audio.csv"
        )
    else:
        print("\nClean train dataset already exists.")

    if not exists(clean_eval):
        print("\nClean eval dataset missing. Running decode filter...")
        run(
            f"{sys.executable} -u scripts/filter_decodable_dataset.py "
            "--input data/processed/exp048_eval.parquet "
            "--output data/processed/exp048_eval_clean.parquet "
            "--bad-rows reports/dataset_analysis/exp048_eval_bad_audio.csv"
        )
    else:
        print("\nClean eval dataset already exists.")

    print("\nSetup complete. Ready for Exp049 training.")
    print("\nExpected files:")
    for p in [
        *required_manifest_files,
        index_path,
        "data/processed/exp048_train_clean.parquet",
        "data/processed/exp048_eval_clean.parquet",
    ]:
        print(f" - {p}: {'FOUND' if exists(p) else 'MISSING'}")


if __name__ == "__main__":
    main()
