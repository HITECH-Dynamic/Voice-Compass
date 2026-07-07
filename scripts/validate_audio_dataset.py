import argparse
import json
import hashlib
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.afrivoices_dataset import AfriVoicesDataset


def file_hash(path):
    path = Path(path)
    if not path.exists():
        return None

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset(args):
    ds = AfriVoicesDataset(
        manifest_path=args.input,
        anv_index_path=args.anv_index,
        swahili_index_path=args.swahili_index,
        max_duration=args.max_duration,
    )

    good = []
    bad = []

    for i in tqdm(range(len(ds)), desc=f"Validating {args.input}"):
        try:
            _ = ds[i]
            good.append(i)
        except Exception as e:
            row = ds.df.iloc[i].to_dict()
            row["row_index"] = i
            row["error"] = repr(e)
            bad.append(row)

    clean_df = ds.df.iloc[good].copy()
    bad_df = pd.DataFrame(bad)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.bad_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metadata).parent.mkdir(parents=True, exist_ok=True)

    clean_df.to_parquet(args.output, index=False)
    bad_df.to_csv(args.bad_report, index=False)

    metadata = {
        "dataset_version": args.dataset_version,
        "input": args.input,
        "output": args.output,
        "bad_report": args.bad_report,
        "rows_input": len(ds),
        "rows_clean": len(clean_df),
        "rows_bad": len(bad_df),
        "validated": True,
        "max_duration": args.max_duration,
        "anv_index": args.anv_index,
        "swahili_index": args.swahili_index,
        "input_sha256": file_hash(args.input),
        "anv_index_sha256": file_hash(args.anv_index),
        "swahili_index_sha256": file_hash(args.swahili_index),
    }

    with open(args.metadata, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nInput rows: {len(ds)}")
    print(f"Clean rows: {len(clean_df)}")
    print(f"Bad rows: {len(bad_df)}")
    print(f"Wrote clean dataset: {args.output}")
    print(f"Wrote bad report: {args.bad_report}")
    print(f"Wrote metadata: {args.metadata}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bad_report", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--dataset_version", default="dataset_v1")
    parser.add_argument("--anv_index", default="data/afrivoices/indexes/anv_audio_index.parquet")
    parser.add_argument("--swahili_index", default="data/afrivoices/indexes/swahili_tar_index.parquet")
    parser.add_argument("--max_duration", type=float, default=30.0)
    args = parser.parse_args()

    validate_dataset(args)


if __name__ == "__main__":
    main()
