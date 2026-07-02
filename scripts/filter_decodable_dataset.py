from pathlib import Path
import argparse
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.afrivoices_dataset import AfriVoicesDataset


def filter_file(input_path, output_path, bad_rows_path):
    ds = AfriVoicesDataset(manifest_path=input_path, max_duration=30.0)

    good_indices = []
    bad_rows = []

    for i in range(len(ds)):
        row = ds.df.iloc[i]
        try:
            sample = ds[i]
            if sample["audio"] is not None and len(sample["audio"]) > 0:
                good_indices.append(i)
            else:
                raise ValueError("Decoded audio was empty.")
        except Exception as e:
            bad_rows.append({
                "row_index": i,
                "unique_id": row.get("unique_id", ""),
                "language": row.get("language", ""),
                "audio_filename": row.get("audio_filename", ""),
                "indexed_parquet_file": row.get("indexed_parquet_file", ""),
                "error": repr(e),
            })

    good_df = ds.df.iloc[good_indices].reset_index(drop=True)
    bad_df = pd.DataFrame(bad_rows)

    output_path = Path(output_path)
    bad_rows_path = Path(bad_rows_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bad_rows_path.parent.mkdir(parents=True, exist_ok=True)

    good_df.to_parquet(output_path, index=False)
    bad_df.to_csv(bad_rows_path, index=False)

    print(f"Input: {input_path}")
    print(f"Good rows: {len(good_df)}")
    print(f"Bad rows: {len(bad_df)}")
    print(f"Wrote clean dataset: {output_path}")
    print(f"Wrote bad row report: {bad_rows_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bad-rows", required=True)
    args = parser.parse_args()

    filter_file(args.input, args.output, args.bad_rows)


if __name__ == "__main__":
    main()
