# Dataset Validation Workflow

Training should use clean, validated parquet files whenever possible.

## Standard Workflow

1. Build raw balanced dataset.
2. Validate audio once.
3. Save clean parquet.
4. Save bad-audio report.
5. Save metadata.
6. Reuse the clean dataset for future model experiments.

## Do Not Revalidate When Only Changing

- learning rate
- LoRA rank
- LoRA alpha
- batch size
- gradient accumulation
- scheduler
- max training steps
- decoding parameters

## Revalidate When Changing

- samples per language
- sampling strategy
- manifests
- audio indexes
- source data
- max duration filter

## Example

```bash
python scripts/validate_audio_dataset.py \
  --input data/processed/exp059_train.parquet \
  --output data/processed/exp059_train_clean.parquet \
  --bad_report reports/dataset_analysis/exp059_train_bad_audio.csv \
  --metadata datasets/dataset_v1/train_metadata.json \
  --dataset_version dataset_v1
