# AfriVoices Unified Manifest Schema

Each final manifest row should contain:

| Column | Description |
|--------|-------------|
| id | Unique utterance ID |
| language | ISO 639-3 language code |
| language_name | Human-readable language name |
| source_repo | Hugging Face source repository |
| split | train / dev / test |
| speech_type | scripted / unscripted / unknown |
| audio_path | Local audio path or resolved HF file reference |
| transcription | Ground-truth transcript when available |
| duration_seconds | Audio duration in seconds |
| sample_rate | Audio sample rate |
| speaker_id | Speaker identifier if available |
| dialect | Dialect or region if available |
| metadata | Optional raw metadata JSON |

Target output files:

- data/afrivoices/manifests/manifest_train.parquet
- data/afrivoices/manifests/manifest_dev.parquet
- data/afrivoices/manifests/manifest_test.parquet
