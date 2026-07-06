# Voice-Compass Experiment Config Schema

This document defines the configuration schema for multilingual ASR experiments.

## Purpose

The YAML config should define what experiment is being run.

The Python training code should define how the experiment is executed.

## Top-Level Sections

- `experiment`
- `dataset`
- `inputs`
- `outputs`
- `audio`
- `model`
- `lora`
- `training`
- `augmentation`
- `evaluation`
- `tracking`
- `deployment`

---

## experiment

| Key | Type | Default | Description |
|---|---:|---:|---|
| `name` | string | required | Experiment name, for example `exp059` |
| `description` | string | optional | Short experiment description |
| `objective` | string | optional | Main research objective |
| `phase` | string | optional | Project phase |
| `seed` | integer | `42` | Global random seed |

---

## dataset

| Key | Type | Default | Description |
|---|---:|---:|---|
| `languages` | list[string] | required | ISO language codes |
| `samples_per_language` | integer | optional | Total rows sampled per language |
| `train_rows_per_language` | integer | optional | Explicit train rows per language |
| `eval_rows_per_language` | integer | optional | Explicit eval rows per language |
| `eval_fraction` | float | `0.2` | Evaluation fraction |
| `balancing_strategy` | string | `balanced` | Sampling strategy |
| `speech_types` | list[string] | optional | Scripted/spontaneous filters |
| `min_duration_seconds` | float | `0.25` | Minimum clip duration |
| `max_duration_seconds` | float | `30.0` | Maximum clip duration |
| `shuffle` | boolean | `true` | Shuffle sampled rows |

---

## inputs

| Key | Type | Default | Description |
|---|---:|---:|---|
| `train_manifest` | path | required | Unified train manifest |
| `dev_manifest` | path | optional | Unified dev manifest |
| `test_manifest` | path | optional | Unified test manifest |
| `anv_audio_index` | path | optional | ANV parquet audio index |
| `swahili_tar_index` | path | optional | Swahili TAR audio index |

---

## outputs

| Key | Type | Default | Description |
|---|---:|---:|---|
| `train` | path | required | Output training parquet |
| `eval` | path | required | Output eval parquet |
| `summary` | path | optional | Dataset summary CSV |
| `predictions` | path | optional | Saved predictions |
| `metrics` | path | optional | Saved metrics JSON |

---

## audio

| Key | Type | Default | Description |
|---|---:|---:|---|
| `sample_rate` | integer | `16000` | Target audio sample rate |
| `max_audio_seconds` | float | `30.0` | Maximum audio duration |
| `normalize` | boolean | `false` | Normalize audio amplitude |
| `trim_silence` | boolean | `false` | Trim silence before feature extraction |
| `cache_enabled` | boolean | `false` | Use local cached audio |
| `cache_dir` | path | `data/cache/audio` | Cache location |
| `overwrite_cache` | boolean | `false` | Rebuild cached files |

---

## model

| Key | Type | Default | Description |
|---|---:|---:|---|
| `name` | string | `openai/whisper-small` | Base Whisper model |
| `processor` | string | same as `name` | Processor/tokenizer model |
| `max_sequence_length` | integer | optional | Maximum input/context sequence setting |
| `decoder_max_length` | integer | optional | Maximum output transcript token length |
| `forced_decoder_ids` | value | optional | Whisper decoder prompt control |
| `suppress_tokens` | value | optional | Token suppression control |

---

## lora

| Key | Type | Default | Description |
|---|---:|---:|---|
| `enabled` | boolean | `true` | Enable LoRA fine-tuning |
| `rank` | integer | `16` | LoRA rank |
| `alpha` | integer | `32` | LoRA alpha |
| `dropout` | float | `0.05` | LoRA dropout |
| `bias` | string | `none` | LoRA bias setting |
| `target_modules` | list[string] | `q_proj`, `v_proj` | LoRA target modules |

---

## training

| Key | Type | Default | Description |
|---|---:|---:|---|
| `output_dir` | path | required | Checkpoint directory |
| `learning_rate` | float | `2e-5` | Learning rate |
| `optimizer` | string | trainer default | Optimizer |
| `scheduler` | string | `linear` | LR scheduler |
| `warmup_steps` | integer | `0` | Warmup steps |
| `weight_decay` | float | `0.0` | Weight decay |
| `max_steps` | integer | required | Training steps |
| `num_train_epochs` | float | optional | Epoch-based training |
| `per_device_train_batch_size` | integer | `2` | Train batch size |
| `per_device_eval_batch_size` | integer | `2` | Eval batch size |
| `gradient_accumulation_steps` | integer | `1` | Gradient accumulation |
| `fp16` | boolean | `true` | Use fp16 training |
| `bf16` | boolean | `false` | Use bf16 training |
| `gradient_checkpointing` | boolean | `false` | Reduce memory use |

---

## augmentation

| Key | Type | Default | Description |
|---|---:|---:|---|
| `enabled` | boolean | `false` | Enable audio augmentation |
| `speed_perturbation` | list[float] | optional | Speed factors, for example `[0.9, 1.0, 1.1]` |
| `noise` | boolean | `false` | Add background noise |
| `volume` | boolean | `false` | Volume perturbation |
| `reverb` | boolean | `false` | Reverb augmentation |

---

## evaluation

| Key | Type | Default | Description |
|---|---:|---:|---|
| `report_global_wer` | boolean | `true` | Compute global WER |
| `report_per_language_wer` | boolean | `true` | Compute WER per language |
| `report_kaggle_average_wer` | boolean | `true` | Compute unweighted mean WER across six languages |
| `language_order` | list[string] | `swa,kik,luo,som,mas,kln` | Kaggle language order |
| `save_predictions` | boolean | `true` | Save prediction table |
| `save_metrics` | boolean | `true` | Save metrics JSON |

---

## tracking

| Key | Type | Default | Description |
|---|---:|---:|---|
| `report_to` | string | `wandb` | Reporting backend |
| `wandb_project` | string | `voice-compass` | W&B project |
| `logging_steps` | integer | `1` | Log frequency |
| `eval_steps` | integer | `100` | Eval frequency |
| `save_steps` | integer | `100` | Save frequency |

---

## deployment

| Key | Type | Default | Description |
|---|---:|---:|---|
| `export_onnx` | boolean | `false` | Export ONNX model |
| `quantization` | string | optional | Quantization mode |
| `raspberry_pi_target` | boolean | `false` | Optimize for Raspberry Pi testing |

