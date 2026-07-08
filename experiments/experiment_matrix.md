# Master Experiment Matrix

This is the high-level experiment tracking table for Voice-Compass / AfriVoices ASR.

The goal is to track which experiments actually help reduce Kaggle-style average WER.

| Exp | Phase | Goal | Primary Variable | Dataset Version | Model | Status | Decision | Notes |
|---|---|---|---|---|---|---|---|---|
| exp056 | Phase 4 | Build ANV audio index | Audio indexing | anv_index_v1 | N/A | Completed | Adopt | Required for fast 5-language audio lookup |
| exp057 | Phase 4 | Build Swahili TAR index | Audio indexing | swahili_index_v1 | N/A | Completed | Adopt | Required for Swahili spontaneous data |
| exp058 | Phase 4 | Indexed multilingual smoke | Index-aware training | balanced_6lang_v0 | Whisper Small + LoRA | Completed | Partial adopt | Keep indexed loading; exposed validation/cache bottleneck |
| exp059 | Phase 4 | Configurable trainer + Kaggle WER | YAML config + balanced data + per-language WER | balanced_6lang_v1 | Whisper Small + LoRA | In progress | Pending | Clean dataset and validate trainer end-to-end |

## Required WER Columns Going Forward

Each WER-producing experiment should record:

- global_wer
- kaggle_avg_wer
- wer_swa
- wer_kik
- wer_luo
- wer_som
- wer_mas
- wer_kln

## Decision Labels

- Adopt
- Partial adopt
- Reject
- Pending


| Exp    | Status     | Decision      |
| ------ | ---------- | ------------- |
| Exp059 | ✅ Complete | Partial Adopt |

| Experiment | Pipeline Version | Notes                     |
| ---------- | ---------------- | ------------------------- |
| Exp059     | Legacy           | TAR thrashing discovered  |
| Exp060     | Optimized        | Infrastructure validation |
| Exp061     | Optimized        | New multilingual baseline |
