# Voice-Compass Experiment Matrix

## Purpose

This document is the master optimization roadmap for the AfriVoices ASR project.

Unlike the experiment log (chronological), this matrix is organized by **optimization variable**. It tracks:

- what can be tuned,
- whether it has been explored,
- current best values,
- research priority,
- expected impact on WER.

---

# Priority Scale

★★★★★ = Highest expected impact on Kaggle WER

★★★★☆ = High

★★★☆☆ = Medium

★★☆☆☆ = Low

★☆☆☆☆ = Very Low

---

# Status

⬜ Not Started

🟨 In Progress

✅ Completed

🏆 Best Known Configuration

---

# DATASET

| Variable | Current | Best | Status | Priority | Notes |
|-----------|--------|------|--------|----------|------|
| Samples per language | 80 | TBD | 🟨 | ★★★★★ | Increase gradually while maintaining balance |
| Train / Eval split | 80 / 20 | TBD | 🟨 | ★★★★☆ | Monitor generalization |
| Language balancing | Equal sampling | Equal sampling | ✅🏆 | ★★★★★ | Mirrors Kaggle weighting philosophy |
| Speech type | Mixed | TBD | ⬜ | ★★★☆☆ | Compare scripted vs spontaneous |
| Domain balancing (Swahili) | Random | TBD | ⬜ | ★★★☆☆ | Agriculture / Health / Government / etc. |
| Maximum duration | 30 s | TBD | 🟨 | ★★☆☆☆ | Explore shorter clips |

---

# AUDIO PIPELINE

| Variable | Current | Best | Status | Priority | Notes |
|-----------|--------|------|--------|----------|------|
| ANV Global Index | Enabled | Enabled | ✅🏆 | ★★★★★ | Exp056 |
| Swahili TAR Index | Enabled | Enabled | ✅🏆 | ★★★★★ | Exp057 |
| Local audio cache | Disabled | TBD | ⬜ | ★★★★☆ | Candidate for Exp059 |
| Whisper feature cache | Not implemented | TBD | ⬜ | ★★★☆☆ | Future optimization |
| Audio normalization | Disabled | TBD | ⬜ | ★★☆☆☆ | Evaluate impact |
| Silence trimming | Disabled | TBD | ⬜ | ★★☆☆☆ | Evaluate impact |

---

# MODEL

| Variable | Current | Best | Status | Priority | Notes |
|-----------|--------|------|--------|----------|------|
| Base model | Whisper Small | Whisper Small | ✅🏆 | ★★★★★ | Current baseline |
| Context / Max sequence length | Default | TBD | ⬜ | ★★★★☆ | Particularly relevant for longer utterances |
| Decoder max length | Default | TBD | ⬜ | ★★★☆☆ | Prevent transcript truncation |
| Forced decoder IDs | Default | TBD | ⬜ | ★★☆☆☆ | Whisper decoding control |
| Suppress tokens | Default | TBD | ⬜ | ★★☆☆☆ | Advanced decoding |

---

# LORA

| Variable | Current | Best | Status | Priority | Notes |
|-----------|--------|------|--------|----------|------|
| Rank | 16 | TBD | 🟨 | ★★★★☆ | Previous optimization work |
| Alpha | 32 | TBD | 🟨 | ★★★★☆ | Previous optimization work |
| Dropout | 0.05 | TBD | 🟨 | ★★★☆☆ | Regularization |
| Target modules | q_proj,v_proj | TBD | ⬜ | ★★★☆☆ | Explore additional modules |

---

# TRAINING

| Variable | Current | Best | Status | Priority | Notes |
|-----------|--------|------|--------|----------|------|
| Learning rate | 2e-5 | TBD | 🟨 | ★★★★★ | Continue tuning |
| Batch size | 2 | TBD | 🟨 | ★★★★☆ | GPU dependent |
| Gradient accumulation | 2 | TBD | 🟨 | ★★★☆☆ | Effective batch size |
| Scheduler | Linear | TBD | ⬜ | ★★★☆☆ | Cosine / Constant / etc. |
| Warmup steps | 0 | TBD | ⬜ | ★★☆☆☆ | Stability study |
| Weight decay | 0 | TBD | ⬜ | ★★★☆☆ | Regularization |
| Max training steps | 100 | TBD | 🟨 | ★★★★☆ | Scale with dataset size |

---

# AUGMENTATION

| Variable | Current | Best | Status | Priority | Notes |
|-----------|--------|------|--------|----------|------|
| Speed perturbation | Disabled | TBD | ⬜ | ★★★★☆ | Tiago recommendation |
| Noise augmentation | Disabled | TBD | ⬜ | ★★★★☆ | Robustness |
| Volume augmentation | Disabled | TBD | ⬜ | ★★★☆☆ | Robustness |
| Reverberation | Disabled | TBD | ⬜ | ★★☆☆☆ | Later experiment |

---

# EVALUATION

| Variable | Current | Best | Status | Priority | Notes |
|-----------|--------|------|--------|----------|------|
| Overall WER | Enabled | TBD | ✅ | ★★★★★ | Existing metric |
| Per-language WER | Planned | TBD | 🟨 | ★★★★★ | Align with Kaggle |
| Kaggle Average WER | Planned | TBD | 🟨 | ★★★★★ | Primary optimization target |
| Runtime | Recorded | TBD | 🟨 | ★★★☆☆ | Engineering metric |
| Throughput | Planned | TBD | ⬜ | ★★☆☆☆ | Samples/sec |

---

# DEPLOYMENT

| Variable | Current | Best | Status | Priority | Notes |
|-----------|--------|------|--------|----------|------|
| ONNX export | Not started | TBD | ⬜ | ★★☆☆☆ | Post-hackathon |
| Quantization | Not started | TBD | ⬜ | ★★☆☆☆ | Raspberry Pi |
| Raspberry Pi benchmarking | Not started | TBD | ⬜ | ★★☆☆☆ | Edge deployment |

---

# Guiding Principle

For every future experiment:

1. Change **one primary optimization variable** whenever practical.
2. Measure:
   - Overall WER
   - Per-language WER
   - Kaggle Average WER
   - Runtime
3. Update this matrix with the best-known configuration.
4. Promote improvements to the default YAML configuration only after they have been validated.

This document is the master roadmap for Phase 4 and Phase 5 optimization.
