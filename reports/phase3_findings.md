# Phase 3 Findings Report

## Project

AfriVoices ASR Hackathon

Model:
Whisper Large-v3 + LoRA

Status:
IN PROGRESS

---

# Objective

Optimize Whisper Large-v3 for the AfriVoices ASR Hackathon using a controlled one-variable-at-a-time experimental methodology.

Primary Metric:

- Word Error Rate (WER)

---

# Champion Timeline

| Day | Experiment | WER | Why it became Champion |
|-----|------------|------|------------------------|
| Day 8 | Exp026 | **0.2318** | Improved validation split and optimized LoRA configuration |

Current Champion:

Exp026
WER = **0.2318**

---

# Experiments Completed

## Validation Split

### Finding

Increasing the validation set produced a more reliable evaluation while maintaining strong performance.

Decision:

✓ Adopt larger validation split.

---

## LoRA Target Modules

Variables Tested

- Attention only
- Attention + Feed Forward

Finding

Including the feed-forward layers improved adaptation.

Decision

✓ Keep:

- q_proj
- k_proj
- v_proj
- out_proj
- fc1
- fc2

---

## LoRA Rank

Finding

Higher rank did not consistently improve WER.

Decision

✓ Rank = 16

---

## LoRA Alpha

Finding

Alpha 32 provided sufficient adapter capacity.

Decision

✓ Alpha = 32

---

## Learning Rate

Finding

2e-5 produced the most stable convergence.

Decision

✓ Learning Rate = 2e-5

---

## Scheduler

Experiments

- Linear
- Cosine
- constant_with_warmup

Finding

Linear remained the most stable scheduler.

constant_with_warmup diverged late in training.

Decision

✓ Linear scheduler

---

## Warmup

Finding

Increasing warmup steps did not improve WER.

Decision

✓ Warmup = 10

---

## Weight Decay

Experiments

- None
- 0.001
- 0.05

Finding

Weight decay produced no measurable improvement.

Decision

✓ No explicit weight decay

---

## LoRA Dropout

Experiments

- 0.05
- 0.10

Finding

Increasing LoRA dropout produced no measurable improvement.

Decision

✓ LoRA Dropout = 0.05

---

# Current Champion Configuration

Base Model

- Whisper Large-v3

LoRA

- Rank = 16
- Alpha = 32
- Dropout = 0.05

Target Modules

- q_proj
- k_proj
- v_proj
- out_proj
- fc1
- fc2

Training

- Learning Rate = 2e-5
- Linear scheduler
- Warmup = 10
- No explicit weight decay

Data

- Train = 2570
- Validation = 211

Best Validation WER

## 0.2318

---

# Remaining Phase 3 Work

- Exp034 — Audio augmentation
- Exp035 — Language balancing
- Longer champion training

---

# Notes

This document is updated throughout Phase 3 and serves as the official optimization summary.
