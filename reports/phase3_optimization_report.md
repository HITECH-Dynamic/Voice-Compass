# Phase 3 — Optimization Report

## Project

AfriVoices ASR Hackathon

## Objective

Identify an effective Whisper Large-v3 LoRA training configuration using controlled, one-variable-at-a-time experimentation before training on the AfriVoices competition dataset.

---

# Methodology

Each experiment modified exactly one variable while keeping all remaining training parameters fixed.

Performance was evaluated using validation Word Error Rate (WER).

This methodology allowed each observed change in performance to be attributed to a single experimental factor.

---

# Variables Studied

## Validation Split

Result:

Larger validation split produced more reliable evaluation.

Decision:

Adopt approximately an 80/20 train/validation split.

---

## LoRA Rank

Result:

Rank 16 consistently performed well.

Decision:

Rank = 16

---

## LoRA Alpha

Result:

Higher alpha values did not improve WER.

Decision:

Alpha = 32

---

## Target Modules

Result:

Applying LoRA to both attention and feed-forward layers improved adaptation.

Decision:

Use:

* q_proj
* k_proj
* v_proj
* out_proj
* fc1
* fc2

---

## Learning Rate

Result:

2e-5 produced the most stable optimization.

Decision:

Learning rate = 2e-5

---

## Learning Rate Scheduler

Experiments:

* Linear
* Cosine
* constant_with_warmup

Result:

Linear scheduler consistently produced the strongest validation performance.

Decision:

Use the default linear scheduler.

---

## Warmup

Experiments:

* 10
* 50

Result:

Increasing warmup did not improve WER.

Decision:

Warmup = 10

---

## Weight Decay

Experiments:

* None
* 0.001
* 0.05

Result:

No measurable improvement.

Decision:

No explicit weight decay.

---

## LoRA Dropout

Experiments:

* 0.05
* 0.10

Result:

No measurable improvement.

Decision:

LoRA Dropout = 0.05

---

## Audio Augmentation

Experiments:

Training-only speed and gain augmentation.

Result:

Validation WER increased.

Decision:

Do not use augmentation in the current competition configuration.

---

## Effective Batch Size

Experiments:

Effective batch size:

4

8

Result:

Performance remained close to the champion but did not improve upon it.

Decision:

Retain effective batch size = 4.

---

# Champion Configuration

Base Model

Whisper Large-v3

LoRA

* Rank = 16
* Alpha = 32
* Dropout = 0.05

Target Modules

* q_proj
* k_proj
* v_proj
* out_proj
* fc1
* fc2

Training

* Learning rate = 2e-5
* Linear scheduler
* Warmup = 10
* Effective batch size = 4
* No explicit weight decay

Dataset

* FLEURS Swahili
* Train = 2570
* Validation = 211

Best Validation WER

**0.2318**

---

# Key Findings

Several optimizer parameters significantly influenced performance, including learning rate, validation strategy, and LoRA configuration.

Other parameters—including warmup, weight decay, dropout, augmentation, and effective batch size—produced little or no measurable improvement.

This suggests that the current training recipe is approaching a local optimum on the FLEURS benchmark.

---

# Conclusion

Phase 3 successfully established a stable and reproducible Whisper Large-v3 LoRA training configuration.

This optimized pipeline will serve as the baseline for Phase 4, where experimentation transitions from trainer optimization to data-centric improvements using the AfriVoices competition dataset.
