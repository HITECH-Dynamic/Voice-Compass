# Exp063 — Extended Multilingual Training

**Date:** 2026-07-13  
**Day:** 26  
**Status:** Prepared

## Objective

Reduce multilingual WER by extending the validated Exp062 training configuration from 1,000 to 2,000 steps.

## Hypothesis

Exp062 improved substantially over Exp061 when training increased from 500 to 1,000 steps. Extending training to 2,000 steps may produce additional multilingual adaptation while best-checkpoint selection protects against late-training regression.

## Baselines

### Exp062 local evaluation

- Global WER: 0.9727
- Kaggle-average validation WER: 1.0508

### Exp062 public leaderboard

- Kaggle WER: 1.42222

## Controlled Configuration

Held constant:

- Base model: openai/whisper-small
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: q_proj, v_proj
- Learning rate: 2e-5
- Train rows: 3,000
- Validated evaluation rows: 599
- Best checkpoint metric: Kaggle-average WER
- Clean Exp061 train/eval manifests

Primary variable:

- max_steps: 1,000 → 2,000

## Success Criteria

- Primary: public Kaggle WER below 1.42222
- Strong result: public Kaggle WER below 1.30
- Stretch result: public Kaggle WER below 1.20

## Parallel Inference Work

Prepare a separate inference-only submission experiment addressing:

- language-aware decoding where Whisper supports the language token;
- runaway repetition;
- excessively long hallucinated transcripts;
- generation settings evaluated separately from the training change.
