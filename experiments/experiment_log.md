# Experiment 001

Date: June 21, 2026

# Purpose

Establish a baseline training pipeline using Whisper Small and Hugging Face Trainer.

# Configuration

Model:
openai/whisper-small

Trainer:
Hugging Face Trainer

Experiment Tracking:
Weights & Biases

Training Length:
50 steps

# Results

-

# Observations

-

# Next Steps

-
# Experiment 001

Date: June 21, 2026

Purpose:

Establish a baseline training pipeline using Whisper Small and Hugging Face Trainer.

Configuration:

- Model: openai/whisper-small
- Trainer: Hugging Face Trainer
- Experiment tracking: Weights & Biases
- Training length: 50 steps
- LoRA: No
- Augmentation: No
- Language weighting: No

## Results

- Experiment 001 completed successfully on Kaggle GPU.
- Device: Tesla T4 / CUDA.
- Training steps: 50.
- Runtime: 154.2 seconds.
- Final train loss: 2.26689.
- Last logged loss: 1.21388.
- Model saved to: outputs/exp001-smoke-test.
- W&B run synced successfully.

## Observations

- Loss decreased steadily, confirming the training loop works.
- W&B captured loss, learning rate, grad norm, epoch, and runtime.
- FLEURS Swahili loads correctly using config sw_ke.
- Kaggle GPU is now functional after account verification.


Experiment 001
---------------
Model: Whisper Small
Dataset: FLEURS sw_ke
Train examples: 100
Eval examples: 50
Steps: 50
Runtime: 154 sec
Final train loss: 2.26689
Last logged loss: 1.21388
Platform: Kaggle Tesla T4
Result: SUCCESS

Observation:
Loss decreased steadily and checkpointing worked.


Next Steps:

## Experiment 002

Model: Whisper Small

Dataset: FLEURS Swahili (sw_ke)

Train examples: 100

Validation examples: 50

Steps: 500

Hardware: Tesla T4 (Kaggle)

Runtime: 1552 sec (~25.9 min)

Final train loss: 0.269

Last logged loss: 0.000879

Result: SUCCESS

Key observation:

The model memorized the dataset. Increasing training duration is no longer the limiting factor. Additional data should be explored before introducing more complexity.

# Experiment 003

Model: Whisper Small

Dataset: FLEURS Swahili (sw_ke)

Train examples: 1000

Validation examples: 100

Steps: 500

Hardware: Tesla T4

Runtime: ~1610 sec

Final train loss: 0.839

Last logged loss: 0.231

Result: SUCCESS

Observation:

Increasing dataset size from 100 to 1000 examples prevented severe overfitting and produced much more stable learning.

Conclusion:

Data quantity is more important than additional training duration.


# Experiment 004

Status: PLANNED

Purpose:

Add Word Error Rate (WER) evaluation.

Only change relative to Experiment 003:

- Introduce WER metric.

Question:

How good are the transcriptions?


# Experiment 004

Status: RUNNING

Purpose:

Add Word Error Rate evaluation.

Only change relative to Experiment 003:

- WER metric

Question:

How good are the transcriptions?


# Experiment 004

Status: RUNNING

Purpose:

Add Word Error Rate evaluation.

Only change relative to Experiment 003:

- WER metric

Question:

How good are the transcriptions?


# Experiment 004

Model: Whisper Small

Dataset: FLEURS Swahili (sw_ke)

Train examples: 1000

Validation examples: 100

Steps: 500

Hardware: Tesla T4

Runtime: ~1891 sec

Final train loss: 0.8395

Final eval loss: 0.6423

Final WER: 0.3250

Result: SUCCESS

WER progression:

100 steps → 0.4077
200 steps → 0.3385
300 steps → 0.3380
400 steps → 0.3307
500 steps → 0.3250

Observation:

WER improved steadily throughout training.

Conclusion:

Phase 2 evaluation infrastructure is operational and WER is now the primary metric for future experiments.


# Experiment 005

Status: PLANNED

Purpose:

Evaluate whether additional training beyond 500 steps improves WER.

Only change:

MAX_STEPS

500 → 1000

Question:

Does WER continue improving?


# Experiment 005

Status: PARTIAL SUCCESS

Model: Whisper Small

Dataset: FLEURS Swahili (sw_ke)

Train examples: 1000

Validation examples: 100

Target steps: 1000

Best WER observed:

0.3094

Observation:

Increasing training duration beyond 500 steps continued improving WER, but with diminishing returns.

Training terminated near step 800 because the Kaggle session ran out of storage while writing checkpoints.

Conclusion:

Longer training helps, but the gains are smaller. Future runs should limit checkpoint accumulation.


# Experiment 006

Status: PLANNED

Purpose:

Test whether increasing training data improves WER.

Only major modeling change relative to Experiment 005:

TRAIN_SAMPLES

1000 → 2000

Configuration:

- Model: Whisper Small
- Dataset: FLEURS Swahili (sw_ke)
- Train examples: 2000
- Validation examples: 100
- Target steps: 1000
- WER evaluation: enabled

Infrastructure change:

- save_steps = 500
- save_total_limit = 2

Question:

Does more data reduce WER?

Baseline to beat:

Experiment 005 best WER = 0.3094


# Experiment 006

Status: SUCCESS

Model: Whisper Small

Dataset: FLEURS Swahili (sw_ke)

Train examples: 2000

Validation examples: 100

Steps: 1000

Hardware: NVIDIA L4 (Google Colab)

Best WER observed:

0.2813

Final WER:

0.3016

Observation:

Doubling the amount of training data from 1000 to 2000 examples produced a clear improvement over Experiment 005.

Performance peaked around step 900.

Conclusion:

Increasing data quantity continues to provide meaningful gains and appears more important than simply extending training duration.


# Experiment 007

Status: INFRASTRUCTURE COMPLETE

Purpose:

Build prediction-saving infrastructure.

Components:

- scripts/save_predictions.py

Outputs:

- exp007_predictions.csv

Observation:

Prediction artifacts are intentionally excluded from Git via .gitignore.

Next step:

Generate predictions using the Experiment 006 model and perform qualitative error analysis.

