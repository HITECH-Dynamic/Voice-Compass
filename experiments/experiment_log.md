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
