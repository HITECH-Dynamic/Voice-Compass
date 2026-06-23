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


# Experiment 007

Status: SUCCESS

Purpose:

Generate prediction/reference pairs from the Experiment 006 model.

Source model:

outputs/exp006-whisper-small-2000examples

Output:

results/exp007_predictions.csv

Standalone script WER:

0.3401

Observation:

Prediction generation succeeded. The standalone WER differs from the W&B best WER from Experiment 006, so W&B remains the official evaluation metric. The prediction CSV will be used for qualitative error analysis.

Conclusion:

Prediction-saving infrastructure is operational.


# Experiment 008

Status: SUCCESS

Purpose:

Perform qualitative error analysis.

Major error categories:

- Phonetic substitutions
- Word splitting and merging
- Named entity instability
- Number errors
- Severe acoustic confusions

Observation:

Most outputs remain understandable despite transcription errors.

Conclusion:

Data quantity appears more valuable than additional training duration.

# Experiment 009

Status: BASELINE V1 ESTABLISHED

Model:

Whisper Small

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

2000

Validation examples:

100

Steps:

1000

Best WER:

0.2813

Hardware:

Google Colab L4

Characteristics:

- Reproducible
- Stable
- W&B integrated
- Prediction infrastructure
- Error analysis completed

Conclusion:

This model becomes the reference baseline for Phase 3 optimization.


# Experiment 010

Status: PLANNED

Purpose:

Evaluate whether a larger model improves ASR performance.

Model:

Whisper Medium

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

2000

Validation examples:

100

Steps:

1000

Reference baseline:

Whisper Small best WER = 0.2813

Question:

Does model capacity matter more than data quantity?


# Experiment 010

Status: SUCCESS

Model:

Whisper Medium

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

2000

Validation examples:

100

Steps:

1000

Runtime:

~37.7 minutes

Final train loss:

0.2962

Final eval loss:

0.4129

Best WER:

0.2059

WER progression:

100 → 0.3380
200 → 0.2995
300 → 0.2527
400 → 0.2444
500 → 0.2278
600 → 0.2070
700 → 0.2111
800 → 0.2111
900 → 0.2059
1000 → 0.2059

Observation:

Whisper Medium substantially outperformed Whisper Small.

Conclusion:

Stable Baseline v2 established.

Current champion:

Whisper Medium
2000 examples
1000 steps

Best WER = 0.2059


# Experiment 011

Status: PLANNED

Purpose:

Evaluate Whisper Medium using the full FLEURS Swahili training split.

Model:

Whisper Medium

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

3070

Validation examples:

100

Steps:

1000

Reference baseline:

Experiment 010 best WER = 0.2059

Question:

Does full-data Whisper Medium improve WER?


# Experiment 011

Status: SUCCESS

Model:

Whisper Medium

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

3070

Validation examples:

100

Steps:

1000

Final train loss:

0.3407

Final eval loss:

0.3954

Final WER:

0.2246

Best WER:

0.2002

WER progression:

100 → 0.3281
200 → 0.2839
300 → 0.2730
400 → 0.2350
500 → 0.2538
600 → 0.2262
700 → 0.2366
800 → 0.2371
900 → 0.2002
1000 → 0.2246

Observation:

Full FLEURS Swahili training improved the best WER compared with Experiment 010.

Conclusion:

Experiment 011 becomes the current champion.

Current champion:

Whisper Medium
3070 examples
1000 steps
Best WER = 0.2002


# Experiment 012

Status: PLANNED

Purpose:

Evaluate whether a lower learning rate improves Whisper Medium performance.

Model:

Whisper Medium

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

3070

Validation examples:

100

Steps:

1000

Learning rate:

5e-6

Reference baseline:

Experiment 011 best WER = 0.2002

Question:

Does lower learning rate improve WER?


# Experiment 012

Status: SUCCESS

Model:

Whisper Medium

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

3070

Validation examples:

100

Steps:

1000

Learning rate:

5e-6

Final train loss:

0.4031

Final eval loss:

0.4276

Final WER:

0.2111

Best WER:

0.2111

Observation:

Lower learning rate produced stable training but did not beat Experiment 011.

Conclusion:

Learning rate 5e-6 is worse than 1e-5 for this setup.

# Experiment 013

Status: PLANNED

Purpose:

Evaluate whether a higher learning rate improves Whisper Medium performance.

Model:

Whisper Medium

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

3070

Validation examples:

100

Steps:

1000

Learning rate:

2e-5

Reference baseline:

Experiment 011 best WER = 0.2002

Question:

Does higher learning rate improve WER?


# Experiment 013

Status: SUCCESS

Model:

Whisper Medium

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

3070

Validation examples:

100

Steps:

1000

Learning rate:

2e-5

Final train loss:

0.3214

Final eval loss:

0.3722

Final WER:

0.1893

Best WER:

0.1893

Observation:

Higher learning rate initially produced worse WER, but later converged to the best result observed so far.

Conclusion:

Learning rate 2e-5 outperformed both 1e-5 and 5e-6.

Current champion:

Experiment 013

WER = 0.1893


# Phase 3A Reproducibility

Status: STARTED

Action:

Created a separate Colab training requirements file.

File:

requirements-colab.txt

Reason:

The existing requirements.txt reflects the local development environment and does not match the Colab training environment used for the winning experiments.

Next step:

Add full seed control to the training script.


# Phase 3A Seed Control

Status: COMPLETE

Implemented:

- Python random seed
- NumPy seed
- PyTorch seed
- CUDA seed
- HuggingFace seed
- Trainer seed
- Dataset seed

Purpose:

Improve experimental reproducibility.

Motivation:

Feedback from Tiago Hersan emphasized the importance of random seeds to distinguish genuine improvements from random variation.


# Experiment 014

Status: PLANNED

Purpose:

Reproducibility test for the current champion configuration.

Model:

Whisper Medium

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

3070

Validation examples:

100

Steps:

1000

Learning rate:

2e-5

Seed:

123

Reference baseline:

Experiment 013 best WER = 0.1893

Question:

Does the champion configuration remain strong under a different random seed?


# Experiment 014

Status: SUCCESS

Purpose:

Reproducibility test for the champion configuration.

Model:

Whisper Medium

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

3070

Validation examples:

100

Steps:

1000

Learning rate:

2e-5

Seed:

123

Final train loss:

0.3294

Final eval loss:

0.3884

Final WER:

0.1979

Best WER:

0.1963

Observation:

Changing the random seed produced only a small degradation in WER.

Conclusion:

Experiment 013's result is robust and not attributable to a lucky seed.

Current standings:

Exp013 (seed 42): 0.1893

Exp014 (seed 123): 0.1963


# Experiment 015

Status: PLANNED

Purpose:

Second reproducibility test for the champion configuration.

Model:

Whisper Medium

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

3070

Validation examples:

100

Steps:

1000

Learning rate:

2e-5

Seed:

456

Reference results:

Experiment 013 seed 42 best WER = 0.1893
Experiment 014 seed 123 best WER = 0.1963

Question:

Does the champion configuration remain stable across a third random seed?


# Experiment 015

Status: SUCCESS

Purpose:

Second reproducibility test for the champion configuration.

Model:

Whisper Medium

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

3070

Validation examples:

100

Steps:

1000

Learning rate:

2e-5

Seed:

456

Final train loss:

0.3192

Final eval loss:

0.4100

Final WER:

0.2042

Best WER:

0.2042

Observation:

Changing the seed produced only modest variation.

Conclusion:

Experiment 013's champion result is reproducible.

Phase 3A Reproducibility completed.

Current standings:

Exp013 (seed 42): 0.1893

Exp014 (seed 123): 0.1963

Exp015 (seed 456): 0.2042


# Experiment 016

Status: PLANNED

Purpose:

Scaling test using Whisper Large-v3.

Model:

Whisper Large-v3

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

3070

Validation examples:

100

Steps:

1000

Learning rate:

2e-5

Seed:

42

Reference baseline:

Experiment 013 best WER = 0.1893

Question:

Does Whisper Large-v3 outperform Whisper Medium using the same recipe?


# Experiment 016

Status: INFRASTRUCTURE-LIMITED

Purpose:

Full fine-tuning scaling test with Whisper Large-v3.

Model:

Whisper Large-v3

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

3070

Validation examples:

100

Steps:

1000

Learning rate:

2e-5

Seed:

42

Outcome:

Did not complete.

Failure modes:

- FP16 batch size 4: CUDA out-of-memory.
- FP16 batch size 2 with gradient accumulation 2: FP16 gradient scaler error.
- BF16 batch size 2: CUDA out-of-memory.
- BF16 batch size 1 with gradient accumulation 4: unstable exploding loss and evaluation dtype mismatch.

Conclusion:

Whisper Large-v3 full fine-tuning is not feasible on the current L4 setup. Continue Phase 4 using LoRA or a higher-memory GPU.

Next experiment:

Experiment 017 — Whisper Large-v3 LoRA.


# Experiment 017

Status: PLANNED

Purpose:

Whisper Large-v3 LoRA scaling test.

Question:

Can LoRA make Large-v3 trainable where full fine-tuning failed?

# EXP017

Model:
Whisper Large-v3

Method:
LoRA

Parameters:
r = 16
alpha = 32
target_modules = q_proj, v_proj

Trainable parameters:
7,864,320 (0.51%)

Training:
1000 steps
1.30 epochs

Final metrics:
Eval loss = 0.5209
WER = 0.2798

Status:
✅ Benchmark
Freeze configuration.
Future experiments should branch from this baseline.
# Experiment 017

Status: SUCCESS

Model:

Whisper Large-v3

Method:

LoRA

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

3070

Validation examples:

100

Steps:

1000

Learning rate:

2e-5

Seed:

42

LoRA configuration:

- r = 16
- alpha = 32
- dropout = 0.05
- target_modules = q_proj, v_proj

Trainable parameters:

7,864,320

Trainable percentage:

0.5069%

Final train loss:

1.187

Final eval loss:

0.5209

Final WER:

0.2798

Best WER:

0.2798

Observation:

Large-v3 LoRA trained successfully on L4 and improved steadily through step 1000.

Conclusion:

LoRA makes Whisper Large-v3 feasible on the available GPU environment, but this configuration does not outperform the Whisper Medium champion.

Git tag:

exp017_whisper_large_v3_lora_r16_alpha32_qv

# Experiment 018

Status: PLANNED

Purpose:

Expand Large-v3 LoRA target modules.

Model:

Whisper Large-v3

Method:

LoRA

Target modules:

q_proj, k_proj, v_proj, out_proj

Reference baseline:

Experiment 017 best WER = 0.2798

Question:

Does broader LoRA coverage improve WER?


## Exp018

Date: 2026-06-22

Model:
Whisper Large-v3

Configuration:
- r=16
- alpha=32
- target_modules=[q_proj, k_proj, v_proj, out_proj]
- lr=2e-5
- max_steps=1000

Results:
Final WER: 0.2637
Train loss: 1.027
Epochs completed: 1.30

Findings:
Expanded qkvo target modules train stably and outperform previous configurations.
Improvement slows near epoch 1.0, suggesting additional epochs may yield smaller gains.

Decision:
Promote qkvo architecture to baseline.

Next:
Exp019 — identical configuration with extended training (~2 epochs).

cat >> experiments/experiment_log.md <<'EOF'

# Experiment 018

Status: SUCCESS

Model:

Whisper Large-v3

Method:

LoRA

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

3070

Validation examples:

100

Steps:

1000

Learning rate:

2e-5

Seed:

42

LoRA configuration:

- r = 16
- alpha = 32
- dropout = 0.05
- target_modules = q_proj, k_proj, v_proj, out_proj

Trainable parameters:

15,728,640

Trainable percentage:

1.0088%

Final eval loss:

0.4859

Final WER:

0.2637

Best WER:

0.2637

Observation:

Expanded attention adapters improved over previous LoRA configurations. Training was stable and converged smoothly.

Conclusion:

q_proj, k_proj, v_proj, and out_proj should become the new baseline architecture.

Git tag:

exp018_whisper_large_v3_lora_r16_alpha32_qkvo

# Experiment 019

Status: PLANNED

Purpose:

Determine whether longer training time improves over Experiment 018.
Only variable changed from Exp018:

MAX_STEPS increased from 1000 to 2000.

Model:

Whisper Large-v3

Method:

LoRA

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

3070

Validation examples:

100

Steps:

2000

Learning rate:

2e-5

Seed:

42

LoRA configuration:

- r = 16
- alpha = 32
- dropout = 0.05

Target modules:

q_proj, k_proj, v_proj, out_proj,

Reference baseline:

Experiment 018 best WER = 0.2637

Question:

Does expanding training time improve WER?

## Exp019 — Whisper Large V3 LoRA (qkvo) — 2000 Steps

Date: 2026-06-23

Changes:
- Same configuration as Exp018
- Increased training from 1000 → 2000 steps

Results:
- Validation WER: 0.2668

Comparison:
- Exp018 WER: 0.2637

Conclusion:
- Additional training did not improve WER.
- Slight regression observed.
- More steps alone are unlikely to be the highest-value optimization path.



# Experiment 020

Status: PLANNED

Purpose:

Test whether adding feed-forward LoRA adapters improves over the Exp018 champion.

Model:

Whisper Large-v3

Method:

LoRA

Dataset:

FLEURS Swahili (sw_ke)

Train examples:

3070

Validation examples:

100

Steps:

1000

Learning rate:

2e-5

Seed:

42

LoRA configuration:

- r = 16
- alpha = 32
- dropout = 0.05
- target_modules = q_proj, k_proj, v_proj, out_proj, fc1, fc2

Reference baseline:

Experiment 018 best WER = 0.2637

Previous test:

Experiment 019 showed that increasing steps from 1000 to 2000 did not improve WER.

Question:

Does expanding LoRA into the feed-forward layers improve WER?

Only variable changed from Exp018:

Added fc1 and fc2 to target_modules.

Success criterion:

WER < 0.2637


# Experiment 020

Status: COMPLETE

Result:

WER = 0.2278

Comparison:

Exp018 = 0.2637
Exp020 = 0.2278

Improvement:

13.6% relative WER reduction

Conclusion:

Adding fc1 and fc2 produced the strongest improvement observed so far.

New baseline architecture:

q_proj
k_proj
v_proj
out_proj
fc1
fc2

Future experiments should keep this architecture fixed while exploring other variables.

