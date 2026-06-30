# Champion Timeline

| Date | Experiment | WER | Change | Status |
|------|------------|----:|--------|--------|
| Day 2 | Baseline | TBD | Initial Whisper Large-v3 baseline | Champion |
| Day 8 | Exp026 | **0.2318** | Improved validation split + optimized LoRA configuration | 🏆 Current Champion |

---



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


# Experiment 021

Status: PLANNED

Purpose:

Evaluate whether earlier stopping improves WER.

Model:

Whisper Large-v3

Method:

LoRA

Dataset:

FLEURS Swahili (sw_ke)

Configuration:

- r = 16
- alpha = 32
- target_modules = q_proj, k_proj, v_proj, out_proj, fc1, fc2

Steps:

800

Learning rate:

2e-5

Reference baseline:

Exp020 best WER = 0.2278

Only variable changed:

MAX_STEPS 1000 → 800

Question:

Can earlier stopping outperform Exp020?

Success criterion:

WER < 0.2278


# Experiment 021

Status: COMPLETE

Purpose:

Evaluate whether earlier stopping improves WER.

Configuration:

Same as Exp020 except MAX_STEPS changed from 1000 to 800.

Result:

WER = 0.2418

Reference baseline:

Exp020 best WER = 0.2278

Conclusion:

Earlier stopping did not improve performance. Exp021 is not promoted.

Current champion remains:

Exp020

# Experiment 022

Status: PLANNED

Purpose:

Test whether increasing LoRA rank improves over the Exp020 champion.

Model:

Whisper Large-v3

Method:

LoRA

Dataset:

FLEURS Swahili (sw_ke)

Configuration:

- r = 32
- alpha = 32
- target_modules = q_proj, k_proj, v_proj, out_proj, fc1, fc2
- learning_rate = 2e-5
- max_steps = 1000
- seed = 42

Reference baseline:

Exp020 best WER = 0.2278

Only variable changed from Exp020:

LoRA rank increased from 16 to 32.

Question:

Does increasing LoRA rank improve WER?

Success criterion:

WER < 0.2278


# Experiment 022

Status: COMPLETE

Purpose:

Test whether increasing LoRA rank improves over the Exp020 champion.

Configuration:

Same as Exp020 except:

- r = 32
- alpha = 32

Result:

Best WER = 0.2387
Final WER = 0.2397

Reference baseline:

Exp020 best WER = 0.2278

Conclusion:

Increasing LoRA rank from 16 to 32 did not improve WER. Exp022 is not promoted.

Current champion remains:

Exp020

Next candidate:

Exp023 alpha study.


# Experiment 023

Status: PLANNED

Purpose:

Evaluate the effect of increasing LoRA alpha while keeping the Exp020 champion architecture fixed.

Model:

Whisper Large-v3

Method:

LoRA

Dataset:

FLEURS Swahili (sw_ke)

Configuration:

- r = 16
- alpha = 64
- target_modules = q_proj, k_proj, v_proj, out_proj, fc1, fc2
- learning_rate = 2e-5
- max_steps = 1000
- seed = 42

Reference baseline:

Exp020 best WER = 0.2278

Only variable changed:

LoRA alpha increased from 32 to 64.

Question:

Does increasing LoRA alpha improve WER?

Success criterion:

WER < 0.2278


# Experiment 023

Status: COMPLETE

Purpose:

Test whether increasing LoRA alpha improves over the Exp020 champion.

Configuration:

Same as Exp020 except:

- r = 16
- alpha = 64

Result:

Best WER = 0.2319
Final WER = 0.2376

Reference baseline:

Exp020 best WER = 0.2278

Conclusion:

Increasing LoRA alpha from 32 to 64 did not improve WER. Exp023 is not promoted.

Current champion remains:

Exp020

Next direction:

Begin multilingual sampling experiments using the Exp020 champion configuration.


# Experiment 024

Status: PLANNED

Purpose:

Test whether lowering the learning rate improves over the Exp020 champion.

Model:

Whisper Large-v3

Method:

LoRA

Dataset:

FLEURS Swahili (sw_ke)

Configuration:

- r = 16
- alpha = 32
- target_modules = q_proj, k_proj, v_proj, out_proj, fc1, fc2
- learning_rate = 1e-5
- max_steps = 1000
- seed = 42

Reference baseline:

Exp020 best WER = 0.2278

Only variable changed:

Learning rate decreased from 2e-5 to 1e-5.

Question:

Does a lower learning rate improve WER?

Success criterion:

WER < 0.2278


# Experiment 024

Status: COMPLETE

Purpose:

Test whether lowering the learning rate improves over the Exp020 champion.

Configuration:

Same as Exp020 except:

- learning_rate = 1e-5

Result:

Best WER = 0.2595
Final WER = 0.2600

Reference baseline:

Exp020 best WER = 0.2278

Conclusion:

Lowering the learning rate from 2e-5 to 1e-5 did not improve WER. Exp024 is not promoted.

Current champion remains:

Exp020

# Experiment 025

Status: PLANNED

Purpose:

Test whether a slightly higher learning rate improves over the Exp020 champion.

Model:

Whisper Large-v3

Method:

LoRA

Dataset:

FLEURS Swahili (sw_ke)

Configuration:

- r = 16
- alpha = 32
- target_modules = q_proj, k_proj, v_proj, out_proj, fc1, fc2
- learning_rate = 3e-5
- max_steps = 1000
- seed = 42

Reference baseline:

Exp020 best WER = 0.2278

Only variable changed:

Learning rate increased from 2e-5 to 3e-5.

Question:

Does a slightly higher learning rate improve WER?

Success criterion:

WER < 0.2278


# Experiment 025

Status: COMPLETE

Purpose:

Test whether a slightly higher learning rate improves over the Exp020 champion.

Configuration:

Same as Exp020 except:

- learning_rate = 3e-5

Result:

Best WER = 0.2257
Final WER = 0.2371

Reference baseline:

Exp020 best WER = 0.2278

Conclusion:

Learning rate 3e-5 reached the lowest observed WER so far but did not maintain it through the end of training. Exp025 is not promoted as the stable champion because the final model underperformed Exp020.

Current stable champion remains:

Exp020

Important finding:

Exp025 suggests that a higher learning rate may be useful if paired with early stopping, best-checkpoint saving, or scheduler changes.

Next direction:

Learning-rate scheduler / checkpoint-selection study.


# Experiment 026

Status: PLANNED

Purpose:

Re-run the Exp020 champion configuration with a full available validation split based on Tiago's feedback.

Model:

Whisper Large-v3

Method:

LoRA

Dataset:

FLEURS Swahili (sw_ke)

Configuration:

- r = 16
- alpha = 32
- target_modules = q_proj, k_proj, v_proj, out_proj, fc1, fc2
- learning_rate = 2e-5
- max_steps = 1000
- seed = 42

Split:

- train_samples = 2570
- eval_samples = 211

Only variable changed from Exp020:

Validation split increased from 100 to the full available 211 validation examples.

Question:

Does the champion configuration remain reliable under a full available validation set?

Note:

Exp026 should become the new evaluation baseline for future optimization experiments.


# Experiment 026

Status: COMPLETE

Purpose:

Re-run the Exp020 champion configuration with the full available FLEURS Swahili validation split.

Result:

- Final WER = 0.2318
- Final eval loss = 0.4474
- Train samples = 2570
- Validation samples = 211
- Runtime = approximately 3h 7m

Conclusion:

The champion configuration remained strong under a larger validation split. Exp026 becomes the new trustworthy evaluation baseline.

# Experiment 027

Status: PLANNED

Purpose:

Test Tiago's suggestion to explicitly set LoRA task_type to SEQ_2_SEQ_LM.

Configuration:

Same as Exp026 except:

- task_type = TaskType.SEQ_2_SEQ_LM

Reference baseline:

Exp026 final WER = 0.2318

Question:

Does explicitly setting the PEFT LoRA task type improve Whisper seq2seq ASR fine-tuning?

Success criterion:

WER < 0.2318


# Experiment 027

Status: FAILED

Purpose:

Test Tiago's suggestion to explicitly set LoRA task_type to SEQ_2_SEQ_LM.

Result:

Failed before training began.

Error:

got multiple values for keyword argument 'input_ids'

Conclusion:

task_type=TaskType.SEQ_2_SEQ_LM is incompatible with the current PEFT + Whisper setup. Do not promote.

Decision:

Return to Exp026 baseline.

"So Exp027 likely failed because PeftModelForSeq2SeqLM assumes a text seq2seq interface and passes input_ids, while Whisper training uses input_features for audio. There is also a PEFT GitHub issue specifically about Whisper and input_ids, which strongly supports this being a known PEFT/Whisper compatibility pattern.

Decision: keep Exp027 closed as failed/incompatible. Do not spend compute retrying it during Phase 3. We can add a note later that the failure is likely due to PEFT’s SEQ_2_SEQ_LM wrapper being designed around text seq2seq models, not Whisper’s audio encoder interface."


# Experiment 028

Status: PLANNED

Purpose:

Evaluate whether automatic best-checkpoint selection improves the final model.

Model:

Whisper Large-v3

Method:

LoRA

Configuration:

Same as Exp026 except:

- save_strategy = "steps"
- save_steps = 100
- evaluation_strategy = "steps"
- eval_steps = 100
- load_best_model_at_end = True
- metric_for_best_model = "wer"
- greater_is_better = False

Reference baseline:

Exp026 final WER = 0.2318

Only variable changed:

Best checkpoint selection.

Question:

Does restoring the best validation checkpoint improve the final WER?

Success criterion:

WER < 0.2318


# Experiment 028 Revised

Status: PLANNED

Purpose:

Test an explicit cosine learning-rate scheduler.

Configuration:

Same as Exp026 except:

- lr_scheduler_type = "cosine"

Infrastructure note:

Best-checkpoint saving is now part of the standard training pipeline and is not treated as the experimental variable.

Reference baseline:

Exp026 WER = 0.2318

Question:

Does cosine scheduling improve WER versus the previous default scheduler behavior?

Success criterion:

WER < 0.2318


## Exp028 — Whisper Large-v3 LoRA + Cosine Scheduler

**Date:** 2026-06-26

### Purpose
Evaluate whether replacing the default linear learning-rate scheduler with a cosine scheduler improves validation WER.

### Variable Changed
- `lr_scheduler_type`
  - Linear → Cosine

### Constant Parameters
- Whisper Large-v3
- LoRA r=16
- LoRA alpha=32
- LoRA targets: q_proj, k_proj, v_proj, out_proj, fc1, fc2
- Learning rate: 2e-5
- Train/Validation split: 2570 / 211
- Batch size unchanged
- Max steps: 1000

### Results
- Validation WER: **0.2505**

### Comparison
Exp026 (Linear Scheduler): **0.2318**
Exp028 (Cosine Scheduler): **0.2505**

Difference:
+0.0187 WER (≈8% relative degradation)

### Conclusion
The cosine scheduler reduced performance under the current training configuration. The default linear scheduler remains the preferred scheduler for future experiments.

| Component           | Champion                                   |
| ------------------- | ------------------------------------------ |
| Base model          | Whisper Large-v3                           |
| Validation split    | 2570 / 211                                 |
| LoRA targets        | q_proj, k_proj, v_proj, out_proj, fc1, fc2 |
| Rank                | 16                                         |
| Alpha               | 32                                         |
| Learning rate       | 2e-5                                       |
| Scheduler           | Linear                                     |
| Warmup              | 10 *(until Exp029)*                        |
| Weight decay        | TBD                                        |
| LoRA dropout        | 0.05 *(until Exp031)*                      |
| Task type           | Default                                    |
| Best validation WER | **0.2318 (Exp026)**                        |



# Experiment 029

Status: PLANNED

Purpose

Determine whether increasing warmup from 10 to 50 steps improves convergence.

Reference

Exp026

Variable Changed

warmup_steps

10

↓

50

Everything Else

Unchanged.

Success Criterion

WER < 0.2318


# Experiment 029

Status: COMPLETE

Result:

- Best observed WER = 0.2359
- Final WER = 0.2460
- Final eval loss = 0.4456

Conclusion:

warmup_steps=50 did not improve over Exp026. Reject.

Decision:

Restore warmup_steps=10.

# Experiment 030

Status: PLANNED

Purpose:

Test whether weight decay improves generalization.

Reference baseline:

Exp026 WER = 0.2318

Only variable changed:

- weight_decay = 0.01

Success criterion:

WER < 0.2318


# Experiment 030

Status: COMPLETE

Purpose:

Test whether weight_decay=0.01 improves generalization.

Result:

- Final WER = 0.2318
- Final eval loss = 0.4474

Conclusion:

weight_decay=0.01 matched Exp026 but did not clearly improve it.

Decision:

Do not promote yet.

# Experiment 031

Status: PLANNED

Purpose:

Test lighter weight decay.

Reference baseline:

Exp026 WER = 0.2318

Only variable changed:

- weight_decay = 0.001

Success criterion:

WER < 0.2318

## Exp031 — Weight Decay = 0.001

**Objective**
Evaluate whether a smaller weight decay (0.001) improves Whisper Small fine-tuning performance compared to the baseline and Exp030.

**Configuration**
- Based on: Exp026
- weight_decay = 0.001
- All other parameters unchanged

**Results**
- Eval Loss: 0.44737
- WER: 0.2318

**Outcome**
Matched the baseline and Exp030 exactly. No measurable improvement observed.

**Decision**
Weight decay is not currently a productive optimization direction. Restore the baseline configuration (no explicit weight decay) and move to the next optimization study.
# Experiment 032

Status: PLANNED

Purpose:

Test constant_with_warmup scheduler.

Reference baseline:

Exp026 / Exp030 / Exp031 WER = 0.2318

Only variable changed:

- lr_scheduler_type = "constant_with_warmup"

Important reset:

- Remove explicit weight_decay from Exp031.

Success criterion:

WER < 0.2318


# Experiment 032

Status: COMPLETE

Purpose:

Test constant_with_warmup scheduler.

Result:

- Best usable WER = 0.2419
- Final WER = 4.0899
- Final eval loss = 0.38857

Conclusion:

constant_with_warmup was rejected. WER diverged late despite decreasing eval loss.

Decision:

Restore default linear scheduler.

# Experiment 033

Status: PLANNED

Purpose:

Test whether stronger LoRA dropout improves generalization.

Reference baseline:

Exp026 WER = 0.2318

Only variable changed:

- lora_dropout = 0.10

Success criterion:

WER < 0.2318


## Exp033 — Whisper Large-v3 LoRA Dropout 0.10

**Date:** 2026-06-27

### Objective
Evaluate whether increasing LoRA dropout from 0.05 to 0.10 improves generalization and lowers validation WER.

### Configuration
- Base model: Whisper Large-v3
- LoRA rank: 128
- LoRA alpha: 64
- LoRA dropout: **0.10**
- Learning rate: 2e-5
- Epochs: 1.56
- Train/Validation split: 2570 / 211

### Results
- Best Validation WER: **0.2325**

### Comparison
- Exp026 (dropout 0.05): **0.2325**
- Exp033 (dropout 0.10): **0.2325**

Difference: **0.0000**

### Conclusion
Increasing LoRA dropout from 0.05 to 0.10 produced no measurable improvement. The model does not appear to be overfitting enough for additional LoRA regularization to be beneficial.

### Decision
Close the LoRA dropout study.

Future experiments will continue using:

- LoRA dropout = **0.05**
# Experiment 034

Status: PLANNED

Purpose:

Test training-only audio augmentation.

Hypothesis:

Light speed and gain augmentation will improve acoustic robustness and reduce WER.

Reference baseline:

Exp026 WER = 0.2318

Only variable changed:

- Training audio augmentation enabled

Augmentation:

- Speed perturbation: 0.9x, 1.0x, 1.1x
- Gain perturbation: -3 dB to +3 dB

Validation:

- No augmentation applied to validation set.

Success criterion:

WER < 0.2318


## Exp034 — Audio Augmentation (Speed + Gain)

Date: 2026-06-28

### Objective
Evaluate whether simple training-time audio augmentation improves Whisper Large-v3 LoRA performance.

### Configuration
- Base model: Whisper Large-v3
- LoRA: q_proj, k_proj, v_proj, out_proj, fc1, fc2
- Dataset: FLEURS Swahili
- Train/Validation: 2570 / 211 (80/20)
- Learning rate: 2e-5
- Batch size: unchanged
- Epochs: unchanged
- Audio augmentation:
  - Random speed perturbation
  - Random gain adjustment
- Validation data left untouched

### Result
Validation WER: 0.2484

### Comparison
Exp026 (baseline): 0.2318

Difference:
+0.0166 WER
≈7.2% relative degradation

### Conclusion
Training-only audio augmentation did not improve recognition performance on the current validation set.

For the present training recipe, Whisper Large-v3 appears to generalize better without these augmentations.

### Decision
Close augmentation study.

Exp026 remains the project champion.


## Exp035 — Effective Batch Size Study

Status: PLANNED

Reference:
Exp026 champion

Hypothesis:
Increasing the effective batch size from 4 to 8 will improve optimization stability and reduce WER.

Variable Changed:
gradient_accumulation_steps

2 → 4

Effective batch:
4 → 8

Success Criterion:
WER < 0.2318


## Experiment 035 — Effective Batch Size Study

**Status:** COMPLETE

**Date:** Day 11 (2026-06-28)

### Objective

Evaluate whether increasing the effective batch size improves Whisper Large-v3 LoRA performance.

### Hypothesis

Increasing the effective batch size from 4 to 8 will provide smoother optimization and reduce validation WER.

### Configuration

**Reference Baseline**

* Exp026 (Champion)

**Variable Changed**

* Gradient Accumulation Steps:

  * 2 → 4

Effective Batch Size:

* 4 → 8

Everything else remained identical to the Exp026 champion.

### Results

Validation WER:

**0.2371**

Reference:

Exp026 = **0.2318**

Difference:

+0.0053 WER

### Interpretation

Increasing the effective batch size produced stable training but did not improve recognition accuracy over the current champion.

Training converged normally and remained competitive, suggesting that effective batch size is not a major performance driver for the current dataset.

### Conclusion

Hypothesis not supported.

### Decision

Close the effective batch size study.

Retain Exp026 as the current champion configuration.

# | Exp036A | 
AfriVoices Dataset Profiling | Whisper Large-v3 (N/A) | Dataset Engineering | Completed | Successfully profiled the full AfriVoices competition corpus (1.08M utterances, 5,542.97 hours, 6 languages, 7 source repositories). Identified significant language imbalance and validated dataset accessibility. Established reusable profiling pipeline to support manifest generation and future experiments. |

# | Exp036B | 
Unified AfriVoices Manifest Generation | N/A | Dataset Engineering | Planned | Build normalized train/dev/test manifest parquet files from all AfriVoices source repositories. This is the final data-engineering step before Exp037 competition baseline training. |

# | Exp036B | 
Unified AfriVoices Manifest Generation | N/A | Dataset Engineering | Completed | Generated train/dev/test manifests across all six AfriVoices languages. Added row-level unique IDs. Train/dev/test rows: 1,015,803 / 35,578 / 30,618. |

| Exp037 | First AfriVoices Real-Data Training Run | Whisper Large-v3 + LoRA | Unified Manifest / Balanced Subset | Planned | First real competition-data training run. Use Exp026 champion settings with filtered balanced AfriVoices subset. Primary goal is end-to-end validation of manifest loading, audio resolution, training, validation, and checkpointing. |

| Exp037 | AfriVoices Audio Resolver Test | N/A | Audio Resolution | Completed | Successfully resolved and decoded both ANV-KE parquet_bytes audio and Swahili tar.xz/WebM audio from unified manifests. Confirmed 16 kHz waveform output. Identified need to optimize parquet shard lookup before large-scale training. |

| Exp038A | AfriVoices Dataset Class Test | N/A | Dataset Loader | Planned | Build a dataset-style loader that reads unified manifests, filters bad rows, resolves audio, and returns decoded waveform/transcript examples before Whisper processor integration. |

| Exp038A | AfriVoices Dataset Class Test | N/A | Dataset Loader | Completed | Built reusable AfriVoicesDataset class and validated manifest-to-waveform loading for both ANV Parquet audio and Swahili tar.xz/WebM audio. |

| Exp038B | Whisper Processor Dataset Integration | Whisper Large-v3 Processor | Dataset Loader | Planned | Extend AfriVoicesDataset so decoded audio examples are converted into Whisper input_features and tokenized labels. No training yet. |

| Exp038B | Whisper Processor Integration | Whisper Large-v3 | Dataset Pipeline | Completed | Connected AfriVoicesDataset to WhisperProcessor. Verified end-to-end conversion from manifest to decoded waveform, Whisper input_features, and tokenized labels for both ANV and Swahili datasets. |

| Exp039 | First End-to-End Whisper Training | Whisper Large-v3 + LoRA | Real AfriVoices Smoke Training | Planned | Connect AfriVoicesWhisperDataset to a data collator, Whisper model, Trainer, validation metrics, and checkpoint saving. First goal is to prove the full training loop on a small balanced real-data subset. |

| Exp039 | First End-to-End Whisper Training | Whisper Large-v3 + LoRA | Smoke Training | Completed | Successfully trained and evaluated Whisper Large-v3 with LoRA on real AfriVoices data. Verified complete pipeline from manifest through audio decoding, feature extraction, Trainer, evaluation, and checkpoint generation. Train loss: 1.033, Eval loss: 1.8156. |

| Exp040A | ANV Audio Index Builder | N/A | Dataset Optimization | Planned | Build filename-to-parquet-shard lookup index for ANV audio so training no longer scans many large Parquet files per example. |

| Exp040B | Indexed ANV Resolver | N/A | Dataset Optimization | Planned | Update AfriVoicesDataset to use prebuilt ANV audio index for direct filename-to-shard lookup, with fallback to original scanning behavior. |

| Exp040A | Bounded ANV Audio Index Builder | N/A | Dataset Optimization | Completed | Revised from full-corpus indexing to bounded smoke index after full indexing proved too expensive. Built Kikuyu train/scripted index over 2 shards with 2,550 unique audio filenames. |
| Exp040B | Indexed ANV Resolver | N/A | Dataset Optimization | Completed | Updated AfriVoicesDataset to use optional ANV audio index for direct filename-to-Parquet lookup. Successfully resolved a Kikuyu example directly from indexed shard train_scripted_001.parquet. |

| Exp041 | Full ANV Audio Index | N/A | Dataset Optimization | Planned | Build a complete filename-to-Parquet index for every ANV language and shard. This becomes the production lookup table used during multilingual training. |


| Exp041A | ANV Index Strategy Inspection | N/A | Dataset Optimization | Planned | Inspect whether ANV audio filename-to-shard mapping can be derived without downloading every large audio Parquet shard. |
| Exp041B | Incremental ANV Index Builder | N/A | Dataset Optimization | Planned | Build ANV audio indexes incrementally by language/split/speech type to avoid long fragile full-corpus runs. |
| Exp041C | ANV Index Coverage Validation | N/A | Dataset Optimization | Planned | Validate indexed audio coverage against the unified AfriVoices manifests before using the index for training. |

| Exp041A | ANV Dataset Index Strategy Inspection | N/A | Dataset Optimization | Completed | Determined that transcript metadata does not contain a deterministic filename-to-Parquet mapping. Confirmed that the correct architecture is a persistent global filename → Parquet index for future preprocessing and training. |

