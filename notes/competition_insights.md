# Competition Insights — ASR Winning Solution Patterns

## Purpose

Track reusable lessons from strong ASR competition solutions and decide which ideas are worth testing in Voice-Compass.

---

## Recurring Patterns

### 1. Strong pretrained models matter

Top solutions commonly use large pretrained multilingual speech models, including:

- Whisper Large / Large-v3 / Large-v3 Turbo
- w2v-BERT 2.0
- XLS-R / wav2vec-style encoders

Voice-Compass decision:

Continue with Whisper first because the pipeline is already working. Compare other architectures only after the Whisper training and evaluation loop is stable.

---

### 2. Validation strategy is critical

Strong teams often create their own validation split rather than relying only on the provided dev set.

Useful validation principles:

- speaker independence
- age balance
- gender balance
- language balance
- official dev set reserved as a secondary unseen check

Voice-Compass decision:

Maintain larger, more representative validation splits and evaluate per language.

---

### 3. Hyperparameters are dataset-specific

Winning solutions emphasize tuning rather than copying defaults.

Important knobs:

- learning rate
- effective batch size
- warmup
- scheduler
- dropout
- weight decay
- training length

Voice-Compass decision:

Continue one-variable-at-a-time experiments and avoid assuming another competition's hyperparameters will transfer directly.

---

### 4. Dropout deserves a dedicated study

Multiple strong ASR solutions found gains from tuned dropout.

Candidate values to study later:

- attention dropout: 0.05
- hidden dropout: 0.05
- feature projection dropout: 0.05
- layerdrop: 0.025
- higher dropout around 0.15 for some Whisper configurations

Voice-Compass decision:

Add dropout as a dedicated experiment after the multilingual training/evaluation loop is stable.

---

### 5. Augmentation is not guaranteed

Some top solutions used no augmentation. Others used heavy augmentation.

Common augmentations:

- time stretch
- pitch shift
- volume gain
- Gaussian noise
- background noise

Voice-Compass decision:

Treat augmentation as an experiment, not an assumption.

---

### 6. Inference can improve scores

Useful inference-time strategies:

- beam search
- larger beam sizes
- 2-gram KenLM decoding for CTC models
- CTranslate2 conversion for faster Whisper inference

Voice-Compass decision:

After WER/CER evaluation works, test beam search before more expensive model changes. Keep CTranslate2 for edge deployment work.

---

### 7. Full fine-tuning vs LoRA

Full fine-tuning can be stronger when compute is abundant.

LoRA remains appropriate when:

- memory is limited
- iteration speed matters
- edge deployment matters
- rapid experimentation is needed

Voice-Compass decision:

Continue LoRA for near-term experiments. Consider full fine-tuning only if compute and time allow.

---

## Current Strategic Takeaway

The project should not pivot architectures yet.

Current priority:

1. Finish usable AfriVoices indexed data pipeline.
2. Run multilingual smoke training.
3. Add WER/CER.
4. Establish first meaningful multilingual baseline.
5. Then test inference improvements and regularization.

