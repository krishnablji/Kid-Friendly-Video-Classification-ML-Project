# Kid-Friendly Video Classification ML Project

A modern, production-grade Child-Safety Video Guardrail pipeline designed to determine whether video content is **`SAFE`** (Kid-Friendly) or **`UNSAFE`** (Not Kid-Friendly) through multi-stage perception and fine-tuned Transformer classification.

---

## Architecture Overview: 4-Stage Content Moderation Pipeline

```
========================================================================================
                         STAGE 1: VIDEO INGESTION & SAMPLING
========================================================================================
                                   [ Raw Video Upload (.mp4) ]
                                                │
                                                ▼
                          [ OpenCV / FFmpeg Preprocessing Engine ]
                          • Frame Downsampling: 1 Frame Per Second (FPS)
                          • Video Format Normalization & Validation
                                                │
                                                ▼
========================================================================================
                 STAGE 2: MULTIMODAL PERCEPTION API ENGINE ("EYES")
========================================================================================
                               [ Google Gemini 1.5/2.5 API ]
                                                │
                                                ▼
                         [ Neutral Chronological Prompting Strategy ]
                         • Task: Generate dense, timestamped visual narratives
                         • Instruction: Describe actions, movements, objects, and
                           environmental events objectively without inferring intent
                                                │
                                                ▼
                   [ Scene Narrative Output: e.g., "[00:04 - 00:08]: ..."]
                                                │
                                                ▼
========================================================================================
    STAGE 3: LOCAL ML TEXT CLASSIFIER - NO AI / NO EXTERNAL APIS ("JUDGE")
========================================================================================
  * Operates 100% offline/locally on local compute (CPU/GPU) using PyTorch & Scikit-Learn *
  * Zero Third-Party API calls, zero LLM prompts, pure machine learning inference *

                                     [ Input Sequence ]
                                                │
                                                ▼
                            [ Subword Tokenizer (Fast Tokenizer) ]
                            • Max Sequence Length: 512 Tokens
                            • Generates: input_ids, attention_mask
                                                │
                                                ▼
               [ Pre-Trained Fine-Tuned Transformer Encoder Backbone ]
               • Model: microsoft/deberta-v3-base (or roberta-base)
               • Mechanism: Disentangled Bidirectional Self-Attention
               • Extracts: 768-dimensional contextual token embeddings
                                                │
                                                ▼
                  [ Sequence Classification Pooling & Classification Head ]
                  • Standard AutoModelForSequenceClassification Layer
                  • Structure: LayerNorm ──▶ Dropout (p = 0.1) ──▶ Linear(768, 2)
                                                │
                                                ▼
                                    [ Loss & Optimization ]
                  • Loss Function: CrossEntropyLoss (with Class Weights)
                  • Optimizer: AdamW (LR: 2e-5, Weight Decay: 0.01)
                                                │
                                                ▼
========================================================================================
                          STAGE 4: INFERENCE OUTPUT & AUDIT LOG
========================================================================================
                                     [ Softmax Probabilities ]
                                                │
                                                ▼
                                  [ Calibrated Decision Engine ]
                                  • If P(UNSAFE) >= 0.35 ──▶ UNSAFE
                                  • If P(UNSAFE) <  0.35 ──▶ SAFE
                                                │
                                                ▼
                                   [ Final JSON API Output ]
                                   {
                                     "verdict": "UNSAFE",
                                     "confidence": 0.984,
                                     "raw_narrative": "..."
                                   }
```

---

## Component Separation: Perception vs. Local Machine Learning

| Stage | Role | Execution Environment | Technology Stack |
| :--- | :--- | :--- | :--- |
| **Stage 2 ("Eyes")** | Video Visual Perception & Narrative Generator | Cloud API | Google Gemini 1.5/2.5 Multimodal API |
| **Stage 3 ("Judge")** | Child-Safety Sequence Classifier | **100% Offline / Local Compute** | **Fine-Tuned DeBERTa-v3 (`microsoft/deberta-v3-base`) + PyTorch** |

> **Key Privacy & Safety Guarantee:** Stage 3 runs strictly on local hardware with zero external API calls, zero cloud dependencies, and zero third-party LLM prompts. Decision boundaries are learned via fine-tuned Transformer weights and optimized with PyTorch.

---

## Dataset: SafeWatch-Bench-200K (Audited Splits)

The model is trained on a strictly audited, balanced distribution of **25,000 samples** sourced from [**`Zhaorun/SafeWatch-Bench-200K-720P`**](https://huggingface.co/datasets/Zhaorun/SafeWatch-Bench-200K-720P):

* **Total Samples:** 25,000 (12,500 `SAFE`, 12,500 `UNSAFE`)
* **Training Set:** 20,000 samples (80% stratified) $\rightarrow$ `data/safewatch_train.csv`
* **Validation Set:** 2,500 samples (10% stratified) $\rightarrow$ `data/safewatch_val.csv`
* **Testing Set:** 2,500 samples (10% stratified) $\rightarrow$ `data/safewatch_test.csv`
* **Audited Non-Sexual Filtering:** 100% free of adult, erotic, or sexual keywords/flags. Focused purely on child safety: Physical Altercations, Hazards, Weapons, Harassment, and Wholesome/Educational content.

---

## Project Structure

```
Kid-Friendly-Video-Classification-ML-Project/
├── data/
│   ├── safewatch_train.csv         # 20,000 audited training samples
│   ├── safewatch_val.csv           # 2,500 validation samples
│   ├── safewatch_test.csv          # 2,500 held-out test samples
│   └── test_predictions.csv       # Test set predictions with probabilities
├── models/
│   └── deberta_kid_safe/           # Local fine-tuned DeBERTa-v3 weights & tokenizer
├── src/
│   ├── dataset.py                  # SafeWatch-Bench extraction & adult-filtering module
│   ├── model.py                    # PyTorch DeBERTa-v3 model architecture & datasets
│   └── sentence_scanner.py         # Real-time sentence-level early-exit scanner
├── train.py                        # Local PyTorch Transformer fine-tuning script
├── predict.py                      # Offline local inference CLI (outputs calibrated JSON)
└── README.md                       # Complete 4-stage pipeline documentation
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install torch transformers accelerate sentencepiece tiktoken pandas scikit-learn
```

### 2. Fine-Tune the Local Transformer
```bash
python train.py --epochs 2 --lr 2e-5 --batch_size 8 --output_dir models/deberta_kid_safe
```

### 3. Run Local Inference
```bash
python predict.py --text "The video shows a man holding a handgun pointing it at another person's chest."
```

Output:
```json
{
  "verdict": "UNSAFE",
  "confidence": 0.9840,
  "raw_narrative": "The video shows a man holding a handgun pointing it at another person's chest.",
  "probabilities": {
    "SAFE": 0.0160,
    "UNSAFE": 0.9840
  },
  "threshold": 0.35
}
```
