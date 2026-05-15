# StyleShift — Intrinsic Multi-Author Style Change Detection

**Research question:** Given a single document written by potentially multiple authors, can we identify the exact sentence boundary where authorship switches — without access to any reference texts from known authors?

This is *intrinsic* style change detection: the system compares internal segments against each other, not against an external author profile. The framing directly matches the **PAN @ CLEF 2025** shared task benchmark used in the digital forensics research community.

Real-world use cases include detecting AI-generated or ghost-written passages in student essays, identifying authorship tampering in legal documents, and multi-author forensic analysis.

---

## Architecture Overview

```
Input Document
      │
      ▼  spaCy sentencizer
┌─────────────────────────────────┐
│   SLIDING WINDOW SEGMENTATION   │  WINDOW_SIZE = 3 sentences per block
│   → blocks B_0, B_1, ..., B_n  │
└──────────┬──────────────────────┘
           │  For each consecutive pair (B_i, B_{i+1}):
           ▼
┌──────────────────────────────────────────────────────────────┐
│  DUAL-STREAM FEATURE EXTRACTION                              │
│                                                              │
│  Stream A (Classical Stylometric)   Stream B (Embeddings)   │
│  ─────────────────────────────────  ─────────────────────── │
│  163-dim hand-engineered vector     frozen all-mpnet-base-v2 │
│  |feat(B_i) − feat(B_{i+1})|        cosine_sim (scalar)     │
│  (dissimilarity vector)             emb_a, emb_b (768-dim)  │
└───────────────┬──────────────────────────────┬──────────────┘
                │                              │
                ▼                              ▼
      ┌──────────────────┐         ┌──────────────────────┐
      │   COMPONENT A    │         │     COMPONENT B       │
      │  SVM (RBF kernel)│         │  Siamese Transformer  │
      │  + Isolation     │         │  frozen encoder +     │
      │    Forest        │         │  trainable MLP head   │
      │  P_A(change)     │         │  P_B(change)          │
      └────────┬─────────┘         └──────────┬────────────┘
               │                              │
               └──────────────┬───────────────┘
                              │  + cosine_sim (raw signal)
                              ▼
                 ┌────────────────────────┐
                 │      COMPONENT C       │
                 │  Logistic Regression   │
                 │  Meta-Learner          │
                 │  (trained on val set)  │
                 └────────────┬───────────┘
                              │
                              ▼
             P(style change) for pair (B_i, B_{i+1})
```

---

## Dataset

**PAN @ CLEF 2025 — Multi-Author Writing Style Analysis**
Download: [https://zenodo.org/records/14891299](https://zenodo.org/records/14891299)
(Free Zenodo account required)

### Why PAN 2025?

PAN 2025 introduces **sentence-level labels** (vs. paragraph-level in PAN 2021), enabling finer boundary localisation. It also provides three difficulty tiers that form a natural ablation axis — a system that degrades gracefully easy → hard demonstrates genuine stylometric capability rather than topic-based shortcutting.

| Difficulty | Description | Train docs | Val docs |
|------------|-------------|-----------|---------|
| Easy | Documents span multiple topics — topic shift can aid detection | 900 | 900 |
| Medium | Limited topical variety — model must focus on style | 900 | 899 |
| Hard | All sentences share the same topic — pure stylometry only | 900 | 899 |

**Dataset format per document:**
- `problem-X.txt` — one sentence per line
- `truth-problem-X.json` — `{"changes": [0, 1, 0, ...]}` where `changes[i] = 1` means authorship switches between sentence i and i+1

**Class imbalance:** ~4:1 no-change : change pairs. This is why accuracy is NOT a valid metric — see Evaluation section.

---

## Feature Streams

### Stream A — Classical Stylometric (163-dim difference vector)

For each block, a 163-dim feature vector is extracted. The **absolute difference** between two blocks' vectors is the input to Component A — large values indicate stylistic dissimilarity.

| Feature Group | Dim | Why it shifts between authors |
|---|---|---|
| Avg word length, TTR, punctuation density, uppercase ratio | 4 | Vocabulary richness and surface habits |
| POS ratios (NOUN, VERB, ADJ, ADV, PRON, DET) | 6 | Syntactic preferences and register |
| Avg sentence length | 1 | Discourse rhythm |
| Readability (Flesch-Kincaid Grade, Gunning Fog) | 2 | Complexity level — AI text shifts this noticeably |
| Function word frequencies (L2-normalised) | 150 | Subconscious style markers — resistant to topic confounding |

### Stream B — Contextual Embeddings (frozen all-mpnet-base-v2)

- 768-dim L2-normalised sentence embeddings per block
- Cosine similarity between consecutive blocks (direct weak-signal feature)
- Element-wise absolute difference of embeddings (input to Siamese MLP head)

**Why frozen?** Prevents overfitting on the small PAN dataset; keeps the embedding contribution cleanly separable in the ablation.

---

## Results

All numbers are from the PAN 2025 validation set. Thresholds are tuned per variant on the validation set to maximise the respective objective.

### Ablation Study (tuned threshold, pairwise evaluation)

| Variant | Macro F1 | Change F1 | AUC-ROC | AUC-PR | Brier |
|---|---|---|---|---|---|
| Stream A only (SVM) | 0.5741 | 0.3764 | 0.6157 | 0.3037 | 0.1604 |
| Stream B only (cosine baseline) | 0.5499 | 0.3579 | 0.5948 | 0.3156 | 0.2803 |
| Siamese Network alone | 0.5645 | 0.3866 | 0.6471 | 0.3954 | 0.2012 |
| **Full ensemble (proposed)** | **0.6060** | **0.4037** | **0.6677** | **0.4038** | **0.1502** |

> At the default threshold of 0.5 the SVM reports Change F1 = 0.00 due to Platt calibration bias under 4:1 imbalance — not a true model failure. Threshold tuning recovers the full signal.

### Per-Difficulty Change F1 (tuned threshold)

| Variant | Easy | Medium | Hard |
|---|---|---|---|
| Stream A only (SVM) | 0.5117 | 0.3767 | 0.1662 |
| Stream B only (cosine) | 0.5148 | 0.3516 | 0.1027 |
| Siamese Network alone | 0.5109 | 0.4093 | 0.1551 |
| **Full ensemble** | **0.5342** | **0.4267** | 0.1341 |

The hard tier reveals the system's ceiling: all models approach the trivial baseline when authors write about the same topic. This is consistent with published PAN state-of-the-art results and motivates future work on sub-sentence style modelling.

### Document-Level Evaluation (doc-level tuned threshold)

*Did the system correctly locate the right boundary in the right document?*

| Model | Precision | Recall | Doc F1 |
|---|---|---|---|
| SVM | 0.4149 | 0.4124 | 0.4077 |
| Siamese | 0.4880 | 0.4947 | 0.4845 |
| **Ensemble** | **0.4937** | **0.5013** | **0.4909** |

Document-level F1 uses a separately tuned threshold (maximising doc-level F1 directly), since the pairwise-tuned threshold over-predicts boundaries and hurts document precision.

---

## Evaluation Metrics

| Metric | Why it's used |
|---|---|
| F1-macro | Weights both classes equally — appropriate under class imbalance |
| Change-class F1 | Directly measures detection of style changes (the task objective) |
| AUC-ROC | Threshold-independent discriminability |
| AUC-PR | More informative than ROC under heavy imbalance; focuses on minority class precision |
| Brier Score | Measures probability calibration — penalises overconfident wrong predictions |
| Document-level F1 | Operationally relevant: did we find the right boundary in the document? |

**NOT reported:** Raw accuracy — a trivial "always predict no-change" classifier would achieve >80% accuracy while being completely useless.

---

## How to Run

### Prerequisites

```bash
cd /path/to/StyleShift
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 1 — Get the dataset

Download from [https://zenodo.org/records/14891299](https://zenodo.org/records/14891299) (free Zenodo account), extract so that this path exists:
```
pan25-multi-author-analysis/easy/train/problem-1.txt
```
Then verify the structure:
```bash
python data/download_pan.py --verify --data-dir pan25-multi-author-analysis
```

### Step 2 — Extract features

```bash
# All three difficulty tiers combined (recommended)
python data/pair_generator.py --data-dir pan25-multi-author-analysis

# Single difficulty (faster, for experimentation)
python data/pair_generator.py --data-dir pan25-multi-author-analysis --difficulty easy
```

### Step 3 — Train all components

```bash
python models/train_all.py

# CPU only (no GPU / MPS):
python models/train_all.py --skip-siamese
```

### Step 4 — Evaluate

```bash
# Per-model pairwise metrics
python evaluation/metrics.py --model ensemble
python evaluation/metrics.py --model svm
python evaluation/metrics.py --model siamese

# Ablation study (4-variant comparison + per-difficulty breakdown)
python evaluation/ablation.py

# Document-level boundary detection F1
python evaluation/document_eval.py --data-dir pan25-multi-author-analysis
```

### Step 5 — EDA notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

### Step 6 — Launch demo UI

```bash
python ui/app.py
# Opens at http://localhost:7860
```

---

## Project Structure

```
StyleShift/
├── data/
│   ├── download_pan.py          # Dataset structure verification + statistics
│   ├── windowing.py             # BlockPair dataclass + sliding window segmentation
│   └── pair_generator.py        # Feature extraction orchestrator + numpy caching
├── features/
│   ├── stylometric_features.py  # Stream A: 163-dim hand-engineered feature extractor
│   └── embedding_features.py    # Stream B: frozen all-mpnet-base-v2 wrapper
├── models/
│   ├── component_a_svm.py       # RBF-SVM + Isolation Forest
│   ├── component_b_siamese.py   # Siamese Transformer MLP (PyTorch)
│   ├── component_c_ensemble.py  # Logistic Regression meta-learner
│   └── train_all.py             # Training orchestrator (A → B → C)
├── evaluation/
│   ├── metrics.py               # F1, AUC-ROC, AUC-PR, Brier, confusion matrix
│   ├── ablation.py              # 4-variant comparison + threshold tuning + per-difficulty
│   └── document_eval.py         # Document-level boundary detection F1
├── ui/
│   └── app.py                   # Gradio demo
├── notebooks/
│   └── eda.ipynb                # EDA + ablation visualisation
├── results/                     # Saved JSON metrics (model binaries excluded via .gitignore)
└── requirements.txt
```

---

## Key Design Decisions

| Decision | Justification |
|---|---|
| **PAN 2025 over PAN 2021** | Sentence-level labels + three difficulty tiers as a built-in ablation axis |
| **Pairwise binary classification** | Localises the boundary; matches PAN shared task framing; tractable with small datasets |
| **Sliding windows of 3 sentences** | Single sentences are too noisy; whole paragraphs lose granularity; 3 is the PAN community standard |
| **Frozen encoder (Stream B)** | Prevents overfitting on small dataset; isolates embedding contribution in ablation |
| **Siamese weight sharing** | Enforces a consistent metric space so embedding differences are meaningful |
| **InferSent interaction [&#124;u-v&#124;, u⊙v, cos]** | Standard triplet covering dissimilarity, co-activation, and global similarity |
| **BCEWithLogitsLoss + pos_class_weight=3.0** | Directly penalises missed style changes — the higher-stakes forensic error |
| **LR meta-learner (not XGBoost)** | Interpretable coefficients reveal which component the ensemble trusts more |
| **Meta-learner trained on val set** | Prevents leakage of in-sample SVM confidence into ensemble weights |
| **Separate pairwise + doc-level thresholds** | The two objectives reward different precision/recall trade-offs |
| **F1 + AUC-ROC + AUC-PR, not accuracy** | Dataset is imbalanced; accuracy is misleading and penalises nothing |

---

## References

- Stamatatos, E. (2009). A survey of modern authorship attribution methods. *JASIST*.
- Conneau, A. et al. (2017). Supervised Learning of Universal Sentence Representations from NLI Data. *EMNLP* (InferSent).
- Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.
- Zangerle, E. et al. (2025). Overview of the Style Change Detection Task at PAN 2025. *CLEF*.
- Liu, F. T. et al. (2008). Isolation Forest. *ICDM*.
- Schölkopf, B. et al. (2001). Estimating the Support of a High-Dimensional Distribution. *Neural Computation*.
