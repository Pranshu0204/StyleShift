"""
Component C — Logistic Regression Meta-Learner (Stacking)

Combines outputs from Components A and B into a final prediction.

Input features per pair (3-dim):
  [0] svm_prob        — P(style_change) from SVM
  [1] siamese_prob    — P(style_change) from Siamese
  [2] cosine_sim      — raw cosine similarity between the two blocks

Trained on validation set predictions to avoid train-set leakage:
in-sample SVM probabilities are overconfident and would bias the ensemble.
LR coefficients reveal how much weight the ensemble places on each signal.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

RESULTS_DIR = Path("results")

# ── Hyperparameters ──
META_C = 1.0          # L2 regularization: C is the inverse of regularization strength
META_MAX_ITER = 1000  # enough iterations for convergence on 3 features


def build_meta_features(
    svm_probs: np.ndarray,      # (N, 2) — column 1 is P(style_change)
    siamese_probs: np.ndarray,  # (N,)
    cosine_sims: np.ndarray,    # (N,) — from X_emb[:, 0]
) -> np.ndarray:
    """Stack three signals into a (N, 3) feature matrix for the meta-learner."""
    return np.column_stack([
        svm_probs[:, 1],    # P(style_change) from SVM
        siamese_probs,       # P(style_change) from Siamese
        cosine_sims,         # cosine similarity (raw feature, not probability)
    ]).astype(np.float32)


def train_meta_learner(X_meta: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """
    Train the meta-learner on validation set predictions.
    Must be called with predictions FROM the validation set — not training set.
    """
    meta = LogisticRegression(C=META_C, penalty="l2", max_iter=META_MAX_ITER, random_state=42)
    meta.fit(X_meta, y)

    print(f"  Meta-learner coefficients: "
          f"SVM={meta.coef_[0][0]:.3f}, "
          f"Siamese={meta.coef_[0][1]:.3f}, "
          f"CosineSim={meta.coef_[0][2]:.3f}")
    return meta


def save_meta_learner(meta: LogisticRegression) -> None:
    joblib.dump(meta, RESULTS_DIR / "component_c_meta.joblib")
    print(f"  Meta-learner saved → {RESULTS_DIR}/component_c_meta.joblib")


def load_meta_learner() -> LogisticRegression:
    return joblib.load(RESULTS_DIR / "component_c_meta.joblib")
