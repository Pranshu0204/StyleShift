"""
Component A — SVM Classifier + Isolation Forest

Trains two models on the 163-dim stylometric difference vectors (Stream A):

1. SVC(kernel='rbf', probability=True, class_weight='balanced')
   Platt scaling enables calibrated probabilities for the meta-learner.
   balanced class weights handle the ~4:1 no-change:change imbalance.

2. IsolationForest — frames style change as anomaly detection.
   Learns the distribution of same-author differences; style-change pairs
   score as anomalies without requiring negative examples at train time.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ──
SVM_KERNEL = "rbf"
SVM_C = 1.0         # regularization: larger C = tighter margin = risk of overfitting
SVM_GAMMA = "scale" # 1 / (n_features * X.var()) — adapts to feature scale
SVM_CLASS_WEIGHT = "balanced"  # upweights minority class (style changes) automatically

IFOREST_N_ESTIMATORS = 200
IFOREST_CONTAMINATION = 0.15  # expected fraction of outliers (style-change pairs)
IFOREST_RANDOM_STATE = 42


def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> SVC:
    """
    Train an RBF-kernel SVM with balanced class weights and Platt scaling.
    `class_weight='balanced'` automatically sets per-class weights inversely
    proportional to class frequency — equivalent to oversampling the minority
    class without duplicating data.
    """
    print(f"  Training SVM (kernel={SVM_KERNEL}, C={SVM_C}, class_weight={SVM_CLASS_WEIGHT})...")
    svm = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        probability=True,       # enables predict_proba() via Platt scaling
        class_weight=SVM_CLASS_WEIGHT,
        random_state=42,
    )
    svm.fit(X_train, y_train)
    print(f"  SVM trained on {X_train.shape[0]} samples, {X_train.shape[1]} features.")
    return svm


def train_isolation_forest(X_train: np.ndarray) -> IsolationForest:
    """
    Train Isolation Forest on ALL training samples (not just same-author pairs).
    It will learn that same-author differences are the "normal" distribution.
    """
    print(f"  Training Isolation Forest ({IFOREST_N_ESTIMATORS} trees)...")
    iforest = IsolationForest(
        n_estimators=IFOREST_N_ESTIMATORS,
        contamination=IFOREST_CONTAMINATION,
        random_state=IFOREST_RANDOM_STATE,
        n_jobs=-1,
    )
    iforest.fit(X_train)
    return iforest


def svm_predict_proba(svm: SVC, X: np.ndarray) -> np.ndarray:
    """Returns (N, 2) probability array — column 1 is P(style_change)."""
    return svm.predict_proba(X)


def iforest_anomaly_score(iforest: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Returns a (N,) array of anomaly scores in [0, 1].
    Isolation Forest's raw score_samples() output is negative and lower
    means more anomalous. We flip and normalize to [0, 1] so higher = more
    likely a style change.
    """
    raw_scores = iforest.score_samples(X)          # ≤ 0, lower = more anomalous
    normalized = -raw_scores                        # flip sign: higher = more anomalous
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
    return normalized


def save_component_a(svm: SVC, iforest: IsolationForest) -> None:
    joblib.dump(svm, RESULTS_DIR / "component_a_svm.joblib")
    joblib.dump(iforest, RESULTS_DIR / "component_a_iforest.joblib")
    print(f"  Component A saved → {RESULTS_DIR}/")


def load_component_a() -> tuple[SVC, IsolationForest]:
    svm = joblib.load(RESULTS_DIR / "component_a_svm.joblib")
    iforest = joblib.load(RESULTS_DIR / "component_a_iforest.joblib")
    return svm, iforest


if __name__ == "__main__":
    X_train = np.load(RESULTS_DIR / "X_train_stylo.npy")
    y_train = np.load(RESULTS_DIR / "y_train.npy")
    svm = train_svm(X_train, y_train)
    iforest = train_isolation_forest(X_train)
    save_component_a(svm, iforest)
