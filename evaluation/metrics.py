"""
Stage 3 — Evaluation Framework

Reports the full metric suite on the validation set. Standard accuracy
is intentionally excluded as the primary metric because the dataset is
heavily class-imbalanced (many more "no change" than "change" pairs).
A trivial classifier that always predicts "no change" could achieve
>80% accuracy while being completely useless for the detection task.

PRIMARY METRICS (used for model comparison):
  - F1-score (macro): weights both classes equally regardless of frequency.
    WHY macro? Under imbalance, weighted F1 can be high even if the minority
    class (style changes) is completely missed. Macro F1 penalizes that.
  - F1-score (minority class = 1): directly measures detection of changes.
  - AUC-ROC: area under the Receiver Operating Characteristic curve.
    WHY AUC-ROC? It measures discriminability across ALL decision thresholds,
    not just the default 0.5. A model with AUC=0.9 can separate positives
    from negatives 90% of the time regardless of the threshold used.

SECONDARY METRICS:
  - AUC-PR (Precision-Recall): more informative than ROC under heavy imbalance.
    WHY? The ROC curve is optimistic under imbalance because it accounts for
    true negatives (which are easy when negatives dominate). The PR curve
    focuses entirely on the positive class — it exposes whether a model
    has real precision on the minority class or just gets lucky at low recall.
  - Brier Score: mean squared error between predicted probabilities and true
    labels. A model can have high AUC but poorly calibrated probabilities.
    Brier Score penalizes overconfident wrong predictions.
  - Normalized confusion matrix as a seaborn heatmap.

Usage:
    python evaluation/metrics.py --model ensemble
    python evaluation/metrics.py --model svm
    python evaluation/metrics.py --model siamese
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

RESULTS_DIR = Path("results")
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))


def load_val_predictions() -> dict:
    """Load precomputed validation predictions from training."""
    return {
        "y_val":            np.load(RESULTS_DIR / "y_val.npy"),
        "svm_probs":        np.load(RESULTS_DIR / "val_svm_probs.npy"),
        "siamese_probs":    np.load(RESULTS_DIR / "val_siamese_probs.npy"),
        "cosine_sims":      np.load(RESULTS_DIR / "val_cosine.npy"),
    }


def get_ensemble_probs(data: dict) -> np.ndarray:
    """Apply the trained meta-learner to get ensemble probabilities."""
    from component_c_ensemble import load_meta_learner, build_meta_features
    meta = load_meta_learner()
    X_meta = build_meta_features(data["svm_probs"], data["siamese_probs"], data["cosine_sims"])
    return meta.predict_proba(X_meta)[:, 1]


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> dict:
    """Compute and print the full metric suite for a given probability array."""
    y_pred = (y_prob >= 0.5).astype(int)

    macro_f1   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    change_f1  = f1_score(y_true, y_pred, pos_label=1,        zero_division=0)
    auc_roc    = roc_auc_score(y_true, y_prob)
    brier      = brier_score_loss(y_true, y_prob)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = auc(recall, precision)

    results = {
        "model": model_name,
        "macro_f1": float(macro_f1),
        "change_class_f1": float(change_f1),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "brier_score": float(brier),
    }

    print(f"\n{'='*50}")
    print(f"RESULTS — {model_name.upper()}")
    print(f"{'='*50}")
    print(f"  Macro F1:          {macro_f1:.4f}")
    print(f"  Change-class F1:   {change_f1:.4f}  (minority class)")
    print(f"  AUC-ROC:           {auc_roc:.4f}")
    print(f"  AUC-PR:            {auc_pr:.4f}")
    print(f"  Brier Score:       {brier:.4f}")

    return results


def plot_roc_pr(models_data: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """
    Plot ROC and Precision-Recall curves for all models on one figure.
    models_data: {model_name: (y_true, y_prob)}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#1565C0", "#EF5350", "#4CAF50", "#FF9800"]

    for (model_name, (y_true, y_prob)), color in zip(models_data.items(), colors):
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, lw=2, label=f"{model_name} (AUC={roc_auc:.3f})")

        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        axes[1].plot(recall, precision, color=color, lw=2, label=f"{model_name} (AUC={pr_auc:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(fontsize=8)

    baseline_pr = y_true.mean()
    axes[1].axhline(baseline_pr, color="black", linestyle="--", lw=1, label=f"Random (AP={baseline_pr:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "roc_pr_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  ROC/PR curves saved → {out}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=["same author", "style change"],
        yticklabels=["same author", "style change"],
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Normalized Confusion Matrix — {model_name}")
    plt.tight_layout()
    out = RESULTS_DIR / f"confusion_{model_name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {out}")


def main(model: str) -> None:
    data = load_val_predictions()
    y_true = data["y_val"]

    # Map model name to probability array
    if model == "svm":
        y_prob = data["svm_probs"][:, 1]
    elif model == "siamese":
        y_prob = data["siamese_probs"]
    elif model == "cosine_baseline":
        # Cosine baseline: low cosine → high P(change). Invert and normalize.
        cos = data["cosine_sims"]
        y_prob = 1.0 - (cos - cos.min()) / (cos.max() - cos.min() + 1e-8)
    elif model == "ensemble":
        y_prob = get_ensemble_probs(data)
    else:
        raise ValueError(f"Unknown model: {model}")

    results = compute_metrics(y_true, y_prob, model)
    y_pred = (y_prob >= 0.5).astype(int)
    plot_confusion_matrix(y_true, y_pred, model)

    # Save results
    out = RESULTS_DIR / f"metrics_{model}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Metrics saved → {out}")

    # Also plot all models' curves together if ensemble is requested
    if model == "ensemble":
        all_models = {
            "SVM": (y_true, data["svm_probs"][:, 1]),
            "Siamese": (y_true, data["siamese_probs"]),
            "Cosine baseline": (y_true, 1.0 - (data["cosine_sims"] - data["cosine_sims"].min()) /
                                (data["cosine_sims"].max() - data["cosine_sims"].min() + 1e-8)),
            "Ensemble": (y_true, get_ensemble_probs(data)),
        }
        plot_roc_pr(all_models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ensemble",
                        choices=["svm", "siamese", "cosine_baseline", "ensemble"])
    args = parser.parse_args()
    main(args.model)
