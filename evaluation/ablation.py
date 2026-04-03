"""
Stage 3 — Ablation Study (threshold-tuned, per-difficulty)

Compares four variants on the same validation set:
  1. Stream A only — SVM on stylometric features
  2. Stream B only — cosine similarity threshold baseline
  3. Siamese Network alone — no classical features
  4. Full ensemble (proposed system)

Three views are reported for each variant:
  A) Default threshold (0.5) — the headline number most papers show
  B) Per-variant TUNED threshold — picked on the validation set to
     maximise minority-class (change) F1. For a paper-ready result this
     should be a separate dev fold; for a portfolio project val-tuning
     is fine if labelled honestly as such (which we do here).
  C) Per-difficulty breakdown (easy / medium / hard) — the
     research-defining finding: does the ensemble degrade more gracefully
     than the baselines on `hard` documents where pure stylometry is the
     only available signal?

Also reports AUC-ROC, AUC-PR, and Brier alongside F1. These are
threshold-independent and tell the calibration story that the @0.5 macro F1
alone hides.

Outputs:
  results/ablation_results.json — full numbers (default, tuned, per-difficulty)
  results/tuned_thresholds.json — read by document_eval.py for consistency

Usage:
    python evaluation/ablation.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

RESULTS_DIR = Path("results")
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

# Threshold sweep — fine grid for stable tuning under heavy imbalance
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.025)

# Stable variant keys used in tuned_thresholds.json (consumed by document_eval.py)
VARIANT_KEYS = {
    "Stream A only (SVM)":      "svm",
    "Stream B only (cosine)":   "cosine",
    "Siamese Network alone":    "siamese",
    "Full ensemble (proposed)": "ensemble",
}


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def load_data() -> dict:
    return {
        "y_val":         np.load(RESULTS_DIR / "y_val.npy"),
        "svm_probs":     np.load(RESULTS_DIR / "val_svm_probs.npy"),
        "siamese_probs": np.load(RESULTS_DIR / "val_siamese_probs.npy"),
        "cosine_sims":   np.load(RESULTS_DIR / "val_cosine.npy"),
    }


def load_difficulties(split: str = "val") -> np.ndarray:
    """Per-pair difficulty array, aligned with the y_val / *_probs arrays."""
    with open(RESULTS_DIR / f"doc_ids_{split}.json") as f:
        meta = json.load(f)
    return np.array([m.get("difficulty", "unknown") for m in meta])


# ──────────────────────────────────────────────────────────────
# Scoring primitives
# ──────────────────────────────────────────────────────────────

def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold that maximises minority-class F1 on the val set."""
    best_f1, best_t = -1.0, 0.5
    for t in THRESHOLD_GRID:
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def score(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Full metric suite at the given threshold. AUC-ROC/PR/Brier are
    threshold-independent and identical regardless of `threshold`."""
    pred = (y_prob >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {
        "threshold": float(threshold),
        "macro_f1":  float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "change_f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_true, y_prob)),
        "auc_pr":    float(auc(recall, precision)),
        "brier":     float(brier_score_loss(y_true, y_prob)),
    }


def per_difficulty(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    difficulties: np.ndarray,
    threshold: float,
) -> dict:
    """Score this variant separately on each difficulty subset."""
    out = {}
    for diff in ("easy", "medium", "hard"):
        mask = difficulties == diff
        if mask.sum() == 0:
            continue
        sub = score(y_true[mask], y_prob[mask], threshold)
        sub["n_pairs"] = int(mask.sum())
        out[diff] = sub
    return out


# ──────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────

def print_metric_table(title: str, results: dict, view: str) -> None:
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print('=' * 90)
    print(f"  {'Variant':<32} {'Thr':>5} {'MacroF1':>8} {'ChangeF1':>9} "
          f"{'AUC-ROC':>8} {'AUC-PR':>7} {'Brier':>7}")
    print('-' * 90)
    for variant in results:
        r = results[variant][view]
        marker = "→" if "ensemble" in variant.lower() else " "
        print(f"{marker} {variant:<32} {r['threshold']:>5.2f} "
              f"{r['macro_f1']:>8.4f} {r['change_f1']:>9.4f} "
              f"{r['auc_roc']:>8.4f} {r['auc_pr']:>7.4f} {r['brier']:>7.4f}")
    print('=' * 90)


def print_per_difficulty_table(results: dict, metric: str = "change_f1") -> None:
    print(f"\n{'=' * 78}")
    print(f"  PER-DIFFICULTY BREAKDOWN — {metric} at tuned threshold")
    print('=' * 78)
    print(f"  {'Variant':<32} {'easy':>12} {'medium':>12} {'hard':>12}")
    print('-' * 78)
    for variant in results:
        pd_ = results[variant]["per_difficulty"]
        vals = {d: pd_.get(d, {}).get(metric, float("nan")) for d in ("easy", "medium", "hard")}
        marker = "→" if "ensemble" in variant.lower() else " "
        print(f"{marker} {variant:<32} {vals['easy']:>12.4f} "
              f"{vals['medium']:>12.4f} {vals['hard']:>12.4f}")
    print('=' * 78)
    print("  Decreasing scores from easy → hard are expected: 'hard' documents")
    print("  share a single topic, removing topical signal and isolating pure style.")
    print("  Research question: does the hybrid system degrade more GRACEFULLY")
    print("  than any single-component baseline?")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    data = load_data()
    y_true = data["y_val"]
    difficulties = load_difficulties("val")

    # Build the four variant probability arrays
    cos_inv = 1.0 - (data["cosine_sims"] - data["cosine_sims"].min()) / (
        data["cosine_sims"].max() - data["cosine_sims"].min() + 1e-8
    )

    from component_c_ensemble import load_meta_learner, build_meta_features
    meta = load_meta_learner()
    X_meta = build_meta_features(
        data["svm_probs"], data["siamese_probs"], data["cosine_sims"]
    )
    ensemble_probs = meta.predict_proba(X_meta)[:, 1]

    variants: dict[str, np.ndarray] = {
        "Stream A only (SVM)":      data["svm_probs"][:, 1],
        "Stream B only (cosine)":   cos_inv,
        "Siamese Network alone":    data["siamese_probs"],
        "Full ensemble (proposed)": ensemble_probs,
    }

    # Tune threshold per variant + compute all three views
    results: dict[str, dict] = {}
    tuned_thresholds: dict[str, float] = {}
    for name, prob in variants.items():
        t = best_threshold(y_true, prob)
        tuned_thresholds[VARIANT_KEYS[name]] = t
        results[name] = {
            "default":        score(y_true, prob, 0.5),
            "tuned":          score(y_true, prob, t),
            "per_difficulty": per_difficulty(y_true, prob, difficulties, t),
        }

    # Print the three tables
    print_metric_table("ABLATION — at DEFAULT threshold 0.5", results, "default")
    print_metric_table("ABLATION — at per-variant TUNED threshold", results, "tuned")
    print_per_difficulty_table(results, metric="change_f1")

    # ── Headline finding ──
    best_baseline = max(
        results["Stream A only (SVM)"]["tuned"]["change_f1"],
        results["Stream B only (cosine)"]["tuned"]["change_f1"],
        results["Siamese Network alone"]["tuned"]["change_f1"],
    )
    ensemble_f1 = results["Full ensemble (proposed)"]["tuned"]["change_f1"]
    gain = ensemble_f1 - best_baseline

    print(f"\n{'─' * 60}")
    print(f"  HEADLINE (tuned threshold, change-class F1)")
    print(f"  Best single component : {best_baseline:.4f}")
    print(f"  Full ensemble         : {ensemble_f1:.4f}")
    print(f"  Ensemble gain         : {gain:+.4f}")
    if gain > 0:
        print("  → Hybrid stylometry + deep embeddings outperforms either alone.")
    else:
        print("  → Best single component matches the ensemble on F1@tuned.")
        print("    Inspect AUC-ROC, AUC-PR, Brier above for the calibration story;")
        print("    the ensemble usually wins those even when F1 is comparable.")
    print('─' * 60)

    # Save full results + tuned thresholds (the latter for document_eval.py)
    with open(RESULTS_DIR / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(RESULTS_DIR / "tuned_thresholds.json", "w") as f:
        json.dump(tuned_thresholds, f, indent=2)
    print("\nSaved: results/ablation_results.json")
    print("Saved: results/tuned_thresholds.json  (consumed by document_eval.py)")


if __name__ == "__main__":
    main()
