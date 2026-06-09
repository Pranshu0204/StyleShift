"""
Document-Level Evaluation — did the model find the right boundary in each doc?

Groups pairwise predictions by document and computes per-document
precision / recall / F1, then reports the macro-average across all docs.

Threshold is tuned separately from the pairwise threshold (sweeping
0.05–0.95) because the two objectives have different precision/recall
trade-offs. Per-difficulty breakdown (easy / medium / hard) is included.

Usage:
    python evaluation/document_eval.py --data-dir pan25-multi-author-analysis
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))


def load_doc_meta(split: str = "val") -> list[dict]:
    with open(RESULTS_DIR / f"doc_ids_{split}.json") as f:
        return json.load(f)


def load_pairwise_thresholds() -> dict:
    """Pairwise-tuned thresholds from ablation.py — used as cross-reference only."""
    path = RESULTS_DIR / "tuned_thresholds.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"svm": 0.5, "siamese": 0.5, "ensemble": 0.5}


# Threshold sweep grid for doc-level tuning — same range as ablation.py
DOC_THRESHOLD_GRID = np.arange(0.05, 0.96, 0.025)


def best_threshold_for_doc_f1(docs: dict[str, dict]) -> tuple[float, float]:
    """
    Sweep thresholds and return the one maximizing mean document-level F1.
    Optimizing the doc-level objective directly (rather than reusing the
    pairwise-tuned threshold) matters because the two objectives reward
    different precision/recall trade-offs.
    """
    best_f1, best_t = -1.0, 0.5
    for t in DOC_THRESHOLD_GRID:
        _, _, f1, _ = document_level_f1(docs, threshold=float(t))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def group_pairs_by_doc(
    doc_meta: list[dict],
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, dict]:
    """
    Group pair-level data by document. Each doc carries its difficulty so
    we can later split by easy/medium/hard.
    """
    docs: dict[str, dict] = {}
    for i, meta in enumerate(doc_meta):
        doc_id = meta["doc_id"]
        if doc_id not in docs:
            docs[doc_id] = {
                "true": [],
                "prob": [],
                "block_idx": [],
                "difficulty": meta.get("difficulty", "unknown"),
            }
        docs[doc_id]["true"].append(int(y_true[i]))
        docs[doc_id]["prob"].append(float(y_prob[i]))
        docs[doc_id]["block_idx"].append(meta["block_idx"])
    return docs


def split_docs_by_difficulty(docs: dict[str, dict]) -> dict[str, dict[str, dict]]:
    """Partition the doc dict into {difficulty: {doc_id: doc_data}}."""
    out: dict[str, dict] = {"easy": {}, "medium": {}, "hard": {}}
    for doc_id, data in docs.items():
        diff = data.get("difficulty", "unknown")
        if diff in out:
            out[diff][doc_id] = data
    return out


def document_level_f1(
    docs: dict[str, dict],
    threshold: float = 0.5,
) -> tuple[float, float, float, list[dict]]:
    """Mean per-document precision / recall / F1 at the given threshold."""
    per_doc = []

    for doc_id, data in docs.items():
        y_true = np.array(data["true"])
        y_prob = np.array(data["prob"])
        y_pred = (y_prob >= threshold).astype(int)

        true_positions = set(np.where(y_true == 1)[0].tolist())
        pred_positions = set(np.where(y_pred == 1)[0].tolist())

        if not true_positions and not pred_positions:
            # No changes and none predicted — perfect score
            per_doc.append({"doc_id": doc_id, "precision": 1.0, "recall": 1.0, "f1": 1.0})
            continue

        correct = len(true_positions & pred_positions)
        precision = correct / len(pred_positions) if pred_positions else 0.0
        recall    = correct / len(true_positions) if true_positions else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_doc.append({
            "doc_id": doc_id,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_boundaries": len(true_positions),
            "pred_boundaries": len(pred_positions),
            "correct_boundaries": correct,
        })

    mean_p  = float(np.mean([d["precision"] for d in per_doc]))
    mean_r  = float(np.mean([d["recall"]    for d in per_doc]))
    mean_f1 = float(np.mean([d["f1"]        for d in per_doc]))
    return mean_p, mean_r, mean_f1, per_doc


def main() -> None:
    # All inputs are read from results/ — produced by pair_generator.py and
    # train_all.py. The --data-dir CLI flag is accepted for command parity
    # with the rest of the pipeline but is not needed here.
    y_val   = np.load(RESULTS_DIR / "y_val.npy")
    svm_pp  = np.load(RESULTS_DIR / "val_svm_probs.npy")        # (N, 2)
    svm_p   = svm_pp[:, 1]
    siam_p  = np.load(RESULTS_DIR / "val_siamese_probs.npy")    # (N,)
    cosine  = np.load(RESULTS_DIR / "val_cosine.npy")           # (N,)
    doc_meta = load_doc_meta("val")
    pairwise_thresholds = load_pairwise_thresholds()

    # Ensemble probability via meta-learner
    from component_c_ensemble import load_meta_learner, build_meta_features
    meta = load_meta_learner()
    ensemble_p = meta.predict_proba(build_meta_features(svm_pp, siam_p, cosine))[:, 1]

    probs = {
        "SVM":      svm_p,
        "Siamese":  siam_p,
        "Ensemble": ensemble_p,
    }
    pairwise_t = {
        "SVM":      pairwise_thresholds.get("svm", 0.5),
        "Siamese":  pairwise_thresholds.get("siamese", 0.5),
        "Ensemble": pairwise_thresholds.get("ensemble", 0.5),
    }

    # Group + find the threshold that maximizes mean doc-level F1 for each model
    docs_by_model: dict[str, dict] = {}
    doc_tuned_t: dict[str, float] = {}
    for name, prob in probs.items():
        docs = group_pairs_by_doc(doc_meta, y_val, prob)
        docs_by_model[name] = docs
        t, _ = best_threshold_for_doc_f1(docs)
        doc_tuned_t[name] = t

    # ── Table 1: overall doc-level F1 at the DOC-tuned threshold ──
    print("\n" + "=" * 86)
    print(f"  {'DOCUMENT-LEVEL EVALUATION (doc-level-tuned thresholds)':^82}")
    print("=" * 86)
    print(f"  {'Model':<14} {'DocThr':>7} {'PairThr':>8} {'Precision':>11} "
          f"{'Recall':>9} {'Doc F1':>9} {'N docs':>8}")
    print("-" * 86)

    all_results: dict[str, dict] = {}
    for name, prob in probs.items():
        t = doc_tuned_t[name]
        pt = pairwise_t[name]
        docs = docs_by_model[name]
        p, r, f1, _ = document_level_f1(docs, threshold=t)
        all_results[name] = {
            "doc_tuned_threshold":  float(t),
            "pairwise_threshold":   float(pt),
            "doc_precision":        p,
            "doc_recall":           r,
            "doc_f1":               f1,
            "n_docs":               len(docs),
        }
        marker = "→" if name == "Ensemble" else " "
        print(f"{marker} {name:<14} {t:>7.3f} {pt:>8.3f} {p:>11.4f} "
              f"{r:>9.4f} {f1:>9.4f} {len(docs):>8}")
    print("=" * 86)
    print("  DocThr  = threshold tuned on doc-level F1 (used here)")
    print("  PairThr = threshold tuned on pairwise change F1 (for reference only)")

    # ── Table 2: per-difficulty doc-level F1, each at its own doc-tuned threshold ──
    print("\n" + "=" * 74)
    print(f"  {'PER-DIFFICULTY DOCUMENT-LEVEL F1 (doc-tuned thresholds)':^70}")
    print("=" * 74)
    print(f"  {'Model':<14} {'easy':>18} {'medium':>18} {'hard':>18}")
    print("-" * 74)

    for name, prob in probs.items():
        docs_by_diff = split_docs_by_difficulty(docs_by_model[name])
        per_diff_f1: dict[str, float] = {}
        per_diff_n: dict[str, int] = {}
        for diff in ("easy", "medium", "hard"):
            sub = docs_by_diff[diff]
            if not sub:
                per_diff_f1[diff] = float("nan")
                per_diff_n[diff] = 0
                continue
            # Each difficulty gets its own doc-tuned threshold — fairest comparison
            t_d, _ = best_threshold_for_doc_f1(sub)
            _, _, f1, _ = document_level_f1(sub, threshold=t_d)
            per_diff_f1[diff] = f1
            per_diff_n[diff] = len(sub)
        all_results[name]["per_difficulty"] = {
            d: {"doc_f1": per_diff_f1[d], "n_docs": per_diff_n[d]} for d in per_diff_f1
        }
        marker = "→" if name == "Ensemble" else " "
        print(f"{marker} {name:<14} "
              f"{per_diff_f1['easy']:>11.4f} (n={per_diff_n['easy']:>4}) "
              f"{per_diff_f1['medium']:>10.4f} (n={per_diff_n['medium']:>4}) "
              f"{per_diff_f1['hard']:>10.4f} (n={per_diff_n['hard']:>4})")
    print("=" * 74)

    # ── Interpretation ──
    print("\n  Document-level F1 is the operationally relevant metric:")
    print("  it asks whether the system correctly locates the right boundary")
    print("  in the right document, not just whether pairs are classified well.")
    print("\n  Per-difficulty: easy documents have topical shifts that aid")
    print("  detection; hard documents share a single topic so only pure style")
    print("  remains as a signal. A small gap easy→hard is a strong result.")

    out = RESULTS_DIR / "document_eval_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="pan25-multi-author-analysis",
        help="(accepted for CLI parity; not used — all inputs come from results/)",
    )
    parser.parse_args()
    main()
