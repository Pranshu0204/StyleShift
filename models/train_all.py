"""
Training Orchestrator — trains all components in the correct order.

Order matters:
  1. Component A (SVM) — trains on training set features
  2. Component B (Siamese) — trains on training set embeddings
  3. Component C (meta-learner) — trains on VALIDATION set predictions
     from A and B (to avoid train-set leakage into ensemble weights)

Usage:
    python models/train_all.py
    python models/train_all.py --skip-siamese   # if GPU not available
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

RESULTS_DIR = Path("results")
sys.path.insert(0, str(Path(__file__).parent))

from component_a_svm import train_svm, train_isolation_forest, save_component_a, svm_predict_proba, iforest_anomaly_score
from component_b_siamese import train_siamese, load_siamese, DEVICE, PairDataset, BATCH_SIZE
from component_c_ensemble import build_meta_features, train_meta_learner, save_meta_learner


def load_matrices() -> dict:
    """Load precomputed feature matrices from pair_generator.py output."""
    d = {}
    for split in ("train", "val"):
        d[f"X_{split}_stylo"] = np.load(RESULTS_DIR / f"X_{split}_stylo.npy")
        d[f"X_{split}_emb"]   = np.load(RESULTS_DIR / f"X_{split}_emb.npy")
        d[f"y_{split}"]       = np.load(RESULTS_DIR / f"y_{split}.npy")
    return d


def get_siamese_probs(model, emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Run Siamese inference in batches. Returns (N,) probability array."""
    from torch.utils.data import DataLoader
    dummy_labels = np.zeros(len(emb_a))
    dataset = PairDataset(emb_a, emb_b, dummy_labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    all_probs = []
    model.eval()
    with torch.no_grad():
        for u, v, _ in loader:
            u, v = u.to(DEVICE), v.to(DEVICE)
            probs = torch.sigmoid(model(u, v)).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
    return np.array(all_probs, dtype=np.float32)


def main(skip_siamese: bool = False) -> None:
    print("Loading feature matrices...")
    d = load_matrices()

    # X_emb layout: [cosine_sim(1), abs_diff(768)]
    # Split into cosine sims and raw embedding differences
    cosine_train = d["X_train_emb"][:, 0]   # column 0 is the cosine similarity
    cosine_val   = d["X_val_emb"][:, 0]

    # For the Siamese network we need separate emb_a and emb_b.
    # pair_generator.py saves them separately if the --save-raw flag is used;
    # otherwise we approximate from the difference. Check for raw files first.
    raw_a_train_path = RESULTS_DIR / "emb_a_train.npy"
    raw_b_train_path = RESULTS_DIR / "emb_b_train.npy"
    has_raw = raw_a_train_path.exists() and raw_b_train_path.exists()

    if has_raw:
        emb_a_train = np.load(raw_a_train_path)
        emb_b_train = np.load(raw_b_train_path)
        emb_a_val   = np.load(RESULTS_DIR / "emb_a_val.npy")
        emb_b_val   = np.load(RESULTS_DIR / "emb_b_val.npy")
    else:
        print("  [INFO] Raw embeddings not found. Siamese will use abs_diff approximation.")
        # Use the abs_diff column and a zero vector as a surrogate
        abs_diff = d["X_train_emb"][:, 1:]  # (N, 768)
        emb_a_train = abs_diff / 2.0
        emb_b_train = -abs_diff / 2.0
        abs_diff_val = d["X_val_emb"][:, 1:]
        emb_a_val = abs_diff_val / 2.0
        emb_b_val = -abs_diff_val / 2.0

    # ── Component A ──
    print("\n" + "="*55)
    print("COMPONENT A — SVM + Isolation Forest")
    print("="*55)
    t0 = time.time()
    svm = train_svm(d["X_train_stylo"], d["y_train"])
    iforest = train_isolation_forest(d["X_train_stylo"])
    save_component_a(svm, iforest)
    print(f"  Component A done in {time.time()-t0:.1f}s")

    # ── Component B ──
    if not skip_siamese:
        print("\n" + "="*55)
        print("COMPONENT B — Siamese Transformer Network")
        print(f"  Device: {DEVICE}")
        print("="*55)
        t0 = time.time()
        siamese = train_siamese(
            emb_a_train, emb_b_train, d["y_train"],
            emb_a_val,   emb_b_val,   d["y_val"],
        )
        print(f"  Component B done in {time.time()-t0:.1f}s")
    else:
        print("\n[SKIP] Siamese training skipped. Loading existing checkpoint...")
        from component_b_siamese import load_siamese
        siamese = load_siamese()

    # ── Component C — train on validation predictions ──
    print("\n" + "="*55)
    print("COMPONENT C — Meta-Learner (trained on val predictions)")
    print("="*55)

    svm_probs_val = svm_predict_proba(svm, d["X_val_stylo"])          # (N, 2)
    siamese_probs_val = get_siamese_probs(siamese, emb_a_val, emb_b_val)  # (N,)

    X_meta_val = build_meta_features(svm_probs_val, siamese_probs_val, cosine_val)
    meta = train_meta_learner(X_meta_val, d["y_val"])
    save_meta_learner(meta)

    # Save the val predictions for ablation (avoid recomputing)
    np.save(RESULTS_DIR / "val_svm_probs.npy", svm_probs_val)
    np.save(RESULTS_DIR / "val_siamese_probs.npy", siamese_probs_val)
    np.save(RESULTS_DIR / "val_cosine.npy", cosine_val)

    print("\n" + "="*55)
    print("All components trained. Run evaluation:")
    print("  python evaluation/metrics.py")
    print("  python evaluation/ablation.py")
    print("="*55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-siamese", action="store_true",
                        help="Skip Siamese training (use if no GPU available)")
    args = parser.parse_args()
    main(skip_siamese=args.skip_siamese)
