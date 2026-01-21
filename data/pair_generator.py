"""
Pair Generator — Builds and Caches Feature-Ready Dataset

Loads all BlockPairs from the PAN 2025 dataset, extracts features for
both streams, and saves numpy arrays to disk. Running this once avoids
re-extracting features on every experiment (~10–30 min for all difficulties).

Outputs saved to results/:
  X_train_stylo.npy   — Stream A difference vectors (train)
  X_val_stylo.npy     — Stream A difference vectors (val)
  X_train_emb.npy     — Stream B: [cosine_sim, |emb_diff|] (train)
  X_val_emb.npy       — Stream B (val)
  emb_a_train.npy     — Raw block_a embeddings (train, for Siamese)
  emb_b_train.npy     — Raw block_b embeddings (train, for Siamese)
  emb_a_val.npy       — Raw block_a embeddings (val, for Siamese)
  emb_b_val.npy       — Raw block_b embeddings (val, for Siamese)
  y_train.npy         — Labels (train)
  y_val.npy           — Labels (val)
  doc_ids_train.json  — doc_id + block_idx + difficulty (for doc-level eval)
  doc_ids_val.json    — Same for val split
  feature_names.json  — Feature names for Stream A (163 dimensions)

Usage:
    # All three difficulties combined (recommended for training)
    python data/pair_generator.py --data-dir pan25-multi-author-analysis

    # Single difficulty (useful for per-difficulty ablation experiments)
    python data/pair_generator.py --data-dir pan25-multi-author-analysis --difficulty hard
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "features"))

from stylometric_features import extract_stylometric_features, get_feature_names
from embedding_features import EmbeddingExtractor
from windowing import load_all_pairs, BlockPair

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

PAN_DATA_DIR = Path("pan25-multi-author-analysis")


def extract_stream_a(pairs: list[BlockPair], show_progress: bool = True) -> np.ndarray:
    """
    For each pair, extract stylometric features for both blocks, then
    compute the ABSOLUTE DIFFERENCE vector.

    Taking the absolute difference converts two feature vectors into a
    single dissimilarity vector — large values indicate the two blocks are
    stylistically different. This is the standard approach in pairwise
    stylometric comparison (Stamatatos 2009).
    """
    rows = []
    iterator = tqdm(pairs, desc="Stream A (stylometric)") if show_progress else pairs
    for pair in iterator:
        feat_a = extract_stylometric_features(pair.block_a)
        feat_b = extract_stylometric_features(pair.block_b)
        diff = np.abs(feat_a - feat_b)  # dissimilarity vector
        rows.append(diff)
    return np.stack(rows, axis=0).astype(np.float32)


def extract_stream_b(
    pairs: list[BlockPair], extractor: EmbeddingExtractor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each pair, encode both blocks, then compute:
      1. Cosine similarity (scalar)
      2. Element-wise absolute difference of the 768-dim embeddings

    The cosine similarity is a strong weak-signal feature: a sudden drop
    between consecutive blocks is a reliable indicator of a style or topic
    shift. The difference vector provides richer comparison for the Siamese
    network and meta-learner.

    Returns:
      X_emb:  (N, 1 + 768) float32 — [cosine_sim, abs_diff...] for SVM/meta-learner
      emb_a:  (N, 768) float32 — raw embeddings for block_a (needed by Siamese)
      emb_b:  (N, 768) float32 — raw embeddings for block_b (needed by Siamese)
    """
    all_texts_a = [p.block_a for p in pairs]
    all_texts_b = [p.block_b for p in pairs]

    print("  Encoding block_a texts...")
    emb_a = extractor.encode(all_texts_a)  # (N, 768)
    print("  Encoding block_b texts...")
    emb_b = extractor.encode(all_texts_b)  # (N, 768)

    # Cosine similarity — element-wise dot product of L2-normalized vectors
    # Since encoder outputs are already L2-normalized, dot product = cosine sim
    cosine_sims = np.sum(emb_a * emb_b, axis=1, keepdims=True)  # (N, 1)
    abs_diff = np.abs(emb_a - emb_b)                             # (N, 768)
    X_emb = np.concatenate([cosine_sims, abs_diff], axis=1).astype(np.float32)

    return X_emb, emb_a.astype(np.float32), emb_b.astype(np.float32)


def save_split(
    pairs: list[BlockPair],
    X_stylo: np.ndarray,
    X_emb: np.ndarray,
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    split: str,
) -> None:
    y = np.array([p.label for p in pairs], dtype=np.int32)
    # Store difficulty alongside doc_id so document_eval.py can group by difficulty
    doc_meta = [{"doc_id": p.doc_id, "block_idx": p.block_idx, "difficulty": p.difficulty} for p in pairs]

    np.save(RESULTS_DIR / f"X_{split}_stylo.npy", X_stylo)
    np.save(RESULTS_DIR / f"X_{split}_emb.npy", X_emb)
    # Save raw embeddings separately — required by the Siamese network (train_all.py).
    # The combined X_emb stores only [cosine_sim, abs_diff] which is insufficient
    # for the Siamese forward pass that needs individual emb_a and emb_b.
    np.save(RESULTS_DIR / f"emb_a_{split}.npy", emb_a)
    np.save(RESULTS_DIR / f"emb_b_{split}.npy", emb_b)
    np.save(RESULTS_DIR / f"y_{split}.npy", y)
    with open(RESULTS_DIR / f"doc_ids_{split}.json", "w") as f:
        json.dump(doc_meta, f)

    n_changes = y.sum()
    print(f"  [{split}] {len(pairs)} pairs | changes: {n_changes} ({100*n_changes/len(pairs):.1f}%)")


def main(data_dir: Path, difficulty: str = "all") -> None:
    extractor = EmbeddingExtractor()

    print(f"Loading training pairs (difficulty={difficulty})...")
    train_pairs = load_all_pairs(data_dir, split="train", difficulty=difficulty)
    print(f"  {len(train_pairs)} pairs loaded")

    print(f"\nLoading validation pairs (difficulty={difficulty})...")
    val_pairs = load_all_pairs(data_dir, split="validation", difficulty=difficulty)
    print(f"  {len(val_pairs)} pairs loaded")

    print("\nExtracting Stream A (stylometric)...")
    X_train_stylo = extract_stream_a(train_pairs)
    X_val_stylo   = extract_stream_a(val_pairs)

    print("\nExtracting Stream B (embeddings)...")
    X_train_emb, emb_a_train, emb_b_train = extract_stream_b(train_pairs, extractor)
    X_val_emb,   emb_a_val,   emb_b_val   = extract_stream_b(val_pairs,   extractor)

    print("\nSaving feature matrices...")
    save_split(train_pairs, X_train_stylo, X_train_emb, emb_a_train, emb_b_train, "train")
    save_split(val_pairs,   X_val_stylo,   X_val_emb,   emb_a_val,   emb_b_val,   "val")

    feature_names = get_feature_names()
    with open(RESULTS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    print(f"\nDone. Shapes: stylo={X_train_stylo.shape}, emb={X_train_emb.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="pan25-multi-author-analysis",
        help="Path to extracted PAN 2025 dataset root.",
    )
    parser.add_argument(
        "--difficulty",
        default="all",
        choices=["easy", "medium", "hard", "all"],
        help="Which difficulty tier(s) to include. 'all' combines easy+medium+hard.",
    )
    args = parser.parse_args()
    main(Path(args.data_dir), difficulty=args.difficulty)
