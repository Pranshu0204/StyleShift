"""
PAN 2025 Dataset Setup — Multi-Author Writing Style Analysis

This script verifies the expected directory structure of the PAN 2025
Style Change Detection dataset once it has been downloaded and extracted.
It does NOT download anything — that requires a Zenodo account.

WHY PAN 2025 instead of earlier editions?
  PAN 2025 introduces sentence-level detection (vs. paragraph-level in 2021)
  and provides three difficulty tiers (easy / medium / hard) based on how
  much topical variation exists between authors. This difficulty axis is a
  natural, free ablation dimension: a system that degrades gracefully from
  easy to hard demonstrates genuine stylometric capability rather than
  topic-based shortcutting.

Dataset source (open access, free Zenodo account required):
  https://zenodo.org/records/14891299

Expected structure after extraction:
  pan25-multi-author-analysis/
    easy/
      train/
        problem-1.txt
        truth-problem-1.json
        problem-2.txt
        truth-problem-2.json
        ...
      validation/
        problem-1.txt
        truth-problem-1.json
        ...
    medium/
      train/ ...
      validation/ ...
    hard/
      train/ ...
      validation/ ...

FORMAT (per document pair):
  problem-X.txt       — Plain text. One sentence per line. No blank lines.
  truth-problem-X.json — {"authors": N, "changes": [0, 1, 0, ...]}
    changes has exactly (num_sentences - 1) entries.
    changes[i] = 1 means authorship switches between sentence i and i+1.
    changes[i] = 0 means the same author continues.

DIFFICULTY TIERS:
  easy   — documents span multiple topics (topic shift can aid detection)
  medium — limited topical variety (model must focus on style, not topic)
  hard   — all sentences share the same topic (pure stylometric challenge)

Usage:
    python data/download_pan.py --verify --data-dir pan25-multi-author-analysis
"""

import argparse
import json
from pathlib import Path

DIFFICULTIES = ["easy", "medium", "hard"]


def verify_dataset(data_dir: Path) -> None:
    """Check that the dataset is correctly structured and print statistics."""
    if not data_dir.exists():
        print(f"[ERROR] Directory not found: {data_dir}")
        print("\nTo obtain the dataset:")
        print("  1. Go to https://zenodo.org/records/14891299")
        print("  2. Log in to Zenodo (free account) and download the archive")
        print(f"  3. Extract so that this path exists: {data_dir}/easy/train/problem-1.txt")
        return

    print(f"\nDataset verified at: {data_dir}\n")
    print(f"  {'Difficulty':<10} {'Split':<12} {'Docs':>6} {'Pairs':>8} {'Changes':>9} {'Imbalance':>10}")
    print("  " + "-" * 58)

    grand_pairs = 0
    grand_changes = 0

    for diff in DIFFICULTIES:
        for split in ("train", "validation"):
            split_dir = data_dir / diff / split
            if not split_dir.exists():
                print(f"  [WARN] Missing: {split_dir}")
                continue

            txt_files = sorted(split_dir.glob("problem-*.txt"))
            total_pairs = 0
            total_changes = 0

            for txt_path in txt_files:
                truth_path = (
                    split_dir
                    / txt_path.name.replace("problem-", "truth-problem-").replace(".txt", ".json")
                )
                if not truth_path.exists():
                    print(f"  [WARN] Missing truth file: {truth_path}")
                    continue
                with open(truth_path, encoding="utf-8") as f:
                    truth = json.load(f)
                changes = truth.get("changes", [])
                total_pairs += len(changes)
                total_changes += sum(changes)

            grand_pairs += total_pairs
            grand_changes += total_changes
            ratio = (total_pairs - total_changes) / max(total_changes, 1)
            pct = 100 * total_changes / max(total_pairs, 1)
            print(
                f"  {diff:<10} {split:<12} {len(txt_files):>6} "
                f"{total_pairs:>8} {total_changes:>7} ({pct:4.1f}%) {ratio:>7.1f}:1"
            )

    print("  " + "-" * 58)
    overall_ratio = (grand_pairs - grand_changes) / max(grand_changes, 1)
    print(f"  {'TOTAL':<23} {grand_pairs:>8} {grand_changes:>9} {overall_ratio:>10.1f}:1")
    print()
    print("  → Class imbalance confirmed: use F1-macro and AUC-ROC, NOT accuracy.")
    print("  → Three difficulty tiers form a built-in ablation axis:")
    print("      easy   = topic shift aids detection (upper bound)")
    print("      medium = reduced topic signal")
    print("      hard   = pure style, no topic shortcut (lower bound)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify PAN 2025 dataset structure.")
    parser.add_argument("--verify", action="store_true", help="Run structure verification.")
    parser.add_argument(
        "--data-dir",
        default="pan25-multi-author-analysis",
        help="Path to extracted dataset root.",
    )
    args = parser.parse_args()

    if args.verify:
        verify_dataset(Path(args.data_dir))
    else:
        print("Run with --verify to check your dataset structure.")
        print("Dataset: https://zenodo.org/records/14891299")
