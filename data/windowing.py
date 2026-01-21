"""
Sliding Window Segmentation — PAN 2025 Style Change Detection

PAN 2025 key difference from PAN 2021:
  Labels are at the SENTENCE level, not the paragraph level.
  Each problem-X.txt has one sentence per line. The truth JSON
  has changes[i] = 1 meaning the authorship switches between
  sentence i and sentence i+1. This removes the paragraph-to-window
  label mapping that PAN 2021 required.

WHY sliding windows instead of full-document classification?
  Full-document approaches lose position information — they cannot tell
  you WHERE in the document the switch happens. Pairwise block comparison
  localises the boundary to a specific region. Smaller windows give
  finer granularity but more noise. Larger windows are more stable but
  miss short injected passages. WINDOW_SIZE is a key hyperparameter.

WHY pairwise binary classification?
  We reduce the sequence-level problem to pairwise binary classification.
  For each consecutive pair (block_i, block_{i+1}) we predict:
    Y = 0  →  same author continues
    Y = 1  →  authorship switches (style change boundary)
  This is the standard formulation in PAN shared tasks and makes the
  problem tractable with standard classifiers including SVMs.

Label assignment for window pairs:
  The label of pair (block_i, block_{i+1}) is the change signal at the
  exact sentence boundary between the two blocks:
    label = changes[block_start[i+1] - 1]
  Within-block changes are a known limitation of fixed-size windowing
  — a research observation worth noting in any write-up.

PAN 2025 dataset structure:
  pan25-multi-author-analysis/
    easy/   train/ + validation/   (documents from varied topics)
    medium/ train/ + validation/   (limited topic variety)
    hard/   train/ + validation/   (all sentences share the same topic)
  The three difficulties form a natural ablation axis.
"""

import json
from pathlib import Path
from dataclasses import dataclass

import spacy

# ── Hyperparameter — defined here for easy tuning ──
# 3 sentences: fine-grained but noisy; 5 sentences: stable but coarse.
WINDOW_SIZE = 3  # sentences per block

# spaCy is used ONLY by sentences_from_text(), which is called by the UI
# (ui/app.py) to split user-pasted text. PAN 2025 documents are pre-split
# (one sentence per line) so spaCy is not needed for loading training data.
try:
    NLP = spacy.load("en_core_web_sm", exclude=["ner", "parser", "senter"])
    NLP.add_pipe("sentencizer")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    NLP = spacy.load("en_core_web_sm", exclude=["ner", "parser", "senter"])
    NLP.add_pipe("sentencizer")


@dataclass
class BlockPair:
    """One training example: a pair of consecutive text blocks with a label."""
    doc_id: str      # e.g. "easy/problem-42" — encodes difficulty + document
    block_idx: int   # index of the first block in the pair within this document
    block_a: str     # text of the first block (window_size sentences joined)
    block_b: str     # text of the second block
    label: int       # 0 = same author, 1 = style change at this boundary
    difficulty: str  # "easy", "medium", or "hard" — useful for per-difficulty eval


def sentences_from_text(text: str) -> list[str]:
    """
    Split arbitrary text into sentences via spaCy's sentencizer.
    Used by ui/app.py for user-pasted documents; NOT used for PAN 2025 loading.
    """
    doc = NLP(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def build_windows(sentences: list[str], window_size: int = WINDOW_SIZE) -> tuple[list[str], list[int]]:
    """
    Group sentences into non-overlapping windows of window_size.

    Returns:
      blocks:        list of strings, each block is window_size sentences joined
      block_starts:  sentence index where each block begins (used for label lookup)
    """
    blocks: list[str] = []
    block_starts: list[int] = []
    i = 0
    while i < len(sentences):
        blocks.append(" ".join(sentences[i: i + window_size]))
        block_starts.append(i)
        i += window_size
    return blocks, block_starts


def document_to_pairs(
    doc_id: str,
    sentences: list[str],
    change_labels: list[int],
    window_size: int = WINDOW_SIZE,
    difficulty: str = "easy",
) -> list[BlockPair]:
    """
    Convert one PAN 2025 document into a list of BlockPair training examples.

    change_labels[i] = 1 means authorship switches between sentence i and i+1.
    len(change_labels) == len(sentences) - 1  (confirmed for PAN 2025).

    Label for pair (block_i, block_{i+1}):
      The boundary between the two blocks falls between sentence
      (block_starts[i+1] - 1) and sentence block_starts[i+1].
      → label = change_labels[block_starts[i+1] - 1]
    """
    blocks, block_starts = build_windows(sentences, window_size)

    pairs: list[BlockPair] = []
    for i in range(len(blocks) - 1):
        boundary_idx = block_starts[i + 1] - 1  # last sentence of block_i
        label = int(change_labels[boundary_idx]) if boundary_idx < len(change_labels) else 0
        pairs.append(BlockPair(
            doc_id=doc_id,
            block_idx=i,
            block_a=blocks[i],
            block_b=blocks[i + 1],
            label=label,
            difficulty=difficulty,
        ))
    return pairs


def load_pan_document(txt_path: Path) -> tuple[list[str], list[int]]:
    """
    Load a single PAN 2025 document.

    Returns (sentences, change_labels):
      sentences:     list of individual sentences (one per line in the file)
      change_labels: list of 0/1 integers, length = len(sentences) - 1
    """
    with open(txt_path, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    truth_path = (
        txt_path.parent
        / txt_path.name.replace("problem-", "truth-problem-").replace(".txt", ".json")
    )
    with open(truth_path, encoding="utf-8") as f:
        truth = json.load(f)

    return sentences, truth.get("changes", [])


def load_all_pairs(
    data_dir: Path,
    split: str = "train",
    difficulty: str = "all",
    window_size: int = WINDOW_SIZE,
) -> list[BlockPair]:
    """
    Load all documents from a PAN 2025 split and return as BlockPair list.

    Args:
      data_dir:   path to pan25-multi-author-analysis/
      split:      "train" or "validation"
      difficulty: "easy", "medium", "hard", or "all" (combines all three).
                  Using "all" gives the broadest training set; evaluating
                  per-difficulty afterwards is a natural ablation axis.
      window_size: sentences per block (hyperparameter)
    """
    difficulties = ["easy", "medium", "hard"] if difficulty == "all" else [difficulty]
    all_pairs: list[BlockPair] = []

    for diff in difficulties:
        split_dir = data_dir / diff / split
        if not split_dir.exists():
            print(f"  [WARN] Not found: {split_dir}")
            continue

        txt_files = sorted(split_dir.glob("problem-*.txt"))
        for txt_path in txt_files:
            doc_id = f"{diff}/{txt_path.stem}"  # e.g. "easy/problem-42"
            try:
                sentences, change_labels = load_pan_document(txt_path)
                pairs = document_to_pairs(doc_id, sentences, change_labels, window_size, diff)
                all_pairs.extend(pairs)
            except Exception as exc:
                print(f"  [WARN] Skipping {doc_id}: {exc}")

    return all_pairs
