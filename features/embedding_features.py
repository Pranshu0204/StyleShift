"""
Stream B — Contextual Embedding Features

Uses a frozen sentence-transformer (all-mpnet-base-v2) to produce 768-dim
dense embeddings for each text block.

Design decision — why frozen?
  Using a frozen encoder means we treat it as a general-purpose semantic
  feature extractor. Fine-tuning on a small dataset like PAN risks
  overfitting and would make the embedding less general — the encoder
  would start memorizing training documents rather than learning
  transferable style representations. Fine-tuning is reserved for the
  Siamese MLP head (Stage 2, Component B), which has far fewer parameters.

Why all-mpnet-base-v2?
  It achieves state-of-the-art semantic similarity performance among
  sentence-transformer models while outputting L2-normalized 768-dim
  vectors, making cosine similarity computation trivial (dot product).
"""

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_DIM = 768


class EmbeddingExtractor:
    """Wraps the frozen sentence-transformer for batch encoding."""

    def __init__(self):
        # model.encode() never calls model.train() — inherently frozen
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def encode(self, texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """
        Returns (N, 768) float32 array of L2-normalized embeddings.
        L2-normalized: cosine similarity reduces to a dot product.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def cosine_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
        """
        Batch cosine similarity for (N, 768) arrays.
        A sudden drop in cosine similarity between consecutive blocks is a
        strong signal of a style or topic shift — this is used as a direct
        weak-signal feature in the meta-learner.
        """
        return np.sum(emb_a * emb_b, axis=1)  # dot product of L2-normalized = cosine
