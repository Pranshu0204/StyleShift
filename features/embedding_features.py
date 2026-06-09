"""
Stream B — Contextual Embedding Features

Wraps a frozen all-mpnet-base-v2 sentence-transformer to produce
L2-normalised 768-dim embeddings per text block. The encoder is kept
frozen to avoid overfitting on the small PAN dataset; only the Siamese
MLP head (Component B) is trained.
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
