"""
Component B — Siamese Transformer Network

Shared encoder (frozen all-mpnet-base-v2) processes both blocks and produces
768-dim L2-normalised embeddings u and v. A trainable MLP head classifies
the pair using the InferSent interaction vector [|u-v|, u⊙v, cosine(u,v)].

Weight sharing ensures u and v live in the same metric space so their
difference is meaningful. The MLP learns which dimensions are discriminative
for style change detection.

Architecture: Linear(1537→256) → ReLU → Dropout(0.3) → Linear(256→1)
Loss: BCEWithLogitsLoss with pos_weight to handle class imbalance.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ──
EMBEDDING_DIM = 768
# Interaction vector: [|u-v|(768), u⊙v(768), cosine_sim(1)] = 1537
INTERACTION_DIM = 2 * EMBEDDING_DIM + 1
HIDDEN_DIM = 256
DROPOUT_RATE = 0.3

LEARNING_RATE = 2e-4
BATCH_SIZE = 32
NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3   # stop if val F1 doesn't improve for this many epochs
# Positive class weight: set higher to penalize missed style changes more
# Tune based on class imbalance ratio (see download_pan.py output)
POS_CLASS_WEIGHT = 3.0

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

ENCODER_MODEL = "all-mpnet-base-v2"


class PairDataset(Dataset):
    """
    Wraps precomputed embeddings for the Siamese network.
    We use precomputed embeddings (from pair_generator.py) rather than
    re-encoding on each batch — this speeds up training by ~10x since
    the encoder forward pass is the bottleneck.

    X_emb layout: [cosine_sim(1), abs_diff(768)] — we extract the raw
    embeddings from a separate file (X_{split}_emb_raw.npy) if available,
    or fall back to the diff + cosine representation.
    """

    def __init__(self, emb_a: np.ndarray, emb_b: np.ndarray, labels: np.ndarray):
        # emb_a and emb_b are (N, 768) raw embedding arrays
        self.emb_a = torch.tensor(emb_a, dtype=torch.float32)
        self.emb_b = torch.tensor(emb_b, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.emb_a[idx], self.emb_b[idx], self.labels[idx]


class SiameseMLP(nn.Module):
    """
    The trainable comparison head of the Siamese network.
    Takes pre-computed embeddings for two blocks and predicts P(style_change).
    """

    def __init__(self, interaction_dim: int = INTERACTION_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_dim, 1),
            # No sigmoid here — BCEWithLogitsLoss handles it for numerical stability
        )

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        u, v: (batch, 768) normalized embeddings
        Returns: (batch, 1) logit
        """
        abs_diff = torch.abs(u - v)                          # (batch, 768)
        hadamard = u * v                                     # (batch, 768) — co-activation
        cosine_sim = (u * v).sum(dim=1, keepdim=True)       # (batch, 1)

        # This interaction layer is from InferSent (Conneau et al. 2017)
        interaction = torch.cat([abs_diff, hadamard, cosine_sim], dim=1)  # (batch, 1537)
        return self.classifier(interaction)                  # (batch, 1)

    def predict_proba(self, u: torch.Tensor, v: torch.Tensor) -> np.ndarray:
        """Returns (N,) numpy array of P(style_change)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(u, v)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        return probs


def encode_texts_batch(texts: list[str], encoder: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    """Encode texts with the frozen encoder. Returns (N, 768) float32."""
    return encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def train_siamese(
    emb_a_train: np.ndarray, emb_b_train: np.ndarray, y_train: np.ndarray,
    emb_a_val: np.ndarray,   emb_b_val: np.ndarray,   y_val: np.ndarray,
) -> SiameseMLP:
    """Train the Siamese MLP head on precomputed embedding pairs."""
    from sklearn.metrics import f1_score

    train_dataset = PairDataset(emb_a_train, emb_b_train, y_train)
    val_dataset   = PairDataset(emb_a_val,   emb_b_val,   y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    model = SiameseMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    # BCEWithLogitsLoss: numerically more stable than BCE(Sigmoid(x))
    # pos_weight: scalar weight for positive (style change) class
    pos_weight = torch.tensor([POS_CLASS_WEIGHT], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
        model.train()
        total_loss = 0.0
        for u, v, labels in train_loader:
            u, v, labels = u.to(DEVICE), v.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(u, v).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for u, v, labels in val_loader:
                u, v = u.to(DEVICE), v.to(DEVICE)
                probs = torch.sigmoid(model(u, v)).squeeze(1).cpu().numpy()
                val_preds.extend((probs >= 0.5).astype(int).tolist())
                val_labels.extend(labels.numpy().tolist())

        val_f1 = f1_score(val_labels, val_preds, pos_label=1, zero_division=0)
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch:2d}/{NUM_EPOCHS} — loss: {avg_loss:.4f} — val_F1(change): {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), RESULTS_DIR / "component_b_siamese.pt")
            print(f"    → Best model saved (val_F1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={EARLY_STOPPING_PATIENCE})")
                break

    # Reload best checkpoint
    model.load_state_dict(torch.load(RESULTS_DIR / "component_b_siamese.pt", map_location=DEVICE, weights_only=True))
    print(f"\nSiamese training complete. Best val F1 (change): {best_val_f1:.4f}")
    return model


def load_siamese() -> SiameseMLP:
    model = SiameseMLP()
    model.load_state_dict(torch.load(RESULTS_DIR / "component_b_siamese.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    return model.to(DEVICE)
