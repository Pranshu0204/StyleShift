"""
Stage 4 — Gradio Demo Interface

Provides an interactive UI for intrinsic style change detection:
  1. Text area: paste any multi-author document
  2. Highlighted text: sentences colored by predicted authorship segment
  3. Probability line chart: P(style_change) across all consecutive block pairs
     — peaks indicate predicted boundaries
  4. Feature panel: top-5 stylometric features with the largest absolute
     difference at the predicted boundary (interpretable without SHAP)
  5. Confidence indicator per predicted boundary

Usage:
    python ui/app.py
    # Opens at http://localhost:7860
"""

import sys
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent / "features"))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from stylometric_features import extract_stylometric_features, get_feature_names
from embedding_features import EmbeddingExtractor
from component_a_svm import load_component_a, svm_predict_proba
from component_b_siamese import load_siamese, DEVICE
from component_c_ensemble import load_meta_learner, build_meta_features
from windowing import sentences_from_text, WINDOW_SIZE

RESULTS_DIR = Path("results")
FEATURE_NAMES = get_feature_names()

# ── Segment colors — alternating for visual clarity ──
SEGMENT_COLORS = ["#DDEEFF", "#FFE8CC", "#E8FFDD", "#FFE8F0", "#F0E8FF"]

# ── Load all components once at startup ──
print("Loading models...")
EXTRACTOR = EmbeddingExtractor()
SVM, IFOREST = load_component_a()
SIAMESE = load_siamese()
META = load_meta_learner()
print("Models loaded.")


def analyze_document(text: str, threshold: float = 0.5) -> tuple:
    """
    Process a pasted document through the full ensemble pipeline.
    Returns outputs for all Gradio components.
    """
    if not text or len(text.strip()) < 50:
        return None, "<p>Please enter a longer document (at least 50 characters).</p>", None, ""

    # 1. Split into sentences and create windows (blocks)
    all_sentences = sentences_from_text(text)
    if len(all_sentences) < 2:
        return None, "<p>Could not detect multiple sentences.</p>", None, ""

    # Group sentences into blocks of WINDOW_SIZE
    blocks = []
    block_sentence_ranges = []  # (start_idx, end_idx) for highlighting
    i = 0
    while i < len(all_sentences):
        window = all_sentences[i: i + WINDOW_SIZE]
        blocks.append(" ".join(window))
        block_sentence_ranges.append((i, min(i + WINDOW_SIZE, len(all_sentences))))
        i += WINDOW_SIZE

    if len(blocks) < 2:
        return None, "<p>Document too short — need at least two blocks.</p>", None, ""

    # 2. Extract features for each consecutive pair
    n_pairs = len(blocks) - 1
    stylo_feats = [extract_stylometric_features(b) for b in blocks]
    emb_feats   = EXTRACTOR.encode(blocks, show_progress=False)

    pair_probs = []
    pair_stylo_diffs = []

    for i in range(n_pairs):
        # Stream A: stylometric difference
        diff = np.abs(stylo_feats[i] - stylo_feats[i + 1])
        pair_stylo_diffs.append(diff)

        # Stream A: SVM probability
        svm_prob = svm_predict_proba(SVM, diff.reshape(1, -1))  # (1, 2)

        # Stream B: cosine similarity
        cosine = float(np.dot(emb_feats[i], emb_feats[i + 1]))

        # Stream B: Siamese
        import torch
        u = torch.tensor(emb_feats[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        v = torch.tensor(emb_feats[i + 1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        siamese_prob = float(SIAMESE.predict_proba(u, v)[0])

        # Ensemble
        X_meta = build_meta_features(svm_prob, np.array([siamese_prob]), np.array([cosine]))
        ensemble_prob = float(META.predict_proba(X_meta)[0, 1])
        pair_probs.append(ensemble_prob)

    pair_probs = np.array(pair_probs)
    change_mask = pair_probs >= threshold

    # 3. Assign segment IDs (increment at each predicted boundary)
    sentence_segment = []
    seg_id = 0
    for block_idx, (sent_start, sent_end) in enumerate(block_sentence_ranges):
        for _ in range(sent_start, sent_end):
            sentence_segment.append(seg_id)
        if block_idx < n_pairs and change_mask[block_idx]:
            seg_id += 1

    # 4. Build highlighted HTML
    html = build_highlighted_html(all_sentences, sentence_segment, pair_probs, change_mask)

    # 5. Probability line chart
    fig_prob = plot_probability_line(pair_probs, change_mask, threshold)

    # 6. Top-5 feature panel at the highest-confidence boundary
    feature_panel = ""
    if change_mask.any():
        top_boundary = int(np.argmax(pair_probs * change_mask))
        feature_panel = build_feature_panel(
            pair_stylo_diffs[top_boundary], pair_probs[top_boundary], top_boundary
        )

    return fig_prob, html, feature_panel


def build_highlighted_html(
    sentences: list[str],
    segment_ids: list[int],
    pair_probs: np.ndarray,
    change_mask: np.ndarray,
) -> str:
    """
    Render sentences as colored spans — alternating colors per segment.
    Predicted boundaries are marked with a vertical divider and confidence badge.
    """
    html_parts = [
        f'<div style="font-family: serif; font-size: 14px; line-height: 1.8; padding: 12px;">'
    ]

    current_seg = -1
    for i, (sent, seg_id) in enumerate(zip(sentences, segment_ids)):
        if seg_id != current_seg:
            if current_seg >= 0:
                html_parts.append("</span>")
            color = SEGMENT_COLORS[seg_id % len(SEGMENT_COLORS)]
            label = f"Segment {seg_id + 1}"
            html_parts.append(
                f'<span style="background-color: {color}; padding: 2px 0;" title="{label}">'
            )
            current_seg = seg_id

        html_parts.append(f" {sent}")

        # Add boundary marker after sentence if a change is predicted after this block
        block_idx = i // WINDOW_SIZE
        if block_idx < len(change_mask) and change_mask[block_idx] and (i + 1) % WINDOW_SIZE == 0:
            conf = pair_probs[block_idx]
            html_parts.append(
                f'<span style="display: inline-block; background: #E53935; color: white; '
                f'font-size: 11px; padding: 1px 5px; border-radius: 4px; margin: 0 4px;" '
                f'title="Predicted boundary (confidence {conf:.2f})"> '
                f'⚡ BOUNDARY ({conf:.2f})</span>'
            )

    if current_seg >= 0:
        html_parts.append("</span>")
    html_parts.append("</div>")
    return "".join(html_parts)


def plot_probability_line(
    pair_probs: np.ndarray,
    change_mask: np.ndarray,
    threshold: float,
) -> plt.Figure:
    """
    Line chart of P(style_change) across all consecutive block pairs.
    Peaks above the threshold indicate predicted style change boundaries.
    """
    x = np.arange(len(pair_probs))
    fig, ax = plt.subplots(figsize=(9, 3.5))

    ax.plot(x, pair_probs, color="#1565C0", linewidth=2, marker="o",
            markersize=5, label="P(style change)")
    ax.axhline(threshold, color="#E53935", linestyle="--", linewidth=1.2,
               label=f"Threshold = {threshold:.2f}")

    # Shade predicted boundaries
    for i, (prob, is_change) in enumerate(zip(pair_probs, change_mask)):
        if is_change:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.15, color="#E53935", label="_nolegend_")

    ax.set_xlabel("Block Pair Index")
    ax.set_ylabel("P(style change)")
    ax.set_title("Style Change Probability Across Document")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def build_feature_panel(
    stylo_diff: np.ndarray,
    confidence: float,
    boundary_idx: int,
) -> str:
    """
    Top-5 stylometric features with the largest absolute difference at
    the most confident predicted boundary. Makes the prediction interpretable
    without requiring SHAP computation.
    """
    # Get top-5 feature indices
    n_scalar = min(len(FEATURE_NAMES), len(stylo_diff))
    top_indices = np.argsort(stylo_diff[:n_scalar])[-5:][::-1]

    lines = [
        f"<b>Top-5 features at boundary {boundary_idx} (confidence: {confidence:.2f})</b>",
        "<table style='font-size:13px; border-collapse:collapse; margin-top:6px;'>",
        "<tr><th align='left'>Feature</th><th align='right'>|Diff|</th></tr>",
    ]
    for idx in top_indices:
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feat_{idx}"
        val = float(stylo_diff[idx])
        bar_width = int(val * 100 / (stylo_diff[:n_scalar].max() + 1e-8))
        lines.append(
            f"<tr><td>{name}</td>"
            f"<td align='right'>{val:.4f} "
            f"<span style='display:inline-block;width:{bar_width}px;"
            f"height:10px;background:#1565C0;vertical-align:middle;'></span></td></tr>"
        )
    lines.append("</table>")
    return "".join(lines)


EXAMPLE_DOC = (
    "The history of artificial intelligence begins in antiquity, with myths, stories, "
    "and rumors of artificial beings endowed with intelligence or consciousness. "
    "The seeds of modern AI were planted by classical philosophers who attempted to describe "
    "the process of human thinking as the mechanical manipulation of symbols.\n"
    "lol ok so basically what happened is some really smart dudes decided hey lets make computers "
    "think like humans and then they spent like 50 years failing at it and running out of money. "
    "it was pretty bad honestly and everyone called it the AI winter which is kind of dramatic "
    "but whatever.\n"
    "The resurgence of interest in neural networks during the 1980s, driven by the backpropagation "
    "algorithm, marked a significant turning point. Researchers demonstrated that multilayer "
    "perceptrons could learn complex non-linear functions, opening new possibilities for "
    "pattern recognition and machine learning applications in various domains."
)

with gr.Blocks(title="StyleShift — Style Change Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ✍️ StyleShift: Intrinsic Style Change Detection
        Detects where authorship switches within a single document — **without** reference texts.
        Architecture: SVM (stylometric) + Siamese Transformer + Meta-Learner ensemble.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Paste your document",
                placeholder="Paste a multi-paragraph document here...",
                lines=12,
                value=EXAMPLE_DOC,
            )
            with gr.Row():
                threshold_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Decision Threshold",
                    info="Lower = more sensitive to style changes",
                )
                analyze_btn = gr.Button("Detect Style Changes", variant="primary")

        with gr.Column(scale=3):
            prob_plot = gr.Plot(label="Style Change Probability")
            highlighted_html = gr.HTML(label="Highlighted Document")

    feature_panel = gr.HTML(label="Top Features at Predicted Boundary")

    analyze_btn.click(
        fn=analyze_document,
        inputs=[text_input, threshold_slider],
        outputs=[prob_plot, highlighted_html, feature_panel],
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
