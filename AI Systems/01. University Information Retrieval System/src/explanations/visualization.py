import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap


def plot_ig_heatmap(doc_explanation, max_tokens=15):

    ig = doc_explanation["explanation"]["ig"]

    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 2, 0.2])

    cmap = get_cmap("bwr")
    norm = Normalize(vmin=-1, vmax=1)

    # ======================================================
    # 1. KEY CONCEPTS (MAIN EXPLANATION)
    # ======================================================
    ax_text = fig.add_subplot(gs[0])
    ax_text.axis("off")

    ax_text.set_title(
        f"Document: {doc_explanation['doc_id']}  |  "
        f"F: {doc_explanation['faithfulness']:.3f}  |  "
        f"S: {doc_explanation['sufficiency']:.3f}  |  "
        f"C: {doc_explanation['consistency']:.3f}",
        fontsize=13
    )

    # ======================================================
    # 2. IG BAR PLOT (CLEAN TOKENS)
    # ======================================================
    ax_bar = fig.add_subplot(gs[1])

    if ig is None or len(ig) == 0:
        ax_bar.axis("off")
        return fig

    # ---- CLEAN TOKENS (same logic as engine) ----
    def is_clean(w):
        return (
            len(w) > 3 and
            w.isalpha() and
            not w.endswith((
                "dir", "dır", "tir", "tır",
                "lar", "ler", "ları", "leri",
                "da", "de", "ta", "te",
                "si", "sı", "su", "sü",
                "lu", "lü", "lı", "li"
            ))
        )

    filtered = [(w, s) for w, s in ig if is_clean(w)]

    if not filtered:
        ax_bar.axis("off")
        return fig

    top_ig = sorted(filtered, key=lambda x: abs(x[1]), reverse=True)[:max_tokens]

    tokens = [t for t, _ in top_ig]
    scores = np.array([s for _, s in top_ig])

    scores = scores / (np.max(np.abs(scores)) + 1e-8)
    colors = [cmap(norm(s)) for s in scores]

    ax_bar.bar(range(len(tokens)), scores, color=colors)

    ax_bar.set_xticks(range(len(tokens)))
    ax_bar.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)

    ax_bar.set_ylim(-1, 1)
    ax_bar.set_ylabel("Attribution")

    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # ======================================================
    # 3. COLORBAR
    # ======================================================
    ax_cbar = fig.add_subplot(gs[2])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax_cbar, orientation="horizontal")
    cbar.set_label("Token Attribution (Negative → Positive)")
    cbar.set_ticks([-1, 0, 1])

    return fig