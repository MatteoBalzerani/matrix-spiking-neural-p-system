"""
SNPS Weight Inspector
======================
Compares the ternary weight matrices from:
  - SVM / LogReg (your existing working method)
  - CNN twin (new method)

Run this AFTER you have both weight files available.
Prints detailed statistics and saves comparison plots.

Usage:
    python snps_weight_inspector.py

Edit the paths at the bottom to point to your actual files.
"""
import matplotlib
print(matplotlib.matplotlib_fname())
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ─────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────

def load_cnn_weights(json_path: str) -> dict:
    """Load ternary weights from snps_weights.json (CNN pipeline)."""
    with open(json_path) as f:
        data = json.load(f)
    return {
        "conv_kernels": np.array(data["conv_kernels"], dtype=np.int8),  # (8,1,3,3)
        "fc_weights":   np.array(data["fc_weights"],   dtype=np.int8),  # (10,1352) or (1352,10)
    }

def load_cnn_float_weights(npz_path: str) -> dict:
    """Load raw float weights from snps_weights_float.npz."""
    data = np.load(npz_path)
    return {
        "conv_kernels": data["conv_kernels"],  # (8,1,3,3)
        "fc_weights":   data["fc_weights"],    # (10,1352)
    }

def load_snps_ternary_matrix(path: str) -> np.ndarray:
    """
    Load the ternary FC matrix produced by your SVM/LogReg pipeline.
    Accepts .npy, .npz (key='arr_0' or first key), or .json.
    Shape expected: (1352, 10) or (10, 1352) — we normalize to (10, 1352).
    """
    p = Path(path)
    if p.suffix == ".npy":
        w = np.load(path)
    elif p.suffix == ".npz":
        data = np.load(path)
        key = list(data.keys())[0]
        w = data[key]
    elif p.suffix == ".json":
        with open(path) as f:
            w = np.array(json.load(f), dtype=np.int8)
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")

    # Normalize to (10, 1352)
    if w.shape == (1352, 10):
        w = w.T
    assert w.shape == (10, 1352), f"Unexpected shape: {w.shape}"
    return w.astype(np.int8)


# ─────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────

def ternary_stats(w: np.ndarray, name: str):
    """Print detailed statistics for a ternary weight matrix."""
    total = w.size
    pos   = (w ==  1).sum()
    neg   = (w == -1).sum()
    zero  = (w ==  0).sum()

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Shape        : {w.shape}")
    print(f"  Total weights: {total}")
    print(f"  +1 (excit.)  : {pos:6d}  ({pos/total*100:5.1f}%)")
    print(f"  -1 (inhib.)  : {neg:6d}  ({neg/total*100:5.1f}%)")
    print(f"   0 (silent)  : {zero:6d}  ({zero/total*100:5.1f}%)  ← sparsity")
    print(f"  +1 / -1 ratio: {pos/max(neg,1):.3f}  (ideal = 1.0)")
    print(f"  Active ratio : {(pos+neg)/total*100:.1f}%")

    if w.ndim == 2:
        print(f"\n  Per-row stats (one row = one output class):")
        print(f"  {'Class':>6}  {'  +1':>6}  {'  -1':>6}  {'   0':>6}  {'ratio+1/-1':>10}")
        for i in range(w.shape[0]):
            row  = w[i]
            rpos = (row ==  1).sum()
            rneg = (row == -1).sum()
            rzero= (row ==  0).sum()
            ratio = rpos / max(rneg, 1)
            print(f"  {i:>6}  {rpos:>6}  {rneg:>6}  {rzero:>6}  {ratio:>10.3f}")

    return {"pos": pos, "neg": neg, "zero": zero, "total": total}


def float_stats(w: np.ndarray, name: str):
    """Print statistics for float weights."""
    print(f"\n{'─'*55}")
    print(f"  {name} (float)")
    print(f"{'─'*55}")
    print(f"  Shape : {w.shape}")
    print(f"  Min   : {w.min():.4f}")
    print(f"  Max   : {w.max():.4f}")
    print(f"  Mean  : {w.mean():.4f}")
    print(f"  Std   : {w.std():.4f}")
    print(f"  % > 0 : {(w > 0).mean()*100:.1f}%")
    print(f"  % < 0 : {(w < 0).mean()*100:.1f}%")

    if w.ndim >= 2:
        flat = w.reshape(w.shape[0], -1)
        means = np.abs(flat).mean(axis=1)
        print(f"\n  Per-row mean(|w|): min={means.min():.4f}, "
              f"max={means.max():.4f}, mean={means.mean():.4f}")
        thresholds = 1.05 * means
        print(f"  k=1.05 thresholds: min={thresholds.min():.4f}, "
              f"max={thresholds.max():.4f}, mean={thresholds.mean():.4f}")
        # Simulate ternarize_threshold with k=1.05
        w_q = np.zeros_like(flat, dtype=np.int8)
        for r in range(flat.shape[0]):
            t = 1.05 * np.mean(np.abs(flat[r]))
            w_q[r] = np.where(flat[r] > t, 1, np.where(flat[r] < -t, -1, 0))
        pos  = (w_q ==  1).sum()
        neg  = (w_q == -1).sum()
        zero = (w_q ==  0).sum()
        total = w_q.size
        print(f"\n  Simulated ternarize(k=1.05):")
        print(f"    +1={pos} ({pos/total*100:.1f}%), "
              f"-1={neg} ({neg/total*100:.1f}%), "
              f"0={zero} ({zero/total*100:.1f}%)")
        print(f"    +1/-1 ratio: {pos/max(neg,1):.3f}")


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def plot_fc_comparison(w_cnn: np.ndarray, w_ref: np.ndarray,
                       w_cnn_float: np.ndarray = None,
                       save_path: str = "weight_comparison.png"):
    """
    Side-by-side comparison plots:
      Row 1: ternary weight heatmaps (CNN vs reference)
      Row 2: per-class +1/-1/0 bar charts
      Row 3: float weight distribution histogram (CNN only, if provided)
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("FC Weight Comparison: CNN vs Reference (SVM/LogReg)", fontsize=14)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 1: Heatmaps ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    for ax, w, title in [(ax1, w_cnn, "CNN ternary (10×1352)"),
                         (ax2, w_ref, "Reference ternary (10×1352)")]:
        im = ax.imshow(w, aspect="auto", cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Neuron index (1352)")
        ax.set_ylabel("Class")
        ax.set_yticks(range(10))
        plt.colorbar(im, ax=ax, ticks=[-1, 0, 1], fraction=0.02)

    # ── Row 2: Per-class +1/-1/0 bars ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    classes = list(range(10))
    for ax, w, title in [(ax3, w_cnn, "CNN — per-class weight counts"),
                         (ax4, w_ref, "Reference — per-class weight counts")]:
        pos_counts  = [(w[c] ==  1).sum() for c in classes]
        neg_counts  = [(w[c] == -1).sum() for c in classes]
        zero_counts = [(w[c] ==  0).sum() for c in classes]
        x = np.arange(10)
        w_bar = 0.25
        ax.bar(x - w_bar, pos_counts,  w_bar, label="+1", color="tomato")
        ax.bar(x,         neg_counts,  w_bar, label="-1", color="steelblue")
        ax.bar(x + w_bar, zero_counts, w_bar, label=" 0", color="lightgray")
        ax.set_title(title)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        ax.legend()

    # ── Row 3: Float weight histogram (CNN) ──
    if w_cnn_float is not None:
        ax5 = fig.add_subplot(gs[2, 0])
        ax6 = fig.add_subplot(gs[2, 1])

        flat = w_cnn_float.flatten()
        ax5.hist(flat, bins=100, color="steelblue", edgecolor="none")
        ax5.set_title("CNN float weight distribution (all FC weights)")
        ax5.set_xlabel("Weight value")
        ax5.set_ylabel("Count")
        ax5.axvline(0, color="red", linestyle="--", linewidth=1)

        # Per-row mean(|w|) distribution — shows how thresholds vary
        flat2d    = w_cnn_float.reshape(w_cnn_float.shape[0], -1)
        row_means = np.abs(flat2d).mean(axis=1)
        ax6.bar(range(len(row_means)), row_means, color="steelblue")
        ax6.set_title("CNN float: mean(|w|) per output class\n"
                      "→ adaptive threshold = 1.05 × this value")
        ax6.set_xlabel("Class")
        ax6.set_ylabel("mean(|w|)")
        ax6.set_xticks(range(len(row_means)))

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {save_path}")


def plot_conv_kernels(w_cnn_ternary: np.ndarray,
                      w_cnn_float:   np.ndarray = None,
                      save_path: str = "conv_kernels.png"):
    """
    Visualize the 8 ternary conv kernels side by side,
    with float kernels below if provided.
    """
    n_rows = 2 if w_cnn_float is not None else 1
    fig, axes = plt.subplots(n_rows, 8, figsize=(16, 4 * n_rows))
    fig.suptitle("Conv Kernels (8 filters, 3×3)", fontsize=13)

    if n_rows == 1:
        axes = [axes]

    for k in range(8):
        kernel_t = w_cnn_ternary[k, 0]
        pos = (kernel_t == 1).sum()
        neg = (kernel_t == -1).sum()
        axes[0][k].imshow(kernel_t, cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
        axes[0][k].set_title(f"K{k}\n+1={pos} -1={neg}", fontsize=9)
        axes[0][k].axis("off")

        if w_cnn_float is not None:
            kernel_f = w_cnn_float[k, 0]
            axes[1][k].imshow(kernel_f, cmap="bwr", vmin=kernel_f.min(), vmax=kernel_f.max(),
                              interpolation="nearest")
            axes[1][k].set_title(f"float\n[{kernel_f.min():.2f},{kernel_f.max():.2f}]", fontsize=9)
            axes[1][k].axis("off")

    if n_rows > 1:
        axes[0][0].set_ylabel("Ternary", fontsize=10)
        axes[1][0].set_ylabel("Float",   fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Conv kernel plot saved → {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(
        cnn_json_path:      str,   # snps_weights.json from CNN pipeline
        cnn_float_npz_path: str,   # snps_weights_float.npz from CNN pipeline
        ref_matrix_path:    str,   # ternary FC matrix from SVM or LogReg
):
    print("=" * 55)
    print("  SNPS Weight Inspector")
    print("=" * 55)

    # ── Load ──
    print("\nLoading weights...")
    cnn_weights   = load_cnn_weights(cnn_json_path)
    cnn_float     = load_cnn_float_weights(cnn_float_npz_path)
    ref_fc        = load_snps_ternary_matrix(ref_matrix_path)

    cnn_fc_ternary   = cnn_weights["fc_weights"]     # (10, 1352) int8
    cnn_conv_ternary = cnn_weights["conv_kernels"]   # (8,1,3,3) int8
    cnn_fc_float     = cnn_float["fc_weights"]       # (10, 1352) float
    cnn_conv_float   = cnn_float["conv_kernels"]     # (8,1,3,3) float

    # Ensure FC is (10, 1352)
    if cnn_fc_ternary.shape == (1352, 10):
        cnn_fc_ternary = cnn_fc_ternary.T
    if cnn_fc_float.shape == (1352, 10):
        cnn_fc_float = cnn_fc_float.T

    # ── Stats: CNN ternary ──
    print("\n\n════ CNN TERNARY WEIGHTS ════")
    ternary_stats(cnn_conv_ternary.reshape(8, -1), "Conv kernels (ternary)")
    ternary_stats(cnn_fc_ternary, "FC weights (ternary)")

    # ── Stats: CNN float ──
    print("\n\n════ CNN FLOAT WEIGHTS (before ternarization) ════")
    float_stats(cnn_conv_float.reshape(8, -1), "Conv kernels (float)")
    float_stats(cnn_fc_float,                  "FC weights (float)")

    # ── Stats: Reference (SVM/LogReg) ──
    print("\n\n════ REFERENCE TERNARY WEIGHTS (SVM/LogReg) ════")
    ternary_stats(ref_fc, "FC weights (SVM/LogReg ternary)")

    # ── Key comparison ──
    print("\n\n════ SUMMARY COMPARISON ════")
    cnn_pos  = (cnn_fc_ternary ==  1).sum()
    cnn_neg  = (cnn_fc_ternary == -1).sum()
    cnn_zero = (cnn_fc_ternary ==  0).sum()
    ref_pos  = (ref_fc ==  1).sum()
    ref_neg  = (ref_fc == -1).sum()
    ref_zero = (ref_fc ==  0).sum()
    total    = cnn_fc_ternary.size

    print(f"\n  {'Metric':<25} {'CNN':>10} {'Reference':>10}")
    print(f"  {'─'*47}")
    print(f"  {'% +1':<25} {cnn_pos/total*100:>9.1f}% {ref_pos/total*100:>9.1f}%")
    print(f"  {'% -1':<25} {cnn_neg/total*100:>9.1f}% {ref_neg/total*100:>9.1f}%")
    print(f"  {'% 0 (sparsity)':<25} {cnn_zero/total*100:>9.1f}% {ref_zero/total*100:>9.1f}%")
    print(f"  {'+1/-1 ratio':<25} {cnn_pos/max(cnn_neg,1):>10.3f} {ref_pos/max(ref_neg,1):>10.3f}")
    print(f"  {'active connections':<25} {cnn_pos+cnn_neg:>10} {ref_pos+ref_neg:>10}")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_fc_comparison(cnn_fc_ternary, ref_fc, cnn_fc_float,
                       save_path="weight_comparison.png")
    plot_conv_kernels(cnn_conv_ternary, cnn_conv_float,
                      save_path="conv_kernels.png")

    print("\nInspection complete.")


if __name__ == "__main__":
    import os
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    main(
        cnn_json_path      = os.path.join(PROJECT_ROOT, "cnn_twin_results", "snps_weights.json"),
        cnn_float_npz_path = os.path.join(PROJECT_ROOT, "cnn_twin_results", "snps_weights_float.npz"),
        ref_matrix_path    = os.path.join(PROJECT_ROOT, "ternary_matrix.npy"),
        # ↑ change this to the actual filename of your SVM or LogReg ternary matrix
    )