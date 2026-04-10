import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Dict, FrozenSet, List


def plot_manifolds(
    X: np.ndarray, Z: np.ndarray, hops: Dict[int, FrozenSet[int]], corpus: List[str]
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Projections specifically for visualization scaling
    X_2d = PCA(n_components=2).fit_transform(X) if X.shape[1] > 2 else X
    Z_2d = Z[:, :2] if Z.shape[1] >= 2 else np.hstack([Z, np.zeros((Z.shape[0], 1))])
    Z_1d = Z[:, 0]

    colors = ["#cccccc"] * len(corpus)
    hop_colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0"]

    for hop_idx, nodes in hops.items():
        color = hop_colors[hop_idx % len(hop_colors)]
        for node in nodes:
            colors[node] = color

    def annotate_points(ax, coords):
        for i, _ in enumerate(corpus):
            ax.annotate(f"D{i}", (coords[i, 0] + 0.02, coords[i, 1] + 0.02), fontsize=9)

    axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=100, edgecolors="k")
    axes[0].set_title("Semantic Manifold ($\mathcal{X}$) - 2D PCA")
    annotate_points(axes[0], X_2d)

    axes[1].scatter(Z_2d[:, 0], Z_2d[:, 1], c=colors, s=100, edgecolors="k")
    axes[1].set_title("Structural Manifold ($\mathcal{Z}$) - 2D")
    annotate_points(axes[1], Z_2d)

    axes[2].scatter(Z_1d, np.zeros_like(Z_1d), c=colors, s=100, edgecolors="k")
    axes[2].set_title("Structural Manifold ($\mathcal{Z}$) - 1D Bottleneck")
    axes[2].get_yaxis().set_visible(False)
    for i, _ in enumerate(corpus):
        axes[2].annotate(f"D{i}", (Z_1d[i] + 0.01, 0.01), fontsize=9)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=hop_colors[h % len(hop_colors)],
            markersize=10,
            label=f"Hop {h}",
        )
        for h in hops.keys()
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#cccccc",
            markersize=10,
            label="Unvisited",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(hops) + 1,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout()
    plt.savefig("manifold_visualization.png", bbox_inches="tight")
    print("\n[+] Visualization saved to 'manifold_visualization.png'")
