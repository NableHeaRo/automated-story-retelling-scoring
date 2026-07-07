"""
Plotting utilities for the auto-SR pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe default for headless / batch use
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_similarity_matrix(
    z_matrix: np.ndarray,
    title: str = "Semantic Similarity Matrix",
    save_path: Optional[str] = None,
    cmap: str = "GnBu",
):
    """Plot the story-by-recall semantic similarity (Fisher z) heatmap.

    Mirrors Figure 1C / Figure 2 of the accompanying paper: rows are story
    segments, columns are recall segments, and the diagonal (used for
    OORS) is outlined.

    Parameters
    ----------
    z_matrix : np.ndarray, shape (n_segments, n_segments)
    title : str
    save_path : str, optional
        If given, the figure is saved to this path (e.g. "sub01_matrix.png")
        instead of only being returned.
    cmap : str

    Returns
    -------
    (fig, ax) : matplotlib Figure and Axes
    """
    n_seg = z_matrix.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(z_matrix, cmap=cmap, aspect="equal")

    ax.set_xticks(range(n_seg))
    ax.set_yticks(range(n_seg))
    ax.set_xticklabels([str(i + 1) if (i + 1) % 3 == 0 else "" for i in range(n_seg)])
    ax.set_yticklabels([str(i + 1) if (i + 1) % 3 == 0 else "" for i in range(n_seg)])

    for i in range(n_seg):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2))

    ax.set_title(title)
    ax.set_xlabel("Recall segments")
    ax.set_ylabel("Story segments")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fisher z (Spearman)")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)

    return fig, ax


def plot_serial_position(
    diagonals: Dict[str, np.ndarray],
    groups: Optional[Dict[str, str]] = None,
    group_colors: Optional[Dict[str, str]] = None,
    title: str = "Recall Accuracy Across Story Segments",
    save_path: Optional[str] = None,
):
    """Plot diagonal (OORS-component) values across serial position.

    This reproduces the style of Figure 7 in the accompanying paper:
    individual participant lines (light) plus group mean lines (bold).

    Parameters
    ----------
    diagonals : dict
        Mapping of participant ID -> 1D array of per-segment diagonal
        Fisher-z values (length = n_segments).
    groups : dict, optional
        Mapping of participant ID -> group label (e.g., "HA" / "PWA"). If
        omitted, all participants are plotted in a single color with no
        group means.
    group_colors : dict, optional
        Mapping of group label -> matplotlib color. Defaults to a simple
        two-color palette if not provided.
    title : str
    save_path : str, optional

    Returns
    -------
    (fig, ax) : matplotlib Figure and Axes
    """
    if not diagonals:
        raise ValueError("`diagonals` is empty -- nothing to plot.")

    n_seg = len(next(iter(diagonals.values())))
    x = np.arange(1, n_seg + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    if groups is None:
        for pid, diag in diagonals.items():
            ax.plot(x, diag, color="steelblue", alpha=0.4, linewidth=1)
        mean_line = np.nanmean(np.vstack(list(diagonals.values())), axis=0)
        ax.plot(x, mean_line, color="steelblue", linewidth=4, label="Mean")
    else:
        default_colors = ["#2E7D32", "#C62828", "#1565C0", "#6A1B9A"]
        unique_groups = sorted(set(groups.values()))
        if group_colors is None:
            group_colors = {g: default_colors[i % len(default_colors)] for i, g in enumerate(unique_groups)}

        for pid, diag in diagonals.items():
            g = groups.get(pid)
            ax.plot(x, diag, color=group_colors.get(g, "gray"), alpha=0.25, linewidth=1)

        for g in unique_groups:
            arr = np.vstack([diag for pid, diag in diagonals.items() if groups.get(pid) == g])
            mean_line = np.nanmean(arr, axis=0)
            ax.plot(x, mean_line, color=group_colors[g], linewidth=4, label=f"{g} (mean)")

    ax.set_xlabel("Story Segment (Serial Position)")
    ax.set_ylabel("Diagonal Semantic Similarity (Fisher z)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)

    return fig, ax
