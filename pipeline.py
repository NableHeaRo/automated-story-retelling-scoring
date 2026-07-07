"""
High-level convenience function for scoring a single story retelling.
"""

from __future__ import annotations

from typing import Optional

from .segmentation import segment_text
from .embedding import LaBSEEmbedder
from .similarity import build_similarity_matrix, compute_ors_oors
from .visualize import plot_similarity_matrix


def score_retelling(
    story_text: str,
    recall_text: str,
    num_segments: int = 10,
    overlap_ratio: float = 0.2,
    embedder: Optional[LaBSEEmbedder] = None,
    plot: bool = False,
    plot_save_path: Optional[str] = None,
    plot_title: str = "Semantic Similarity Matrix",
):
    """Score a single story retelling against the original story.

    This runs the full published pipeline: segmentation -> LaBSE
    embedding -> Spearman similarity -> Fisher z-transform -> ORS/OORS.

    Parameters
    ----------
    story_text : str
        The original story, as a single string (Korean or any of the
        100+ languages supported by LaBSE).
    recall_text : str
        A participant's retelling transcript, as a single string.
    num_segments : int, default 10
        Number of segments for both story and recall (published value: 10).
    overlap_ratio : float, default 0.2
        Overlap ratio between adjacent segments (published value: 0.2).
    embedder : LaBSEEmbedder, optional
        Reuse an existing embedder (recommended when scoring many
        recalls in a loop, to avoid reloading the model each time). If
        None, a new one is created (and cached process-wide).
    plot : bool, default False
        If True, generate a similarity-matrix heatmap figure.
    plot_save_path : str, optional
        Path to save the heatmap PNG, if `plot=True`.
    plot_title : str
        Title for the heatmap, if `plot=True`.

    Returns
    -------
    dict with keys:
        "ORS" : float
        "OORS" : float
        "z_matrix" : np.ndarray, shape (num_segments, num_segments)
        "fig" : matplotlib Figure or None (only if plot=True)
    """
    if embedder is None:
        embedder = LaBSEEmbedder()

    story_segments = segment_text(story_text, num_segments=num_segments, overlap_ratio=overlap_ratio)
    recall_segments = segment_text(recall_text, num_segments=num_segments, overlap_ratio=overlap_ratio)

    story_embed = embedder.embed(story_segments)
    recall_embed = embedder.embed(recall_segments)

    z_matrix = build_similarity_matrix(story_embed, recall_embed)
    ors, oors = compute_ors_oors(z_matrix)

    result = {"ORS": ors, "OORS": oors, "z_matrix": z_matrix, "fig": None}

    if plot:
        fig, _ax = plot_similarity_matrix(z_matrix, title=plot_title, save_path=plot_save_path)
        result["fig"] = fig

    return result
