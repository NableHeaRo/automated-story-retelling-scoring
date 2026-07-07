"""
Semantic similarity matrix and automated outcome measures (ORS, OORS)
for the auto-SR pipeline.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import spearmanr


def build_similarity_matrix(story_embeddings: np.ndarray, recall_embeddings: np.ndarray) -> np.ndarray:
    """Build the Fisher-z-transformed semantic similarity matrix.

    For each pair of (story segment, recall segment) embeddings, a
    Spearman correlation is computed and then Fisher z-transformed.

    Parameters
    ----------
    story_embeddings : np.ndarray, shape (n_story_segments, dim)
    recall_embeddings : np.ndarray, shape (n_recall_segments, dim)

    Returns
    -------
    np.ndarray, shape (n_story_segments, n_recall_segments)
        Row i, column j is the Fisher-z semantic similarity between story
        segment i and recall segment j.
    """
    n_story = story_embeddings.shape[0]
    n_recall = recall_embeddings.shape[0]

    z = np.zeros((n_story, n_recall))
    for i in range(n_story):
        for j in range(n_recall):
            corr, _ = spearmanr(story_embeddings[i], recall_embeddings[j])
            z[i, j] = corr

    # Fisher z-transform; clip to avoid +/-inf at r = +/-1
    return np.arctanh(np.clip(z, -0.999999, 0.999999))


def compute_ors_oors(z_matrix: np.ndarray) -> Tuple[float, float]:
    """Compute the two published automated outcome measures.

    ORS (Overall Recall Score): the average, across story segments (rows),
    of the maximum Fisher-z similarity value in that row -- i.e., how well
    each story segment was recalled *somewhere* in the retelling,
    regardless of order.

    OORS (Original-Order Recall Score): the average of the diagonal
    elements of the similarity matrix -- i.e., how well each story segment
    was recalled *in its original position*.

    Parameters
    ----------
    z_matrix : np.ndarray, shape (n_segments, n_segments)
        Fisher-z similarity matrix from `build_similarity_matrix`. Must be
        square (same number of story and recall segments; 10x10 in the
        published study).

    Returns
    -------
    (ORS, OORS) : tuple of float
    """
    if z_matrix.shape[0] != z_matrix.shape[1]:
        raise ValueError(
            "compute_ors_oors expects a square matrix (equal number of "
            f"story and recall segments); got shape {z_matrix.shape}."
        )

    ors = float(np.mean(np.max(z_matrix, axis=1)))
    oors = float(np.mean(np.diag(z_matrix)))
    return ors, oors
