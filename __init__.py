"""
auto-SR: Automated Scoring of Story Retelling Performance
using Large Language Model sentence embeddings.

Reference
---------
This implementation accompanies:

    Yoo, Y. R., Lim, Y., & Sung, J. E. (2026). Automated scoring of story
    retelling performance in people with aphasia using large language
    models. American Journal of Speech-Language Pathology.

The segmentation / similarity-matrix approach is adapted from:

    Herrmann, B. (2025). Language-agnostic, automated assessment of
    listeners' speech recall using large language models. arXiv preprint
    arXiv:2503.01045.

Sentence embeddings are computed with the Language-agnostic BERT Sentence
Embedding (LaBSE) model:

    Feng, F., Yang, Y., Cer, D., Arivazhagan, N., & Wang, W. (2022).
    Language-agnostic BERT sentence embedding. arXiv preprint
    arXiv:2207.04672.

If you use this code, please cite all three references above.
"""

from .segmentation import segment_text
from .embedding import LaBSEEmbedder
from .similarity import build_similarity_matrix, compute_ors_oors
from .pipeline import score_retelling

__all__ = [
    "segment_text",
    "LaBSEEmbedder",
    "build_similarity_matrix",
    "compute_ors_oors",
    "score_retelling",
]

__version__ = "0.1.0"
