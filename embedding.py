"""
Sentence embedding utilities for the auto-SR pipeline.

Uses the Language-agnostic BERT Sentence Embedding (LaBSE) model
(Feng et al., 2022) via the `sentence-transformers` library.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


class LaBSEEmbedder:
    """Thin wrapper around the LaBSE sentence-transformer model.

    The underlying model is loaded lazily on first use and cached, since
    loading it is the slowest part of the pipeline and users typically
    score many recalls against the same story in one session.

    Parameters
    ----------
    model_name : str, default "sentence-transformers/LaBSE"
        Hugging Face model identifier. Only change this if you know what
        you are doing -- the published measures (ORS, OORS) were derived
        using LaBSE specifically.
    device : str, optional
        "cpu", "cuda", or "mps". If None, sentence-transformers will pick
        automatically.
    """

    _shared_model = None
    _shared_model_name: Optional[str] = None

    def __init__(self, model_name: str = "sentence-transformers/LaBSE", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model

        # Reuse a process-wide cached model if the name matches, so that
        # scoring many participants in a loop only pays the load cost once.
        if LaBSEEmbedder._shared_model is not None and LaBSEEmbedder._shared_model_name == self.model_name:
            self._model = LaBSEEmbedder._shared_model
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Install it with "
                "`pip install sentence-transformers`."
            ) from exc

        model = SentenceTransformer(self.model_name, device=self.device)
        LaBSEEmbedder._shared_model = model
        LaBSEEmbedder._shared_model_name = self.model_name
        self._model = model
        return self._model

    def embed(self, segments: List[str]) -> np.ndarray:
        """Encode a list of text segments into embedding vectors.

        Returns
        -------
        np.ndarray of shape (len(segments), 768)
        """
        model = self._load()
        return model.encode(segments, convert_to_numpy=True, show_progress_bar=False)
