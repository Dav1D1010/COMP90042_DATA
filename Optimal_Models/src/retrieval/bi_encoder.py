"""
Bi-Encoder: SentenceTransformer wrapper for encoding claims and evidence.

Design rationale
================

Why all-MiniLM-L6-v2 over ModernBERT?
    384d vs 768d: FAISS index is 1.8 GB vs 3.5 GB on T4.
    384d fits comfortably in Colab's 12 GB RAM.
    MiniLM-L6 is a well-validated sentence embedding model fine-tuned on 1B+ pairs;
    ModernBERT (base) is a general-purpose MLM encoder that would require
    additional contrastive fine-tuning to match retrieval quality.

Why query / passage prefixes?
    all-MiniLM-L6-v2 was trained with instruction-tuned asymmetric encoding:
    "query: <claim>" vs "passage: <evidence>" produces distinct embedding
    subspaces that improve cross-encoding retrieval quality over symmetric
    encoding (no prefix). This mirrors the standard IR practice of treating
    queries and documents differently.

Why normalize embeddings?
    L2-normalised vectors allow cosine similarity via inner product.
    FAISS IndexFlatIP (inner product) on unit vectors = cosine similarity.
    This avoids the memory overhead of IndexFlatL2 (Euclidean) and gives
    semantically meaningful scores in [−1, 1].

Why batch_size=64 on T4?
    384d × 64 passages × 4 bytes (FP32) = 98 KB per batch — tiny.
    The encoder forward pass for 64 sequences of ~20 words each is
    well within T4's 16 GB VRAM budget. Even 128 would fit, but 64
    keeps GPU utilisation high without risking OOM.
"""

import numpy as np
from typing import List


class BiEncoder:
    """Wraps SentenceTransformer for asymmetric claim/evidence encoding.

    Parameters
    ----------
    model_name : str
        HuggingFace SentenceTransformer model identifier.
        Default ``"all-MiniLM-L6-v2"`` (384-dim, fast on T4).
    device : str
        PyTorch device string (``"cuda"``, ``"cpu"``, ``"mps"``).
        Default ``"cuda"``.
    """

    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cuda",
    ):
        self._model_name = model_name
        self._device = device
        self._model = self._load_model(model_name, device)

    @staticmethod
    def _load_model(model_name: str, device: str):
        """Import and instantiate SentenceTransformer, installing if needed."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            import subprocess
            import sys
            subprocess.check_call([
                sys.executable, "-m", "uv", "pip", "install",
                "sentence-transformers",
            ])
            from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name, device=device)

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors produced by this encoder."""
        return self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: int = 64,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """Encode a list of texts into a (N, D) float32 array.

        Parameters
        ----------
        texts : list of str
        normalize : bool
            L2-normalise each vector so that inner product = cosine.
        batch_size : int
            Mini-batch size for the forward pass.
        show_progress_bar : bool
            Show a tqdm progress bar.

        Returns
        -------
        numpy.ndarray  shape ``(len(texts), dim)``, dtype ``float32``.
        """
        return self._model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )

    def encode_queries(
        self,
        claims: List[str],
        normalize: bool = True,
        batch_size: int = 64,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """Encode claims with the ``"query: "`` prefix.

        See `encode` for parameter details.
        """
        prefixed = [f"{self.QUERY_PREFIX}{c}" for c in claims]
        return self.encode(
            prefixed,
            normalize=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

    def encode_passages(
        self,
        passages: List[str],
        normalize: bool = True,
        batch_size: int = 64,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """Encode evidence passages with the ``"passage: "`` prefix.

        See `encode` for parameter details.
        """
        prefixed = [f"{self.PASSAGE_PREFIX}{p}" for p in passages]
        return self.encode(
            prefixed,
            normalize=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
