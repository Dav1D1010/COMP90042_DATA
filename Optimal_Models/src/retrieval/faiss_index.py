"""
FAISS index builder and query engine for evidence retrieval.

Design rationale
================

Why IndexFlatIP?
    Exact inner-product search on L2-normalised vectors = cosine similarity.
    1.2 M × 384 d × 4 bytes (FP32) = 1.8 GB — fits comfortably in RAM.
    No approximation needed at this scale, so we keep perfect recall.

Why not IVF / PQ compression?
    Exact search is feasible and more accurate.  IVF requires training,
    adds latency from coarse→fine search, and loses recall (typically
    95–98 % vs 100 %).  PQ (product quantisation) compresses to 1–2 bytes
    per dimension but degrades similarity fidelity.  Not worth the
    complexity when exact fits in 2 GB.

Why save index and metadata separately?
    FAISS indices are binary blobs (NumPy memmap-friendly).  Metadata
    (evidence id + text for 1.2 M passages) is ~100 MB of JSON — too
    large to embed inside the index but small enough to hold in RAM.
    Keeping them separate lets us reload the index without re-parsing
    JSON, and reload metadata without touching FAISS.

Why FP32 storage?
    IndexFlatIP requires float32.  The bi-encoder can run in FP16 for
    the forward pass, but embeddings must be cast to float32 before
    adding to the index.

Why separate claim / evidence prefix encoding?
    BiEncoder uses "query: " and "passage: " prefixes.  Asymmetric
    encoding improves cross-encoding quality — the encoder projects
    claims and evidence into complementary subspaces rather than
    treating them as interchangeable texts.

Memory budget
─────────────
    1 208 827 passages × 384 d × 4 B  = 1.85 GB  (FAISS index)
    ~100 MB                                    (metadata JSON)
    ─────────────────────────────────────────
    ~2.0 GB  total                            (safe on T4's 12 GB)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


class EvidenceIndex:
    """Build, persist and query a FAISS IndexFlatIP over evidence embeddings.

    Parameters
    ----------
    dimension : int
        Embedding dimensionality (e.g. 384 for all-MiniLM-L6-v2).
        Derived from the bi-encoder — do not hardcode.
    index_path : Path or str, optional
        Where to save / load the FAISS index file.
        Default ``"evidence.index"`` in the current directory.
    """

    def __init__(
        self,
        dimension: int,
        index_path: Optional[Union[str, Path]] = None,
    ):
        self._dimension = dimension
        self._index_path = Path(index_path) if index_path else Path("evidence.index")

        self._index = None
        self._metadata: Dict[int, Dict[str, str]] = {}

    def build_index(
        self,
        embeddings: np.ndarray,
        evidence_dict: Dict[str, str],
    ) -> None:
        """Construct a FAISS IndexFlatIP from pre-computed embeddings.

        Parameters
        ----------
        embeddings : ndarray  shape ``(num_passages, dimension)``, float32.
            Must already be L2-normalised if you want cosine similarity.
        evidence_dict : dict[str, str]
            Mapping ``evidence_id → text`` in the same order as *embeddings*.
        """
        import faiss

        if embeddings.ndim != 2 or embeddings.shape[1] != self._dimension:
            raise ValueError(
                f"Expected embeddings of shape (N, {self._dimension}), "
                f"got {embeddings.shape}"
            )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(embeddings)

        self._metadata = {}
        for idx, (eid, text) in enumerate(evidence_dict.items()):
            self._metadata[idx] = {"id": eid, "text": text}

    def save(self, index_path: Optional[Union[str, Path]] = None) -> None:
        """Persist the FAISS index and metadata to disk.

        The index is written to ``index_path`` (default: the path given at
        construction).  Metadata is written to the same directory as
        ``evidence_metadata.json``.
        """
        import faiss

        if self._index is None:
            raise RuntimeError("No index to save — call build_index() first.")

        ipath = Path(index_path) if index_path else self._index_path
        ipath.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(ipath))

        meta_path = ipath.with_name("evidence_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(self._metadata, f)

    def load(self, index_path: Optional[Union[str, Path]] = None) -> None:
        """Load a previously saved FAISS index and its metadata."""
        import faiss

        ipath = Path(index_path) if index_path else self._index_path
        if not ipath.exists():
            raise FileNotFoundError(f"Index file not found: {ipath}")

        self._index = faiss.read_index(str(ipath))
        self._index_path = ipath

        meta_path = ipath.with_name("evidence_metadata.json")
        if meta_path.exists():
            with open(meta_path) as f:
                raw = json.load(f)
            self._metadata = {int(k): v for k, v in raw.items()}
        else:
            self._metadata = {}

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[Dict]:
        """Retrieve the top-*k* evidence passages for a single query.

        Parameters
        ----------
        query_embedding : ndarray  shape ``(dimension,)`` or ``(1, dimension)``.
            Must be L2-normalised for cosine-similarity scoring.
        k : int
            Number of passages to retrieve.

        Returns
        -------
        list[dict]
            Each dict has keys ``"id"``, ``"text"``, ``"score"``.
            Sorted by descending score (higher = more similar).
        """
        if self._index is None:
            raise RuntimeError("Index not built or loaded.")

        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self._dimension:
            raise ValueError(
                f"Query dimension {q.shape[1]} != index dimension {self._dimension}"
            )

        scores, indices = self._index.search(q, k)

        results = []
        for i in range(k):
            idx = int(indices[0][i])
            score = float(scores[0][i])
            meta = self._metadata.get(idx, {"id": str(idx), "text": ""})
            results.append({"id": meta["id"], "text": meta["text"], "score": score})

        return results

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
    ) -> List[List[Dict]]:
        """Retrieve top-*k* evidence for each query in a batch.

        Parameters
        ----------
        query_embeddings : ndarray  shape ``(num_queries, dimension)``.
        k : int

        Returns
        -------
        list of list of dict
            Outer list length = ``num_queries``; each inner list contains
            up to *k* results in the same format as :meth:`search`.
        """
        if self._index is None:
            raise RuntimeError("Index not built or loaded.")

        q = np.asarray(query_embeddings, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self._dimension:
            raise ValueError(
                f"Query dimension {q.shape[1]} != index dimension {self._dimension}"
            )

        scores_batch, indices_batch = self._index.search(q, k)

        all_results = []
        for n in range(q.shape[0]):
            results = []
            for i in range(k):
                idx = int(indices_batch[n][i])
                score = float(scores_batch[n][i])
                meta = self._metadata.get(idx, {"id": str(idx), "text": ""})
                results.append({"id": meta["id"], "text": meta["text"], "score": score})
            all_results.append(results)

        return all_results

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_built(self) -> bool:
        return self._index is not None

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index is not None else 0


def build_evidence_index(
    evidence_dict: Dict[str, str],
    bi_encoder,
    index_path: Union[str, Path],
    *,
    batch_size: int = 64,
) -> EvidenceIndex:
    """Encode all evidence passages and persist a FAISS index + metadata.

    This is the primary entry point for creating the retrieval backend
    from a raw ``evidence.json``-style dict.

    Parameters
    ----------
    evidence_dict : dict[str, str]
        ``{evidence_id: text}`` mapping for all ~1.2 M passages.
    bi_encoder : BiEncoder
        Pre-initialised bi-encoder (import from ``src.retrieval.bi_encoder``).
    index_path : Path or str
        Where to write the ``.index`` and ``_metadata.json`` files.
    batch_size : int
        Encoding batch size forwarded to ``BiEncoder.encode_passages``.

    Returns
    -------
    EvidenceIndex
        The built index, ready for querying.
    """
    eids = list(evidence_dict.keys())
    texts = [evidence_dict[eid] for eid in eids]

    embeddings = bi_encoder.encode_passages(
        texts,
        normalize=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    index = EvidenceIndex(dimension=bi_encoder.dim, index_path=index_path)
    index.build_index(embeddings, evidence_dict)
    index.save()

    print(
        f"Evidence index built: {index.size:,} passages × {index.dimension}d "
        f"→ {index_path}"
    )
    return index
