"""
Evidence loading and preprocessing.

Provides:
- EvidenceLoader: Lazy-loads evidence.json (1.2M passages) into a dict.
- preprocess_evidence(): Text normalization for evidence passages.

Stopword removal rationale:
    We deliberately keep stopwords in evidence text. ModernBERT-style
    bidirectional encoders rely on full sentence structure for contextual
    understanding — removing words like "the", "is", "are", "a" would
    destroy grammatical structure and degrade the model's ability to
    reason about claim-evidence relationships. Every token participates
    in self-attention; removing even common words shifts attention
    distributions unpredictably.
    
    Traditional TF-IDF retrieval systems remove stopwords because they
    save space and don't affect sparse bag-of-words similarity. For
    transformer models, every token is a signal — including function
    words that carry syntactic and discourse-level information.
"""

import json
import re
from pathlib import Path


class EvidenceLoader:
    """
    Lazy-loading wrapper for evidence.json (~1.2M climate science passages).

    Evidence is indexed by string IDs (e.g., "evidence-0000001") and
    maps to passage text. The dictionary interface supports direct
    lookups for claim-evidence pairing in ClaimDataset.

    Args:
        json_path: Path to evidence.json file.

    Usage:
        >>> loader = EvidenceLoader("data/evidence.json")
        >>> evidence = loader.load()        # force eager load
        >>> text = loader["evidence-0001"]  # lazy lookup
        >>> len(loader)                     # 1_200_000+
    """

    def __init__(self, json_path: str | Path):
        self.json_path = Path(json_path)
        self._data: dict[str, str] | None = None

    def load(self) -> dict[str, str]:
        """
        Load and return the full evidence dict.

        Cached on first call; subsequent calls return the cached dict.
        """
        if self._data is None:
            with open(self.json_path) as f:
                self._data = json.load(f)
        return self._data

    def __getitem__(self, key: str) -> str:
        if self._data is None:
            self.load()
        assert self._data is not None
        return self._data.get(key, "")

    def get(self, key: str, default: str = "") -> str:
        """Dict-like get with default value."""
        if self._data is None:
            self.load()
        assert self._data is not None
        return self._data.get(key, default)

    def __len__(self) -> int:
        if self._data is None:
            self.load()
        assert self._data is not None
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        if self._data is None:
            self.load()
        assert self._data is not None
        return key in self._data

    def keys(self):
        """Yield evidence IDs."""
        if self._data is None:
            self.load()
        assert self._data is not None
        return self._data.keys()

    def items(self):
        """Yield (evidence_id, text) pairs."""
        if self._data is None:
            self.load()
        assert self._data is not None
        return self._data.items()

    def values(self):
        """Yield evidence texts."""
        if self._data is None:
            self.load()
        assert self._data is not None
        return self._data.values()

    def __repr__(self) -> str:
        loaded = self._data is not None
        count = len(self._data) if loaded else "?"
        return f"EvidenceLoader({self.json_path.name!r}, loaded={loaded}, passages={count})"


def preprocess_evidence(text: str) -> str:
    """
    Normalize evidence text: lowercase and collapse whitespace.

    Does NOT remove stopwords — ModernBERT needs full sentences for
    contextual understanding. See module docstring for rationale.

    Args:
        text: Raw evidence passage text.

    Returns:
        Preprocessed (lowercased, whitespace-normalized) text.

    Example:
        >>> preprocess_evidence("  The  Earth's  climate  IS warming.  ")
        "the earth's climate is warming."
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()
