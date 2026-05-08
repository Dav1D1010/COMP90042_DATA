"""
Claim dataset loading and processing for fact-checking classification.

Provides:
- load_claims(): Load claim JSON files into dicts keyed by claim_id.
- LabelEncoder: Bidirectional string ↔ integer label mapping.
- ClaimDataset: PyTorch Dataset yielding (claim, evidence, label) triples.

Class distribution (train split, 1228 claims):
    SUPPORTS:         519  (42.3%) — majority class
    NOT_ENOUGH_INFO:  386  (31.4%)
    REFUTES:          199  (16.2%)
    DISPUTED:         124  (10.1%)

Why this matters for loss design:
    The ~4:1 imbalance between majority (SUPPORTS) and minority (DISPUTED)
    causes standard cross-entropy loss to bias predictions toward the head
    classes. The model learns to be "lazy" — predicting SUPPORTS on every
    sample gives 42% accuracy without learning claim-evidence relationships.
    
    Mitigations (applied in training, not here):
    - LDAM (Label-Distribution-Aware Margin): Larger margins for minority
      classes, pushing decision boundaries away from rare classes.
    - Class-Balanced (CB) loss: Re-weights each sample inversely proportional
      to effective sample count, preventing head-class domination.
    - Label smoothing (α=0.1): Softens targets to prevent overconfidence,
      especially important for the fine-grained NOT_ENOUGH_INFO class.
"""

import json
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset


def load_claims(json_path: str | Path, data_dir: Optional[str | Path] = None) -> dict[str, dict]:
    """
    Load a claim JSON file and return a dict keyed by claim_id.

    The JSON file is expected to have the structure:
        {"claim-0001": {"claim_text": "...", "claim_label": "SUPPORTS",
                        "evidences": ["evidence-0001", ...]}, ...}

    Args:
        json_path: Path to the claim JSON file (e.g., "train-claims.json").
        data_dir: Optional directory to resolve json_path against.
                  If provided and json_path is not absolute, data_dir is
                  prepended. Useful for cross-validation where claims
                  live in a configurable data directory.

    Returns:
        Dict[str, dict] mapping claim_id → claim data dict.

    Example:
        >>> train_claims = load_claims("data/train-claims.json")
        >>> dev_claims = load_claims("dev-claims.json", data_dir="data/")
    """
    path = Path(json_path)
    if data_dir is not None and not path.is_absolute():
        path = Path(data_dir) / path
    with open(path) as f:
        data = json.load(f)
    return data


class LabelEncoder:
    """
    Bidirectional mapping between string labels and integer indices.

    Maps "SUPPORTS" → 0, "REFUTES" → 1, "NOT_ENOUGH_INFO" → 2, "DISPUTED" → 3
    by default (matching LABEL_NAMES order from ProjectConfig).

    Args:
        label_names: Ordered list of label strings. Index position = integer label.
    """

    def __init__(self, label_names: list[str]):
        self.label_names = list(label_names)
        self._str_to_int = {name: idx for idx, name in enumerate(self.label_names)}
        self._int_to_str = {idx: name for idx, name in enumerate(self.label_names)}

    def encode(self, label_str: str) -> int:
        """Convert a label string to its integer index."""
        return self._str_to_int.get(label_str, -1)

    def decode(self, label_int: int) -> str:
        """Convert an integer index back to its label string."""
        return self._int_to_str.get(label_int, "UNKNOWN")

    def __len__(self) -> int:
        return len(self.label_names)

    def __repr__(self) -> str:
        return f"LabelEncoder({self.label_names})"


class ClaimDataset(Dataset):
    """
    PyTorch Dataset for claim-evidence classification.

    Yields dicts with:
        - claim_text (str): The claim to classify.
        - evidence_texts (str): Space-separated evidence passages joined
          by <sep> markers.
        - label (int): Integer class label (0=SUPPORTS, 1=REFUTES,
          2=NOT_ENOUGH_INFO, 3=DISPUTED).
        - claim_id (str): The unique claim identifier.

    The SEP joining is done here rather than in the collator so the
    tokenizer can process the full concatenated string as a single
    sequence with correct subword boundaries at the join points.

    Args:
        claims_dict: Dict[str, dict] from load_claims().
        evidence_dict: Dict[str, str] mapping evidence_id → text.
        tokenizer: Tokenizer object (unused for text; kept for API symmetry
                   — tokenization happens in the collator).
        max_length: Maximum tokenized length hint (enforced by collator).
        data_dir: Optional data directory override for cross-validation setups.
        split: Dataset split label ("train", "dev", "test") for logging/checks.
    """

    def __init__(
        self,
        claims_dict: dict[str, dict],
        evidence_dict: dict[str, str] | None = None,
        tokenizer=None,
        max_length: int = 1024,
        data_dir: Optional[str | Path] = None,
        split: str = "train",
    ):
        self.claims = dict(claims_dict)  # shallow copy
        self.evidence = evidence_dict or {}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_dir = Path(data_dir) if data_dir else None
        self.split = split

        # Flat list for deterministic indexing
        self._items = list(self.claims.items())

        # Build internal label mapping (same order as LABEL_NAMES)
        self._label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        claim_id, claim_data = self._items[idx]

        claim_text = claim_data.get("claim_text", "")
        label_str = claim_data.get("claim_label", "NOT_ENOUGH_INFO")
        evidence_ids = claim_data.get("evidences", [])

        # Collect evidence texts
        evidence_texts = []
        for eid in evidence_ids:
            text = self.evidence.get(eid, "")
            if text:
                evidence_texts.append(text)

        # Join into a single string with SEP markers
        evidence_joined = " <sep> ".join(evidence_texts) if evidence_texts else ""

        return {
            "claim_text": claim_text,
            "evidence_texts": evidence_joined,
            "label": self._label_map.get(label_str, 2),
            "claim_id": claim_id,
        }

    @property
    def class_counts(self) -> dict[str, int]:
        """
        Count samples per class in this dataset.

        Useful for initializing LDAM+CB loss weights.
        """
        counts = {name: 0 for name in self._label_map}
        for _, claim_data in self._items:
            label = claim_data.get("claim_label", "NOT_ENOUGH_INFO")
            counts[label] = counts.get(label, 0) + 1
        return counts

    def split_train_dev(self, dev_ratio: float = 0.1, seed: int = 42):
        """
        Split this dataset into train/dev subsets for cross-validation.

        Args:
            dev_ratio: Fraction of data for dev set (default 0.1).
            seed: Random seed for reproducibility.

        Returns:
            (train_dataset, dev_dataset): Two ClaimDataset instances.
        """
        import random
        random.seed(seed)
        indices = list(range(len(self._items)))
        random.shuffle(indices)

        n_dev = max(1, int(len(indices) * dev_ratio))
        dev_indices = set(indices[:n_dev])
        train_indices = set(indices[n_dev:])

        train_items = [self._items[i] for i in range(len(self._items)) if i in train_indices]
        dev_items = [self._items[i] for i in range(len(self._items)) if i in dev_indices]

        train_ds = ClaimDataset.__new__(ClaimDataset)
        train_ds.claims = dict(train_items)
        train_ds.evidence = self.evidence
        train_ds.tokenizer = self.tokenizer
        train_ds.max_length = self.max_length
        train_ds.data_dir = self.data_dir
        train_ds.split = "train"
        train_ds._items = train_items
        train_ds._label_map = self._label_map

        dev_ds = ClaimDataset.__new__(ClaimDataset)
        dev_ds.claims = dict(dev_items)
        dev_ds.evidence = self.evidence
        dev_ds.tokenizer = self.tokenizer
        dev_ds.max_length = self.max_length
        dev_ds.data_dir = self.data_dir
        dev_ds.split = "dev"
        dev_ds._items = dev_items
        dev_ds._label_map = self._label_map

        return train_ds, dev_ds
