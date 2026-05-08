"""
Two-stage prediction pipeline: retrieve evidence, then classify the claim.

Design rationale
================

Why evidence retrieval first, then classification (two-stage pipeline)?
    The FEVER-style fact-checking problem requires the system to:
      1. Find relevant evidence among ~1.2M climate science passages.
      2. Read the claim+evidence pair and predict a label.
    A single end-to-end model would need to encode the claim against every
    passage simultaneously — infeasible at 1.2M passages on a T4.  The two-
    stage approach decouples the problem: a fast bi-encoder retrieves a
    small candidate set, then a deeper classifier reads only those candidates.

Why k=5 retrieval (default)?
    The training split has an average of 3.36 evidence passages per claim
    (median = 3).  Retrieving k=5 gives the classifier a generous context
    window — it sees all relevant evidence plus a few distractors, which
    improves robustness against retrieval noise.  k=5 matches
    ``ProjectConfig.top_k_retrieval``.

Why concatenate evidence texts for classification?
    The classifier (ClimatronForClassification) takes a single tokenised
    sequence: ``[CLS] claim [SEP] evidence_1 evidence_2 ... evidence_k [SEP]``.
    Concatenating evidence texts preserves the full joint context, letting
    self-attention operate across the entire claim+evidence span.  Splitting
    into separate encoder passes would lose cross-evidence interactions
    (e.g., contradictory evidence passages that jointly indicate DISPUTED).

"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

# Resolve package root so src.* imports work from any working directory.
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent  # Optimal_Models/
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.config import ProjectConfig, LABEL_NAMES  # noqa: E402
from src.data.claims import load_claims, LabelEncoder  # noqa: E402


class Predictor:
    """End-to-end fact-checking predictor: retrieve → classify → format.

    Parameters
    ----------
    bi_encoder : BiEncoder
        Pre-loaded bi-encoder (from ``src.retrieval.bi_encoder``) for
        encoding claims into the same embedding space as the FAISS index.
    evidence_index : EvidenceIndex
        Pre-built or pre-loaded FAISS index (from ``src.retrieval.faiss_index``)
        containing embeddings for ~1.2M evidence passages.
    classifier : ClimatronForClassification
        Fine-tuned classifier (from ``src.models.model``) that takes
        tokenised ``(claim + evidence)`` input and outputs 4-class logits.
    tokenizer : ClimatronTokenizer
        ByteLevel BPE tokenizer (from ``src.data.tokenizer``) for converting
        text to token IDs.  Its underlying ``tokenizers.Tokenizer`` must
        support pair encoding ``[CLS] claim [SEP] evidence [SEP]``.
    config : ProjectConfig
        Central project configuration (device, paths, top_k_retrieval, etc.).
    claim_dataset : ClaimDataset, optional
        If provided, ``predict_all()`` iterates over this dataset directly
        instead of loading claims from disk.  Useful for cross-validation
        where the dataset has already been split.
    """

    def __init__(
        self,
        bi_encoder,
        evidence_index,
        classifier,
        tokenizer,
        config: ProjectConfig,
        claim_dataset=None,
    ):
        self.bi_encoder = bi_encoder
        self.evidence_index = evidence_index
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.config = config
        self.claim_dataset = claim_dataset

        self._label_encoder = LabelEncoder(LABEL_NAMES)
        self._device = torch.device(config.device)

        # Ensure classifier is on the correct device and in eval mode.
        self.classifier.to(self._device)
        self.classifier.eval()

    # ── Public API ───────────────────────────────────────────────────

    def predict_all(self, split: str = "dev") -> Dict[str, Dict]:
        """Run the full pipeline over all claims in *split*.

        Parameters
        ----------
        split : str
            Which data split to evaluate: ``"train"``, ``"dev"``, or
            ``"test"``.  If ``self.claim_dataset`` was provided, the split
            is ignored and the dataset is used directly.

        Returns
        -------
        dict[str, dict]
            Predictions in baseline format:
            ``{claim_id: {"claim_text": str, "claim_label": str,
            "evidences": [str, ...]}}``
            where ``evidences`` is a list of evidence ID strings.
        """
        # Load claims from the provided dataset or from disk.
        if self.claim_dataset is not None:
            claims = dict(self.claim_dataset.claims)
        else:
            claims_path = self.config.data_dir / f"{split}-claims.json"
            claims = load_claims(claims_path)

        predictions: Dict[str, Dict] = {}

        for claim_id, claim_data in claims.items():
            claim_text = claim_data.get("claim_text", "")

            # Stage 1: retrieve top-k evidence
            retrieved = self._retrieve_evidence(
                claim_text, k=self.config.top_k_retrieval
            )
            evidence_ids = [r["id"] for r in retrieved]
            evidence_texts = [r["text"] for r in retrieved]

            # Stage 2: classify claim given the retrieved evidence
            pred_label = self._classify(claim_text, evidence_texts)

            predictions[claim_id] = {
                "claim_text": claim_text,
                "claim_label": pred_label,
                "evidences": evidence_ids,
            }

        return predictions

    def predict(self, claim_text: str, k: Optional[int] = None) -> Dict:
        """Predict label and evidence for a single claim.

        Parameters
        ----------
        claim_text : str
            The claim text to fact-check.
        k : int, optional
            Number of evidence passages to retrieve.  Defaults to
            ``self.config.top_k_retrieval`` (5).

        Returns
        -------
        dict
            ``{"claim_text": str, "claim_label": str, "evidences": [str, ...]}``.
        """
        if k is None:
            k = self.config.top_k_retrieval

        retrieved = self._retrieve_evidence(claim_text, k=k)
        evidence_ids = [r["id"] for r in retrieved]
        evidence_texts = [r["text"] for r in retrieved]
        pred_label = self._classify(claim_text, evidence_texts)

        return {
            "claim_text": claim_text,
            "claim_label": pred_label,
            "evidences": evidence_ids,
        }

    @staticmethod
    def save_predictions(
        predictions: Dict[str, Dict],
        output_path: Union[str, Path],
    ) -> None:
        """Write predictions to a JSON file.

        Parameters
        ----------
        predictions : dict
            Prediction dict in baseline format (as returned by
            :meth:`predict_all` or :meth:`predict`).
        output_path : str or Path
            Destination file path.  Parent directories are created if needed.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=None)
        print(f"Predictions saved to {output_path} ({len(predictions)} claims)")

    # ── Internal methods ─────────────────────────────────────────────

    def _retrieve_evidence(self, claim_text: str, k: int = 5) -> List[Dict]:
        """Encode *claim_text* and search the FAISS index for top-*k* evidence.

        Parameters
        ----------
        claim_text : str
        k : int
            Number of evidence passages to retrieve.

        Returns
        -------
        list[dict]
            Each dict has keys ``"id"``, ``"text"``, ``"score"`` (descending
            by score).
        """
        query_embedding = self.bi_encoder.encode_queries(
            [claim_text],
            normalize=True,
            batch_size=1,
            show_progress_bar=False,
        )
        return self.evidence_index.search(query_embedding[0], k=k)

    def _classify(self, claim_text: str, evidence_texts: List[str]) -> str:
        """Classify a claim given its retrieved evidence texts.

        Concatenates claim and evidence into a single tokenised sequence
        ``[CLS] claim [SEP] evidence_1 evidence_2 ... [SEP]``, passes it
        through the classifier, and returns the predicted label string.

        Parameters
        ----------
        claim_text : str
        evidence_texts : list[str]
            Evidence passage texts retrieved for this claim.

        Returns
        -------
        str
            One of ``"SUPPORTS"``, ``"REFUTES"``, ``"NOT_ENOUGH_INFO"``,
            ``"DISPUTED"``.
        """
        evidence_joined = " ".join(evidence_texts) if evidence_texts else ""

        # Use the underlying tokenizers.Tokenizer for pair encoding, which
        # produces [CLS] claim [SEP] evidence [SEP] via TemplateProcessing.
        encoding = self.tokenizer._tokenizer.encode(claim_text, evidence_joined)
        input_ids = torch.tensor([encoding.ids], device=self._device)
        attention_mask = torch.tensor(
            [encoding.attention_mask], device=self._device
        )

        with torch.no_grad():
            logits = self.classifier(input_ids, attention_mask)
            pred_idx = torch.argmax(logits, dim=-1).item()

        return self._label_encoder.decode(pred_idx)
