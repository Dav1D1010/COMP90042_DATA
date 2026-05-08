"""
Collators for MLM pretraining and claim-evidence classification.

Provides two collate functions for DataLoader:
- MLMCollator: BERT-style masked language modeling with BOS/EOS wrapping
- ClassificationCollator: claim+evidence concatenation for fine-tuning

Token ID conventions (from ModelConfig / tokenizer):
    PAD=0, UNK=1, BOS=2, EOS=3, MASK=4

Why BOS/EOS for pretraining but not classification:
    Pretraining wraps each raw text sequence with BOS/EOS to frame it as a
    complete utterance. This teaches the model to recognize sentence boundaries
    and learn full-sentence semantics — essential for a bidirectional encoder.
    Classification receives a single structured input (claim + SEP + evidence)
    where the SEP token already marks the claim-evidence boundary, making
    additional BOS/EOS framing redundant. The model processes the entire
    concatenated sequence as one query+context unit.
"""

import torch


class MLMCollator:
    """
    Collate function for Masked Language Modeling pretraining.

    Wraps tokenized sequences with BOS/EOS tokens, pads to batch max length,
    and applies random masking following Devlin et al. (2019), "BERT:
    Pre-training of Deep Bidirectional Transformers".

    Masking scheme (applied to 15% of non-special tokens):
        - 80% → [MASK] token
        - 10% → random token (drawn from vocab, excluding special tokens)
        - 10% → unchanged (model must still predict correctly)

    Special tokens (PAD, BOS, EOS, MASK) are never masked — they have
    fixed semantics that the model should not be forced to reconstruct.

    Args:
        tokenizer: Tokenizer object with attributes:
                   pad_token_id, bos_token_id, eos_token_id, mask_token_id,
                   vocab_size.
        mlm_probability: Fraction of eligible tokens to mask (default 0.15).
    """

    def __init__(self, tokenizer, mlm_probability: float = 0.15):
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.mask_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size if isinstance(tokenizer.vocab_size, int) else tokenizer.vocab_size()
        self.mlm_probability = mlm_probability

    def __call__(self, batch: list[list[int]]) -> dict:
        """
        Collates and masks a batch of tokenized text sequences.

        Input: list of tokenized texts (list[int]), no BOS/EOS wrapping.
        Output: dict with {"input_ids", "labels", "attention_mask"}.
        """
        if not batch:
            return {"input_ids": torch.tensor([]), "labels": torch.tensor([]), "attention_mask": torch.tensor([])}

        # Wrap each sequence with BOS/EOS
        wrapped = [[self.bos_id] + list(seq) + [self.eos_id] for seq in batch]

        # Pad to max length in batch (capped at a reasonable limit to prevent OOM)
        max_len = max(len(s) for s in wrapped)
        M = min(max_len, 8192)  # hard cap for safety; typical config caps at 1024
        B = len(batch)

        input_ids = torch.full((B, M), self.pad_id, dtype=torch.long)
        labels = torch.full((B, M), -100, dtype=torch.long)
        attention_mask = torch.zeros((B, M), dtype=torch.long)

        special = {self.pad_id, self.bos_id, self.eos_id, self.mask_id}

        for i, seq in enumerate(wrapped):
            n = min(len(seq), M)
            t = torch.tensor(seq[:n])

            # Determine which positions are eligible for masking
            # (exclude special tokens — they carry structural meaning)
            eligible = torch.tensor(
                [int(x) not in special for x in t.tolist()], dtype=torch.bool
            )
            if eligible.sum() == 0:
                input_ids[i, :n] = t
                attention_mask[i, :n] = 1
                continue

            n_mask = max(1, int(eligible.sum().item() * self.mlm_probability))
            eligible_idx = eligible.nonzero(as_tuple=False).squeeze(-1)
            mask_indices = eligible_idx[torch.randperm(len(eligible_idx))[:n_mask]]

            masked = t.clone()
            labels[i, mask_indices] = t[mask_indices]

            rng = torch.rand(n_mask)
            # 80% → [MASK]
            masked[mask_indices[rng < 0.8]] = self.mask_id
            # 10% → random token (exclude special tokens 0-4)
            replace = (rng >= 0.8) & (rng < 0.9)
            if replace.any():
                random_ids = torch.randint(5, self.vocab_size, (replace.sum().item(),))
                masked[mask_indices[replace]] = random_ids
            # 10% → unchanged (already correct)

            input_ids[i, :n] = masked
            attention_mask[i, :n] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class ClassificationCollator:
    """
    Collate function for claim-evidence classification fine-tuning.

    Concatenates claim text with evidence texts (joined by SEP token),
    encodes via the tokenizer, and pads to batch max length.

    Does NOT apply BOS/EOS wrapping — unlike pretraining where each
    sequence needs explicit framing as a standalone utterance,
    classification receives a structured `claim + SEP + evidence` input
    where the SEP token already provides the semantic boundary.

    Args:
        tokenizer: Tokenizer object with encode(text) → list[int] and
                   pad_token_id, sep_token_id (or sep text marker).
        max_length: Maximum sequence length after tokenization (default 1024).
        label_map: Dict mapping label strings → integer class indices.
                   Default: SUPPORTS→0, REFUTES→1, NOT_ENOUGH_INFO→2, DISPUTED→3.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 1024,
        label_map: dict | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map or {
            "SUPPORTS": 0,
            "REFUTES": 1,
            "NOT_ENOUGH_INFO": 2,
            "DISPUTED": 3,
        }
        self.pad_id = getattr(tokenizer, 'pad_token_id', 0)

    def __call__(self, batch: list[dict]) -> dict:
        """
        Collates a batch of claim-evidence items.

        Each item: {"claim_text": str, "evidence_texts": list[str], "label": str}.
        Returns: {"input_ids": (B, M), "attention_mask": (B, M), "labels": (B,)}.
        """
        input_ids_list = []
        label_indices = []

        for item in batch:
            claim = item["claim_text"]
            evidence_list = item.get("evidence_texts", [])

            # Concatenate claim + evidence with SEP marker
            # The "<sep>" string is tokenized by the SentencePiece model
            # (registered as a user-defined symbol during tokenizer training).
            parts = [claim] + (evidence_list if isinstance(evidence_list, list) else [evidence_list])
            text = " <sep> ".join(parts)
            tokens = self.tokenizer.encode(text)

            # Truncate to max_length
            tokens = tokens[: self.max_length]

            input_ids_list.append(tokens)

            label_str = item.get("label", "NOT_ENOUGH_INFO")
            label_indices.append(self.label_map.get(label_str, 2))

        # Pad to max length in this batch
        B = len(batch)
        M = min(max(len(s) for s in input_ids_list), self.max_length) if input_ids_list else 0

        input_ids = torch.full((B, M), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, M), dtype=torch.long)

        for i, seq in enumerate(input_ids_list):
            n = min(len(seq), M)
            input_ids[i, :n] = torch.tensor(seq[:n])
            attention_mask[i, :n] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_indices, dtype=torch.long),
        }
