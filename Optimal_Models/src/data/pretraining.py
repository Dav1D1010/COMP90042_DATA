"""
Streaming pretraining dataset wrapping HuggingFace datasets.

Provides StreamingPretrainDataset — an IterableDataset that streams
text data from HuggingFace and yields tokenized sequences on-the-fly.

Why streaming (not loading to RAM/disk):
    1.2B+ pretraining tokens at 16384 vocab × 1024 seq_len would require
    thousands of compressed shards on disk (>100GB uncompressed). Colab's
    50GB disk and 12GB RAM cannot hold this. Streaming processes one example
    at a time, keeping only the current batch in GPU memory.
    Even if we could fit the tokenized data, the I/O overhead of writing
    then re-reading terabytes of integer lists would dominate training time.
"""

import time
from datetime import timedelta

import torch
from datasets import load_dataset


class StreamingPretrainDataset(torch.utils.data.IterableDataset):
    """
    Streaming iterable dataset for MLM pretraining.

    Wraps a HuggingFace streaming dataset and yields raw tokenized
    sequences (list[int]) without BOS/EOS wrapping — those are added
    by MLMCollator during batching.

    Includes ETA tracking for monitoring long pretraining runs
    (150M–800M tokens at 16K tokens/step = thousands of steps).

    Default dataset: HuggingFaceFW/fineweb-edu, sample-10BT, train split.
    This is a 10B-token sample from FineWeb-Edu, filtered for educational
    quality — suitable for pretraining a climate-domain encoder.

    Args:
        hf_dataset_name: HuggingFace dataset identifier.
        hf_config: Dataset configuration/subset name.
        split: Dataset split (usually "train" for pretraining).
        streaming: Whether to stream data (default True — required
                   for datasets larger than available RAM).
        tokenizer: Tokenizer object with encode(text) → list[int].
        max_seq_len: Maximum tokenized sequence length.
    """

    def __init__(
        self,
        hf_dataset_name: str = "HuggingFaceFW/fineweb-edu",
        hf_config: str = "sample-10BT",
        split: str = "train",
        streaming: bool = True,
        tokenizer=None,
        max_seq_len: int = 1024,
    ):
        # Allow disabling streaming for tiny test datasets
        # (e.g. hf_dataset_name="wikitext", hf_config="wikitext-2-raw-v1")
        self.ds = load_dataset(
            hf_dataset_name, hf_config, streaming=streaming, split=split
        )
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # ETA tracking
        self._tokens_yielded = 0
        self._start_time: float | None = None
        self._target_tokens: int | None = None

    def set_target(self, num_tokens: int):
        """
        Set target token count for ETA tracking.

        Args:
            num_tokens: Total number of tokens to consume (e.g., 650_000_000).
        """
        self._target_tokens = num_tokens

    def __iter__(self):
        """
        Yields raw tokenized sequences (list[int]) with no BOS/EOS wrapping.

        BOS/EOS is applied by MLMCollator during batching.
        """
        self._start_time = time.time()
        self._tokens_yielded = 0

        for example in self.ds:
            text = example.get("text", "")
            if not text:
                continue
            ids = self.tokenizer.encode(text)
            ids = ids[: self.max_seq_len]
            self._tokens_yielded += len(ids)
            yield ids

    def tokenized(self):
        """
        Returns an iterator over tokenized sequences (list[int]).

        Convenience alias for iter(dataset).
        """
        return iter(self)

    @property
    def tokens_consumed(self) -> int:
        """Number of tokens yielded so far."""
        return self._tokens_yielded

    @property
    def progress(self) -> float:
        """
        Fraction of target tokens consumed (0.0 to 1.0).

        Returns 0.0 if no target has been set via set_target().
        """
        if self._target_tokens is None or self._target_tokens == 0:
            return 0.0
        return min(1.0, self._tokens_yielded / self._target_tokens)

    @property
    def eta(self) -> timedelta | None:
        """
        Estimated time remaining to reach target tokens.

        Returns a timedelta, or None if no target set or no tokens consumed yet.

        Example:
            >>> ds.set_target(650_000_000)
            >>> for batch in loader:
            ...     if step % 100 == 0:
            ...         print(f"ETA: {ds.eta}")
        """
        if self._target_tokens is None or self._tokens_yielded == 0:
            return None
        if self._start_time is None:
            return None
        elapsed = time.time() - self._start_time
        speed = self._tokens_yielded / elapsed if elapsed > 0 else 0
        if speed <= 0:
            return None
        remaining_tokens = self._target_tokens - self._tokens_yielded
        return timedelta(seconds=int(remaining_tokens / speed))

    @property
    def tokens_per_second(self) -> float:
        """Average token throughput since iteration started."""
        if self._start_time is None or self._tokens_yielded == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        return self._tokens_yielded / elapsed if elapsed > 0 else 0.0

    def __len__(self):
        """
        Returns a large sentinel value for DataLoader compatibility.

        Since this is a streaming dataset, the true length is unbounded.
        Returning a large value prevents DataLoader from refusing to iterate.
        The training loop controls termination via token count, not epoch count.
        """
        return 10_000_000
