from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from data_providers.base import StreamBatch


@dataclass
class PoissonToyStream:
    n_features: int
    n_classes: int
    batch_size: int
    n_batches: int
    feature_offset_scale: float = 2.5
    feature_noise: float = 1.0
    seed: int = 0
    dtype: np.dtype = np.float32

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._w = self._rng.normal(
            loc=0.0,
            scale=0.35 / np.sqrt(max(self.n_features, 1)),
            size=(self.n_features, self.n_classes),
        ).astype(self.dtype)
        self._bias = np.full((self.n_classes,), np.log(1.0), dtype=self.dtype)
        for c in range(self.n_classes):
            self._bias[c] += 0.15 * (c % 4)

    def __iter__(self) -> Iterator[StreamBatch]:
        for _ in range(self.n_batches):
            yield self.next_batch()

    def next_batch(self) -> StreamBatch:
        x = self._rng.normal(
            loc=0.0,
            scale=self.feature_noise,
            size=(self.batch_size, self.n_features),
        ).astype(self.dtype)
        if self.n_features > 0:
            x[:, : min(4, self.n_features)] += self.feature_offset_scale / 4.0
        log_mu = x @ self._w + self._bias[None, :]
        mu = np.exp(np.clip(log_mu, -4.0, 4.0)).astype(self.dtype, copy=False)
        target_stats = self._rng.poisson(mu).astype(self.dtype)
        return StreamBatch(x=x, target_stats=target_stats)
