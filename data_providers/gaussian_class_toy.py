from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from data_providers.base import StreamBatch


@dataclass
class GaussianClassToyStream:
    n_features: int
    n_classes: int
    batch_size: int
    n_batches: int
    feature_offset_scale: float = 2.5
    feature_noise: float = 1.0
    seed: int = 0
    dtype: np.dtype = np.float32
    class_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._class_means = np.zeros((self.n_classes, self.n_features), dtype=self.dtype)
        self._class_targets = np.eye(self.n_classes, dtype=self.dtype)
        if self.n_features == 1 and self.n_classes == 2:
            self._class_means[0, 0] = 0.0
            self._class_means[1, 0] = self.feature_offset_scale
            return
        n_anchor = min(self.n_classes, self.n_features)
        for c in range(self.n_classes):
            self._class_means[c, c % n_anchor] = self.feature_offset_scale
            if self.n_features > 1:
                self._class_means[c, (c + 1) % self.n_features] = -0.5 * self.feature_offset_scale

    def __iter__(self) -> Iterator[StreamBatch]:
        for _ in range(self.n_batches):
            yield self.next_batch()

    def next_batch(self) -> StreamBatch:
        cls = self._rng.integers(0, self.n_classes, size=self.batch_size, endpoint=False)
        x = self._rng.normal(
            loc=0.0,
            scale=self.feature_noise,
            size=(self.batch_size, self.n_features),
        ).astype(self.dtype)
        x += self._class_means[cls]
        sample_weight = None if self.class_weights is None else self.class_weights[cls].astype(np.float32, copy=False)
        return StreamBatch(x=x, target_stats=self._class_targets[cls], sample_weight=sample_weight)
