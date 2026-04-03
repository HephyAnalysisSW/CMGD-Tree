from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


@dataclass
class GaussianMixtureStreamProvider:
    """Stream synthetic regression data from two offset Gaussians."""

    n_features: int
    n_targets: int
    batch_size: int
    n_batches: int
    feature_offset_scale: float = 1.0
    target_offset_scale: float = 1.5
    feature_noise: float = 1.5
    target_noise: float = 0.2
    seed: int = 0
    dtype: np.dtype = np.float32

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._w = self._rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(max(self.n_features, 1)),
            size=(self.n_features, self.n_targets),
        ).astype(self.dtype)

        feature_offset = np.zeros(self.n_features, dtype=self.dtype)
        feature_offset[: min(4, self.n_features)] = self.feature_offset_scale
        self._feature_offset = feature_offset

        target_offset = np.zeros(self.n_targets, dtype=self.dtype)
        target_offset[: min(4, self.n_targets)] = self.target_offset_scale
        self._target_offset = target_offset

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for _ in range(self.n_batches):
            yield self.next_batch()

    def next_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        z = self._rng.integers(0, 2, size=self.batch_size, endpoint=False)
        sign = (2.0 * z.astype(self.dtype) - 1.0).reshape(-1, 1)

        x = self._rng.normal(
            loc=0.0,
            scale=self.feature_noise,
            size=(self.batch_size, self.n_features),
        ).astype(self.dtype)
        x += sign * self._feature_offset[None, :]

        y = x @ self._w
        y += sign * self._target_offset[None, :]
        y += self._rng.normal(
            loc=0.0,
            scale=self.target_noise,
            size=(self.batch_size, self.n_targets),
        ).astype(self.dtype)

        return x, y.astype(self.dtype, copy=False)

    @property
    def weights(self) -> np.ndarray:
        return self._w

    @property
    def feature_offset(self) -> np.ndarray:
        return self._feature_offset

    @property
    def target_offset(self) -> np.ndarray:
        return self._target_offset

@dataclass
class GaussianClassStreamProvider:
    """Stream synthetic multi-class data with one-hot labels.

    General case:
      - n_features arbitrary
      - n_classes arbitrary
      - each class is a Gaussian with its own mean vector
      - common isotropic feature_noise

    Special case:
      - if n_features == 1 and n_classes == 2, then
        class 0: N(0, feature_noise^2)
        class 1: N(feature_offset_scale, feature_noise^2)

      So with
        feature_offset_scale = 1.0
        feature_noise = 1.0
      you get exactly
        class 0: N(0, 1)
        class 1: N(1, 1)
    """

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
        self._class_means = np.zeros((self.n_classes, self.n_features), dtype=self.dtype)

        if self.n_features == 1 and self.n_classes == 2:
            self._class_means[0, 0] = 0.0
            self._class_means[1, 0] = self.feature_offset_scale
            return

        n_anchor = min(self.n_classes, self.n_features)
        for c in range(self.n_classes):
            self._class_means[c, c % n_anchor] = self.feature_offset_scale
            if self.n_features > 1:
                self._class_means[c, (c + 1) % self.n_features] = -0.5 * self.feature_offset_scale

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for _ in range(self.n_batches):
            yield self.next_batch()

    def next_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        cls = self._rng.integers(0, self.n_classes, size=self.batch_size, endpoint=False)

        x = self._rng.normal(
            loc=self._class_means[cls],
            scale=self.feature_noise,
            size=(self.batch_size, self.n_features),
        ).astype(self.dtype)

        y = np.zeros((self.batch_size, self.n_classes), dtype=self.dtype)
        y[np.arange(self.batch_size), cls] = 1.0
        return x, y

    @property
    def class_means(self) -> np.ndarray:
        return self._class_means
