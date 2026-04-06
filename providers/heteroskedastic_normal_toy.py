from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from providers.base import PlotConfigProvider, StreamBatch


@dataclass
class HeteroskedasticNormalToyStream(PlotConfigProvider):
    n_features: int
    n_classes: int
    batch_size: int
    n_batches: int
    feature_offset_scale: float = 2.5
    feature_noise: float = 1.0
    seed: int = 0
    dtype: np.dtype = np.float32

    def __post_init__(self) -> None:
        if self.n_classes != 2:
            raise ValueError("HeteroskedasticNormalToyStream expects n_classes=2 for target stats [y, y^2].")
        self._rng = np.random.default_rng(self.seed)

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
            x[:, 0] += self.feature_offset_scale / 4.0

        mu = np.zeros((self.batch_size,), dtype=np.float32)
        if self.n_features > 0:
            mu += 1.25 * x[:, 0]
        if self.n_features > 1:
            mu += -0.75 * x[:, 1]
        if self.n_features > 2:
            mu += 0.6 * (x[:, 2] > 0.0).astype(np.float32)
        if self.n_features > 3:
            mu += -0.4 * (x[:, 3] < 0.0).astype(np.float32)

        log_var = np.full((self.batch_size,), -0.35, dtype=np.float32)
        if self.n_features > 2:
            log_var += 0.7 * x[:, 2]
        if self.n_features > 3:
            log_var += 0.35 * np.abs(x[:, 3])
        var = np.exp(np.clip(log_var, -4.0, 3.0)).astype(np.float32, copy=False)
        y = (mu + np.sqrt(var).astype(np.float32, copy=False) * self._rng.normal(size=self.batch_size)).astype(np.float32)
        target_stats = np.stack((y, y * y), axis=1).astype(np.float32, copy=False)
        return StreamBatch(x=x, target_stats=target_stats)

    def plot_config(self, plot_mode: str = "auto", n_bins: int = 80) -> dict:
        if plot_mode == "auto":
            plot_mode = "heteroskedastic"
        if plot_mode in {"all", "heteroskedastic", "heteroskedastic_normal_scalar"}:
            return {"mode": "heteroskedastic_normal_scalar", "feature_indices": list(range(self.n_features))}
        raise ValueError(f"Unsupported plot_mode '{plot_mode}' for HeteroskedasticNormalToyStream.")
