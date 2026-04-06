from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from families.base import BoostingFamily
from providers.base import StreamBatch
from providers.heteroskedastic_normal_toy import HeteroskedasticNormalToyStream


@dataclass
class HeteroskedasticNormalFamily(BoostingFamily):
    prediction_dim: int
    class_weights: np.ndarray | None = None
    use_weights: bool = False
    name: str = "heteroskedastic_normal"
    monitor_name: str = "Gaussian NLL"
    provider_class = HeteroskedasticNormalToyStream
    min_variance: float = 0.1

    @classmethod
    def from_configs(cls, tree_config: dict, dataset_config: dict) -> "HeteroskedasticNormalFamily":
        if dataset_config.get("n_classes") != 2:
            raise ValueError("heteroskedastic_normal expects n_classes=2 for target stats [y, y^2].")
        return cls(prediction_dim=2)

    def provider_kwargs(self, dataset_config: dict) -> dict:
        return {
            "n_features": dataset_config.get("n_features"),
            "n_classes": dataset_config.get("n_classes"),
            "batch_size": dataset_config.get("batch_size"),
            "n_batches": dataset_config.get("n_batches"),
            "feature_offset_scale": dataset_config.get("feature_offset_scale"),
            "feature_noise": dataset_config.get("feature_noise"),
            "seed": dataset_config.get("seed"),
        }

    def stream_batches(self, dataset_config: dict) -> Iterator[StreamBatch]:
        yield from self.provider_class(**self.provider_kwargs(dataset_config))

    def base_state(self, target_stat_mean: np.ndarray) -> np.ndarray:
        state = target_stat_mean.astype(np.float32, copy=True)
        return self.project_state(state)

    def project_state(self, state_cpu: np.ndarray) -> np.ndarray:
        projected = state_cpu.copy()
        mu = projected[:, 0] if projected.ndim == 2 else projected[0:1]
        if projected.ndim == 2:
            projected[:, 1] = np.maximum(projected[:, 1], mu * mu + self.min_variance)
        else:
            projected[1] = max(projected[1], float(mu[0] * mu[0] + self.min_variance))
        return projected

    def monitoring_loss(
        self,
        state_cpu: np.ndarray,
        target_stats_cpu: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[float, float]:
        pred = self.predict_from_state(state_cpu).astype(np.float64, copy=False)
        y = target_stats_cpu[:, 0].astype(np.float64, copy=False)
        mu = pred[:, 0]
        var = np.maximum(pred[:, 1] - pred[:, 0] * pred[:, 0], self.min_variance)
        per_row = 0.5 * (np.log(2.0 * np.pi * var) + (y - mu) * (y - mu) / var)
        if sample_weight is None:
            return float(np.sum(per_row)), float(state_cpu.shape[0])
        return float(np.sum(sample_weight * per_row)), float(np.sum(sample_weight))
