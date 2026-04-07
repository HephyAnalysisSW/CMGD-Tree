from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from families.base import BoostingFamily, class_weight_vector
from data_providers.base import StreamBatch
from data_providers.gaussian_class_toy import GaussianClassToyStream


@dataclass
class NormalIdentityFamily(BoostingFamily):
    prediction_dim: int
    class_weights: np.ndarray | None
    use_weights: bool
    name: str = "normal_identity"
    monitor_name: str = "MSE"
    provider_class = GaussianClassToyStream

    @classmethod
    def from_configs(cls, tree_config: dict, dataset_config: dict) -> "NormalIdentityFamily":
        class_weights, use_weights = class_weight_vector(tree_config, dataset_config.get("n_classes"))
        return cls(prediction_dim=dataset_config.get("n_classes"), class_weights=class_weights, use_weights=use_weights)

    def provider_kwargs(self, dataset_config: dict) -> dict:
        return {
            "n_features": dataset_config.get("n_features"),
            "n_classes": dataset_config.get("n_classes"),
            "batch_size": dataset_config.get("batch_size"),
            "n_batches": dataset_config.get("n_batches"),
            "feature_offset_scale": dataset_config.get("feature_offset_scale"),
            "feature_noise": dataset_config.get("feature_noise"),
            "seed": dataset_config.get("seed"),
            "class_weights": self.class_weights,
        }

    def stream_batches(self, dataset_config: dict) -> Iterator[StreamBatch]:
        yield from self.provider_class(**self.provider_kwargs(dataset_config))

    def base_state(self, target_stat_mean: np.ndarray) -> np.ndarray:
        return target_stat_mean.astype(np.float32, copy=True)

    def monitoring_loss(
        self,
        state_cpu: np.ndarray,
        target_stats_cpu: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[float, float]:
        pred_cpu = self.predict_from_state(state_cpu)
        pred_sq = np.sum(pred_cpu * pred_cpu, axis=1)
        target_prob = np.sum(pred_cpu * target_stats_cpu, axis=1)
        per_row = 1.0 - 2.0 * target_prob + pred_sq
        if sample_weight is None:
            return float(np.sum(per_row)), float(state_cpu.shape[0])
        return float(np.sum(sample_weight * per_row)), float(np.sum(sample_weight))
