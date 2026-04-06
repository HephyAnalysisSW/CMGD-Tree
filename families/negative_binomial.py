from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from families.base import BoostingFamily, class_weight_vector
from providers.base import StreamBatch
from providers.negative_binomial_toy import NegativeBinomialToyStream


@dataclass
class NegativeBinomialMGDFamily(BoostingFamily):
    prediction_dim: int
    class_weights: np.ndarray | None
    use_weights: bool
    name: str = "negative_binomial"
    monitor_name: str = "Negative Binomial NLL"
    provider_class = NegativeBinomialToyStream
    clip_epsilon: float = 1.0e-6
    shape: float = 4.0

    @classmethod
    def example_defaults(cls) -> dict[str, dict]:
        return {
            "tree": {
                "max_depth": 3,
                "max_leaves": 8,
            },
            "dataset": {
                "n_features": 4,
                "n_classes": 4,
            },
            "training": {
                "n_boost_rounds": 50,
                "learning_rate": 0.2,
            },
        }

    @classmethod
    def from_configs(cls, tree_config: dict, dataset_config: dict) -> "NegativeBinomialMGDFamily":
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
            "shape": self.shape,
        }

    def stream_batches(self, dataset_config: dict) -> Iterator[StreamBatch]:
        yield from self.provider_class(**self.provider_kwargs(dataset_config))

    def base_state(self, target_stat_mean: np.ndarray) -> np.ndarray:
        return np.maximum(np.ones_like(target_stat_mean, dtype=np.float32), self.clip_epsilon)

    def project_state(self, state_cpu: np.ndarray) -> np.ndarray:
        return np.maximum(state_cpu, self.clip_epsilon)

    def monitoring_loss(
        self,
        state_cpu: np.ndarray,
        target_stats_cpu: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[float, float]:
        mu = self.predict_from_state(state_cpu).astype(np.float64, copy=False)
        y = target_stats_cpu.astype(np.float64, copy=False)
        shape = float(self.shape)
        log_shape = math.log(shape)
        per_entry = (
            math.lgamma(shape)
            + np.vectorize(math.lgamma)(y + 1.0)
            - np.vectorize(math.lgamma)(y + shape)
            + shape * (np.log(mu + shape) - log_shape)
            + y * (np.log(mu + shape) - np.log(np.maximum(mu, self.clip_epsilon)))
        )
        per_row = np.sum(per_entry, axis=1)
        if sample_weight is None:
            return float(np.sum(per_row)), float(state_cpu.shape[0])
        return float(np.sum(sample_weight * per_row)), float(np.sum(sample_weight))
