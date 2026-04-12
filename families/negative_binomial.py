from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from families.base import BoostingFamily, class_weight_vector


@dataclass
class NegativeBinomialMGDFamily(BoostingFamily):
    prediction_dim: int
    class_weights: np.ndarray | None
    use_weights: bool
    name: str = "negative_binomial"
    monitor_name: str = "Negative Binomial NLL"
    clip_epsilon: float = 1.0e-6
    shape: float = 4.0

    @classmethod
    def from_configs(cls, tree_config: dict, dataset_config: dict) -> "NegativeBinomialMGDFamily":
        class_weights, use_weights = class_weight_vector(tree_config, dataset_config.get("n_classes"))
        return cls(prediction_dim=dataset_config.get("n_classes"), class_weights=class_weights, use_weights=use_weights)

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
