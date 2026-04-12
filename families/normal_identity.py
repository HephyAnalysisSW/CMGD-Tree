from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from families.base import BoostingFamily, class_weight_vector


@dataclass
class NormalIdentityFamily(BoostingFamily):
    prediction_dim: int
    class_weights: np.ndarray | None
    use_weights: bool
    name: str = "normal_identity"
    monitor_name: str = "MSE"

    @classmethod
    def from_configs(cls, tree_config: dict, dataset_config: dict) -> "NormalIdentityFamily":
        class_weights, use_weights = class_weight_vector(tree_config, dataset_config.get("n_classes"))
        return cls(prediction_dim=dataset_config.get("n_classes"), class_weights=class_weights, use_weights=use_weights)

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
