from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from families.base import BoostingFamily, class_weight_vector


@dataclass
class PoissonMGDFamily(BoostingFamily):
    prediction_dim: int
    class_weights: np.ndarray | None
    use_weights: bool
    name: str = "poisson"
    monitor_name: str = "Poisson NLL"
    clip_epsilon: float = 1.0e-6

    @classmethod
    def from_configs(cls, tree_config: dict, dataset_config: dict) -> "PoissonMGDFamily":
        class_weights, use_weights = class_weight_vector(tree_config, dataset_config.get("n_classes"))
        return cls(prediction_dim=dataset_config.get("n_classes"), class_weights=class_weights, use_weights=use_weights)

    def base_state(self, target_stat_mean: np.ndarray) -> np.ndarray:
        return np.maximum(np.ones_like(target_stat_mean, dtype=np.float32), self.clip_epsilon)

    def project_state(self, state_cpu: np.ndarray) -> np.ndarray:
        pred_proj = np.maximum(state_cpu, self.clip_epsilon)
        if np.any(state_cpu < 0.0):
            warnings.warn("Negative Poisson predictions encountered; clipping to epsilon.", RuntimeWarning)
        return pred_proj

    def monitoring_loss(
        self,
        state_cpu: np.ndarray,
        target_stats_cpu: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[float, float]:
        pred_proj = self.predict_from_state(state_cpu).astype(np.float64, copy=False)
        y64 = target_stats_cpu.astype(np.float64, copy=False)
        lgamma = np.vectorize(math.lgamma)
        per_row = np.sum(pred_proj - y64 * np.log(pred_proj) + lgamma(y64 + 1.0), axis=1)
        if sample_weight is None:
            return float(np.sum(per_row)), float(state_cpu.shape[0])
        return float(np.sum(sample_weight * per_row)), float(np.sum(sample_weight))

@dataclass
class PoissonNGDFamily(PoissonMGDFamily):
    name: str = "poisson_ngd"
    state_clip: float = 12.0

    def base_state(self, target_stat_mean: np.ndarray) -> np.ndarray:
        return np.zeros_like(target_stat_mean, dtype=np.float32)

    def project_state(self, state_cpu: np.ndarray) -> np.ndarray:
        return np.clip(state_cpu, -self.state_clip, self.state_clip)

    def predict_from_state(self, state_cpu: np.ndarray) -> np.ndarray:
        state_proj = self.project_state(state_cpu)
        return np.exp(state_proj).astype(np.float32, copy=False)

    def preconditioned_target(self, state_cpu: np.ndarray, target_stats_cpu: np.ndarray) -> np.ndarray:
        mu = self.dual_prediction(state_cpu)
        return target_stats_cpu / np.maximum(mu, self.clip_epsilon) - 1.0
