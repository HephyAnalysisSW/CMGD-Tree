from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from families.base import BoostingFamily, class_weight_vector
from providers.base import StreamBatch
from providers.poisson_toy import PoissonToyStream


@dataclass
class PoissonMGDFamily(BoostingFamily):
    prediction_dim: int
    class_weights: np.ndarray | None
    use_weights: bool
    name: str = "poisson"
    monitor_name: str = "Poisson NLL"
    provider_class = PoissonToyStream
    clip_epsilon: float = 1.0e-6

    @classmethod
    def from_configs(cls, tree_config: dict, dataset_config: dict) -> "PoissonMGDFamily":
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
        }

    def stream_batches(self, dataset_config: dict) -> Iterator[StreamBatch]:
        yield from self.provider_class(**self.provider_kwargs(dataset_config))

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

    def plot_config(self, n_features: int, plot_mode: str = "auto") -> dict:
        pairs = [
            (feature_idx, target_idx)
            for feature_idx in range(n_features)
            for target_idx in range(self.prediction_dim)
        ]
        if plot_mode == "auto":
            plot_mode = "all"
        if plot_mode == "all":
            return {
                "modes": [
                    {"mode": "feature_target_mean", "pairs": pairs},
                    {"mode": "class_density", "feature_indices": list(range(n_features))},
                ]
            }
        return {"mode": "feature_target_mean", "pairs": pairs}


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
