from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from families.base import BoostingFamily


@dataclass
class HeteroskedasticNormalFamily(BoostingFamily):
    prediction_dim: int
    class_weights: np.ndarray | None = None
    use_weights: bool = False
    name: str = "heteroskedastic_normal"
    monitor_name: str = "Gaussian NLL"
    min_variance: float = 0.1

    @classmethod
    def from_configs(cls, tree_config: dict, dataset_config: dict) -> "HeteroskedasticNormalFamily":
        if dataset_config.get("n_classes") != 2:
            raise ValueError("heteroskedastic_normal expects n_classes=2 for target stats [y, y^2].")
        return cls(prediction_dim=2)

    def base_state(self, target_stat_mean: np.ndarray) -> np.ndarray:
        state = target_stat_mean.astype(np.float32, copy=True)
        return self.project_state(state)

    def project_state(self, state_cpu: np.ndarray) -> np.ndarray:
        projected = state_cpu.copy()
        if projected.ndim == 2:
            mu = projected[:, 0]
            projected[:, 1] = np.maximum(projected[:, 1], mu * mu + self.min_variance)
        else:
            mu = projected[0]
            projected[1] = max(projected[1], float(mu * mu + self.min_variance))
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


@dataclass
class HeteroskedasticNormalNGDFamily(HeteroskedasticNormalFamily):
    name: str = "heteroskedastic_normal_ngd"
    eta2_clip_epsilon: float = 1.0e-6

    @classmethod
    def from_configs(cls, tree_config: dict, dataset_config: dict) -> "HeteroskedasticNormalNGDFamily":
        if dataset_config.get("n_classes") != 2:
            raise ValueError("heteroskedastic_normal_ngd expects n_classes=2 for target stats [y, y^2].")
        return cls(prediction_dim=2)

    def base_state(self, target_stat_mean: np.ndarray) -> np.ndarray:
        mean = float(target_stat_mean[0])
        second_moment = float(target_stat_mean[1])
        variance = max(second_moment - mean * mean, self.min_variance)
        eta1 = mean / variance
        eta2 = -0.5 / variance
        return np.array([eta1, eta2], dtype=np.float32)

    def project_state(self, state_cpu: np.ndarray) -> np.ndarray:
        projected = state_cpu.copy()
        eta2_min = -0.5 / self.min_variance
        if projected.ndim == 2:
            projected[:, 1] = np.clip(projected[:, 1], eta2_min, -self.eta2_clip_epsilon)
        else:
            projected[1] = float(np.clip(projected[1], eta2_min, -self.eta2_clip_epsilon))
        return projected

    def predict_from_state(self, state_cpu: np.ndarray) -> np.ndarray:
        state = self.project_state(state_cpu)
        eta1 = state[:, 0] if state.ndim == 2 else state[0:1]
        eta2 = state[:, 1] if state.ndim == 2 else state[1:2]
        variance = np.maximum(-0.5 / eta2, self.min_variance)
        mean = -0.5 * eta1 / eta2
        second_moment = mean * mean + variance
        if state.ndim == 2:
            return np.stack((mean, second_moment), axis=1).astype(np.float32, copy=False)
        return np.array([float(mean[0]), float(second_moment[0])], dtype=np.float32)

    def preconditioned_target(self, state_cpu: np.ndarray, target_stats_cpu: np.ndarray) -> np.ndarray:
        mean_stats = self.predict_from_state(state_cpu).astype(np.float32, copy=False)
        residual = target_stats_cpu.astype(np.float32, copy=False) - mean_stats
        mean = mean_stats[:, 0] if mean_stats.ndim == 2 else np.array([mean_stats[0]], dtype=np.float32)
        variance = (
            np.maximum(mean_stats[:, 1] - mean_stats[:, 0] * mean_stats[:, 0], self.min_variance)
            if mean_stats.ndim == 2
            else np.array([max(float(mean_stats[1] - mean_stats[0] * mean_stats[0]), self.min_variance)], dtype=np.float32)
        )
        inv_v = 1.0 / variance
        inv_v2 = inv_v * inv_v
        out = np.empty_like(residual, dtype=np.float32)
        out[..., 0] = (inv_v + 2.0 * mean * mean * inv_v2) * residual[..., 0] - (mean * inv_v2) * residual[..., 1]
        out[..., 1] = -(mean * inv_v2) * residual[..., 0] + 0.5 * inv_v2 * residual[..., 1]
        return out
