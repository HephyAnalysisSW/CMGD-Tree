from __future__ import annotations

"""
Family-specific definitions for the current uncurved EF examples.

The data stream yields:
- x: input features
- T(y): target statistics to fit
- optional sample weights

For the current examples:
- normal_identity: T(y) = y
- poisson:         T(y) = y
"""

import math
import warnings
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class StreamBatch:
    x: np.ndarray
    target_stats: np.ndarray
    sample_weight: np.ndarray | None = None


def _class_weight_vector(tree_config: dict, dim: int) -> tuple[np.ndarray | None, bool]:
    configured = tree_config.get("class_weights")
    if configured is None:
        return None, False
    class_weights = np.asarray(configured, dtype=np.float32)
    if class_weights.shape != (dim,):
        raise ValueError("class_weights must have length prediction_dim.")
    if np.any(class_weights < 0.0):
        raise ValueError("class_weights must be non-negative.")
    return class_weights, True


@dataclass
class GaussianClassToyStream:
    n_features: int
    n_classes: int
    batch_size: int
    n_batches: int
    feature_offset_scale: float = 2.5
    feature_noise: float = 1.0
    seed: int = 0
    dtype: np.dtype = np.float32
    class_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._class_means = np.zeros((self.n_classes, self.n_features), dtype=self.dtype)
        self._class_targets = np.eye(self.n_classes, dtype=self.dtype)
        if self.n_features == 1 and self.n_classes == 2:
            self._class_means[0, 0] = 0.0
            self._class_means[1, 0] = self.feature_offset_scale
            return
        n_anchor = min(self.n_classes, self.n_features)
        for c in range(self.n_classes):
            self._class_means[c, c % n_anchor] = self.feature_offset_scale
            if self.n_features > 1:
                self._class_means[c, (c + 1) % self.n_features] = -0.5 * self.feature_offset_scale

    def __iter__(self) -> Iterator[StreamBatch]:
        for _ in range(self.n_batches):
            yield self.next_batch()

    def next_batch(self) -> StreamBatch:
        cls = self._rng.integers(0, self.n_classes, size=self.batch_size, endpoint=False)
        x = self._rng.normal(
            loc=0.0,
            scale=self.feature_noise,
            size=(self.batch_size, self.n_features),
        ).astype(self.dtype)
        x += self._class_means[cls]
        sample_weight = None if self.class_weights is None else self.class_weights[cls].astype(np.float32, copy=False)
        return StreamBatch(x=x, target_stats=self._class_targets[cls], sample_weight=sample_weight)


@dataclass
class PoissonToyStream:
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
        self._w = self._rng.normal(
            loc=0.0,
            scale=0.35 / np.sqrt(max(self.n_features, 1)),
            size=(self.n_features, self.n_classes),
        ).astype(self.dtype)
        self._bias = np.full((self.n_classes,), np.log(1.0), dtype=self.dtype)
        for c in range(self.n_classes):
            self._bias[c] += 0.15 * (c % 4)

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
            x[:, : min(4, self.n_features)] += self.feature_offset_scale / 4.0
        log_mu = x @ self._w + self._bias[None, :]
        mu = np.exp(np.clip(log_mu, -4.0, 4.0)).astype(self.dtype, copy=False)
        target_stats = self._rng.poisson(mu).astype(self.dtype)
        return StreamBatch(x=x, target_stats=target_stats)


@dataclass
class NormalIdentityFamily:
    prediction_dim: int
    class_weights: np.ndarray | None
    use_weights: bool
    name: str = "normal_identity"
    monitor_name: str = "MSE"
    provider_class = GaussianClassToyStream

    @classmethod
    def from_configs(cls, tree_config: dict, dataset_config: dict) -> "NormalIdentityFamily":
        class_weights, use_weights = _class_weight_vector(tree_config, dataset_config.get("n_classes"))
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

    def base_prediction(self, target_stat_mean: np.ndarray) -> np.ndarray:
        return target_stat_mean.astype(np.float32, copy=True)

    def project_prediction(self, pred_cpu: np.ndarray) -> np.ndarray:
        return pred_cpu

    def monitor_metric(
        self,
        pred_cpu: np.ndarray,
        target_stats_cpu: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[float, float]:
        pred_sq = np.sum(pred_cpu * pred_cpu, axis=1)
        target_prob = np.sum(pred_cpu * target_stats_cpu, axis=1)
        per_row = 1.0 - 2.0 * target_prob + pred_sq
        if sample_weight is None:
            return float(np.sum(per_row)), float(pred_cpu.shape[0])
        return float(np.sum(sample_weight * per_row)), float(np.sum(sample_weight))

    def plot_config(self, n_features: int, plot_mode: str = "auto") -> dict:
        mean_pairs = [
            (feature_idx, target_idx)
            for feature_idx in range(n_features)
            for target_idx in range(self.prediction_dim)
        ]
        if plot_mode == "auto":
            plot_mode = "all"
        if plot_mode == "all":
            return {
                "modes": [
                    {"mode": "feature_target_mean", "pairs": mean_pairs},
                    {"mode": "class_density", "feature_indices": list(range(n_features))},
                ]
            }
        if plot_mode == "feature_target_mean":
            return {"mode": "feature_target_mean", "pairs": mean_pairs}
        return {"mode": "class_density", "feature_indices": list(range(n_features))}


@dataclass
class PoissonFamily:
    prediction_dim: int
    class_weights: np.ndarray | None
    use_weights: bool
    name: str = "poisson"
    monitor_name: str = "Poisson NLL"
    provider_class = PoissonToyStream
    clip_epsilon: float = 1.0e-6

    @classmethod
    def from_configs(cls, tree_config: dict, dataset_config: dict) -> "PoissonFamily":
        class_weights, use_weights = _class_weight_vector(tree_config, dataset_config.get("n_classes"))
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

    def base_prediction(self, target_stat_mean: np.ndarray) -> np.ndarray:
        return np.maximum(np.ones_like(target_stat_mean, dtype=np.float32), self.clip_epsilon)

    def project_prediction(self, pred_cpu: np.ndarray) -> np.ndarray:
        pred_proj = np.maximum(pred_cpu, self.clip_epsilon)
        if np.any(pred_cpu < 0.0):
            warnings.warn("Negative Poisson predictions encountered; clipping to epsilon.", RuntimeWarning)
        return pred_proj

    def monitor_metric(
        self,
        pred_cpu: np.ndarray,
        target_stats_cpu: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[float, float]:
        pred_proj = self.project_prediction(pred_cpu).astype(np.float64, copy=False)
        y64 = target_stats_cpu.astype(np.float64, copy=False)
        lgamma = np.vectorize(math.lgamma)
        per_row = np.sum(pred_proj - y64 * np.log(pred_proj) + lgamma(y64 + 1.0), axis=1)
        if sample_weight is None:
            return float(np.sum(per_row)), float(pred_cpu.shape[0])
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


def family_from_configs(tree_config: dict, dataset_config: dict):
    family_name = tree_config.get("family", "normal_identity")
    if family_name == "normal_identity":
        return NormalIdentityFamily.from_configs(tree_config, dataset_config)
    if family_name == "poisson":
        return PoissonFamily.from_configs(tree_config, dataset_config)
    raise ValueError("family must be 'normal_identity' or 'poisson'.")
