from __future__ import annotations

"""
Family-specific definitions for the current uncurved normal model.

The current tree trainer uses this file as the user-facing place for:
- the toy stream
- the target statistic T(y)
- the monitoring metric

Current model:
- Normal with identity sufficient statistic
- T(y) = y
- mean coordinate = predicted vector itself

Examples for later families:

1. Normal, identity statistic
   y:        vector in R^d
   T(y):     y
   pred:     mean vector mu

2. Poisson, identity statistic
   y:        non-negative count vector
   T(y):     y
   pred:     mean count vector mu > 0

3. Binomial with fixed N
   y:        success counts in {0, ..., N}
   T(y):     y
   pred:     mean success count mu in [0, N]

These examples all fit the same near-term pattern:
- the data loader produces T(y)
- the tree fits vector-valued targets in mean coordinates
- the global monitor is the family-specific NLL
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


@dataclass
class CachedTrainingBatch:
    bins: np.ndarray
    target_stats: np.ndarray
    sample_weight: np.ndarray | None = None
    target_codes: np.ndarray | None = None


def _class_weight_vector(tree_config: dict, dim: int) -> tuple[np.ndarray, bool]:
    configured = tree_config.get("class_weights")
    if configured is None:
        return np.ones(dim, dtype=np.float32), False
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
        target_stats = self._class_targets[cls]
        return StreamBatch(x=x, target_stats=target_stats)


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
    class_weights: np.ndarray
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
        }

    def stream_batches(self, dataset_config: dict) -> Iterator[StreamBatch]:
        yield from self.provider_class(**self.provider_kwargs(dataset_config))

    def make_training_batch(self, bins_cpu: np.ndarray, batch: StreamBatch) -> CachedTrainingBatch:
        target_codes = np.argmax(batch.target_stats, axis=1).astype(
            np.int16 if batch.target_stats.shape[1] > 256 else np.uint8,
            copy=False,
        )
        if self.use_weights:
            sample_weight = self.class_weights[target_codes].astype(np.float32, copy=False)
            return CachedTrainingBatch(
                bins=bins_cpu,
                target_stats=batch.target_stats,
                sample_weight=sample_weight,
                target_codes=target_codes.copy(),
            )
        return CachedTrainingBatch(
            bins=bins_cpu,
            target_stats=batch.target_stats,
            target_codes=target_codes.copy(),
        )

    def fit_representation(self, batch: CachedTrainingBatch) -> np.ndarray:
        if batch.target_codes is not None:
            return batch.target_codes
        return batch.target_stats

    def uses_target_codes(self, batch: CachedTrainingBatch) -> bool:
        return batch.target_codes is not None

    def leaf_value(self, target_stat_sum: np.ndarray, total_weight: float, reg_lambda: float) -> np.ndarray:
        return target_stat_sum / (total_weight + reg_lambda)

    def leaf_score(self, target_stat_sum: np.ndarray, total_weight: float, reg_lambda: float) -> float:
        if total_weight <= 0.0:
            return -np.inf
        return float(np.dot(target_stat_sum, target_stat_sum) / (total_weight + reg_lambda))

    def total_weight_from_stats(self, target_stat_sum: np.ndarray, count: int) -> float:
        if self.use_weights:
            return float(np.sum(target_stat_sum))
        return float(count)

    def monitor_metric(self, pred_cpu: np.ndarray, target_stats_cpu: np.ndarray, sample_weight: np.ndarray | None = None) -> tuple[float, float]:
        pred_sq = np.sum(pred_cpu * pred_cpu, axis=1)
        target_prob = np.sum(pred_cpu * target_stats_cpu, axis=1)
        per_row = 1.0 - 2.0 * target_prob + pred_sq
        if sample_weight is None:
            return float(np.sum(per_row)), float(pred_cpu.shape[0])
        return float(np.sum(sample_weight * per_row)), float(np.sum(sample_weight))

    def project_prediction(self, pred_cpu: np.ndarray) -> np.ndarray:
        return pred_cpu

    def plot_config(self, n_features: int, plot_mode: str = "auto") -> dict:
        if plot_mode == "feature_target_mean":
            pairs = []
            for target_idx in range(min(4, self.prediction_dim)):
                feature_idx = target_idx % max(n_features, 1)
                pairs.append((feature_idx, target_idx))
            return {
                "mode": "feature_target_mean",
                "pairs": pairs,
            }
        return {
            "mode": "class_density",
            "feature_indices": list(range(min(4, n_features))),
        }


@dataclass
class PoissonFamily:
    prediction_dim: int
    class_weights: np.ndarray
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

    def make_training_batch(self, bins_cpu: np.ndarray, batch: StreamBatch) -> CachedTrainingBatch:
        target_stats = batch.target_stats.astype(np.float32, copy=False)
        if np.any(target_stats < 0.0):
            warnings.warn("Negative Poisson target statistics encountered; clipping to zero.", RuntimeWarning)
            target_stats = np.maximum(target_stats, 0.0)
        sample_weight = None
        if self.use_weights:
            total_mass = np.sum(target_stats, axis=1)
            sample_weight = np.maximum(total_mass, 1.0).astype(np.float32, copy=False)
        return CachedTrainingBatch(
            bins=bins_cpu,
            target_stats=target_stats,
            sample_weight=sample_weight,
        )

    def fit_representation(self, batch: CachedTrainingBatch) -> np.ndarray:
        return batch.target_stats

    def uses_target_codes(self, batch: CachedTrainingBatch) -> bool:
        return False

    def leaf_value(self, target_stat_sum: np.ndarray, total_weight: float, reg_lambda: float) -> np.ndarray:
        return np.maximum(target_stat_sum / (total_weight + reg_lambda), self.clip_epsilon)

    def leaf_score(self, target_stat_sum: np.ndarray, total_weight: float, reg_lambda: float) -> float:
        if total_weight <= 0.0:
            return -np.inf
        mean = np.maximum(target_stat_sum / total_weight, self.clip_epsilon)
        return float(np.dot(target_stat_sum, np.log(mean)) - total_weight * np.sum(mean))

    def total_weight_from_stats(self, target_stat_sum: np.ndarray, count: int) -> float:
        return float(count)

    def project_prediction(self, pred_cpu: np.ndarray) -> np.ndarray:
        pred_proj = np.maximum(pred_cpu, self.clip_epsilon)
        if np.any(pred_cpu < 0.0):
            warnings.warn("Negative Poisson predictions encountered; clipping to epsilon.", RuntimeWarning)
        return pred_proj

    def monitor_metric(self, pred_cpu: np.ndarray, target_stats_cpu: np.ndarray, sample_weight: np.ndarray | None = None) -> tuple[float, float]:
        pred_proj = self.project_prediction(pred_cpu).astype(np.float64, copy=False)
        y64 = target_stats_cpu.astype(np.float64, copy=False)
        lgamma = np.vectorize(math.lgamma)
        per_row = np.sum(pred_proj - y64 * np.log(pred_proj) + lgamma(y64 + 1.0), axis=1)
        if sample_weight is None:
            return float(np.sum(per_row)), float(pred_cpu.shape[0])
        return float(np.sum(sample_weight * per_row)), float(np.sum(sample_weight))

    def plot_config(self, n_features: int, plot_mode: str = "auto") -> dict:
        pairs = []
        for target_idx in range(min(4, self.prediction_dim)):
            feature_idx = target_idx % max(n_features, 1)
            pairs.append((feature_idx, target_idx))
        return {
            "mode": "feature_target_mean",
            "pairs": pairs,
        }


def family_from_configs(tree_config: dict, dataset_config: dict):
    family_name = tree_config.get("family", "normal_identity")
    if family_name == "normal_identity":
        return NormalIdentityFamily.from_configs(tree_config, dataset_config)
    if family_name == "poisson":
        return PoissonFamily.from_configs(tree_config, dataset_config)
    raise ValueError("family must be 'normal_identity' or 'poisson'.")
