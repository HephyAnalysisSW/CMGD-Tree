from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class StreamBatch:
    x: np.ndarray
    y: np.ndarray
    target_stats: np.ndarray
    sample_weight: np.ndarray | None = None


@dataclass
class CachedTrainingBatch:
    bins: np.ndarray
    y: np.ndarray
    target_stats: np.ndarray
    sample_weight: np.ndarray | None = None
    target_codes: np.ndarray | None = None


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
        y = self._class_targets[cls]
        return StreamBatch(x=x, y=y, target_stats=y)


@dataclass
class NormalIdentityFamily:
    prediction_dim: int
    class_weights: np.ndarray
    use_weights: bool
    name: str = "normal_identity"
    monitor_name: str = "MSE"

    @classmethod
    def from_configs(cls, tree_config: dict, dataset_config: dict) -> "NormalIdentityFamily":
        configured = tree_config.get("class_weights")
        n_classes = dataset_config.get("n_classes")
        if configured is None:
            return cls(
                prediction_dim=n_classes,
                class_weights=np.ones(n_classes, dtype=np.float32),
                use_weights=False,
            )

        class_weights = np.asarray(configured, dtype=np.float32)
        if class_weights.shape != (n_classes,):
            raise ValueError("class_weights must have length n_classes.")
        if np.any(class_weights < 0.0):
            raise ValueError("class_weights must be non-negative.")
        return cls(
            prediction_dim=n_classes,
            class_weights=class_weights,
            use_weights=True,
        )

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
        yield from GaussianClassToyStream(**self.provider_kwargs(dataset_config))

    def make_training_batch(self, bins_cpu: np.ndarray, batch: StreamBatch) -> CachedTrainingBatch:
        target_codes = np.argmax(batch.target_stats, axis=1).astype(
            np.int16 if batch.target_stats.shape[1] > 256 else np.uint8,
            copy=False,
        )
        if self.use_weights:
            sample_weight = self.class_weights[target_codes].astype(np.float32, copy=False)
            return CachedTrainingBatch(
                bins=bins_cpu,
                y=batch.y,
                target_stats=batch.target_stats,
                sample_weight=sample_weight,
                target_codes=target_codes.copy(),
            )
        return CachedTrainingBatch(
            bins=bins_cpu,
            y=batch.y,
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

    def monitor_metric(self, pred_cpu: np.ndarray, y_cpu: np.ndarray, sample_weight: np.ndarray | None = None) -> tuple[float, float]:
        pred_sq = np.sum(pred_cpu * pred_cpu, axis=1)
        target_prob = np.sum(pred_cpu * y_cpu, axis=1)
        per_row = 1.0 - 2.0 * target_prob + pred_sq
        if sample_weight is None:
            return float(np.sum(per_row)), float(pred_cpu.shape[0])
        return float(np.sum(sample_weight * per_row)), float(np.sum(sample_weight))
