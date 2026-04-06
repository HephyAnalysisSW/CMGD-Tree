from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingCacheBatch:
    bins: np.ndarray
    target_stats: np.ndarray
    sample_weight: np.ndarray | None
    state: np.ndarray


class TrainingCache:
    def __init__(self, batches: list[TrainingCacheBatch], prediction_dim: int):
        self.batches = batches
        self.prediction_dim = prediction_dim

    def __iter__(self):
        return iter(self.batches)

    def initialize_states(self, base_state: np.ndarray):
        base = np.asarray(base_state, dtype=np.float32)
        for batch in self.batches:
            batch.state[...] = base

    def target_stat_mean(self) -> np.ndarray:
        total = np.zeros((self.prediction_dim,), dtype=np.float64)
        denominator = 0.0
        for batch in self.batches:
            if batch.sample_weight is None:
                total += np.sum(batch.target_stats, axis=0, dtype=np.float64)
                denominator += batch.target_stats.shape[0]
            else:
                total += np.sum(batch.sample_weight[:, None] * batch.target_stats, axis=0, dtype=np.float64)
                denominator += float(np.sum(batch.sample_weight))
        return (total / max(denominator, 1.0)).astype(np.float32)

    def monitoring_loss(self, family) -> tuple[float, float, float]:
        total_error = 0.0
        total_weight = 0.0
        total_sum = 0.0
        total_count = 0
        for batch in self.batches:
            error_sum, denominator = family.monitoring_loss(batch.state, batch.target_stats, batch.sample_weight)
            total_error += error_sum
            total_weight += denominator
            total_sum += float(np.sum(family.predict_from_state(batch.state)))
            total_count += batch.state.shape[0]
        return total_error / max(total_weight, 1.0), total_sum / max(total_count, 1), total_weight
