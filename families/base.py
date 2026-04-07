from __future__ import annotations

from typing import Iterator

import numpy as np

from data_providers.base import StreamBatch


class BoostingFamily:
    """User-facing interface for uncurved boosting families.

    The trainer treats a family as a provider of:

    - target statistics `T(y)`
    - an initial model state
    - a map from model state to the dual prediction used in residuals
    - a pseudo-response / preconditioned target for tree fitting
    - an additive state update rule
    - a monitoring loss for reporting

    For the current MGD implementation, NGD-specific pieces collapse to simple
    identities:

    - the stored state is already the fitted coordinate
    - `dual_prediction(state)` is the identity map
    - `preconditioned_target(state, T)` is just `T - state`

    Later NGD families can keep the same trainer interface and override only
    these hooks with nonlinear coordinate maps and Fisher-preconditioned targets.
    """

    prediction_dim: int
    class_weights: np.ndarray | None
    use_weights: bool
    name: str
    monitor_name: str
    provider_class: type

    def provider_kwargs(self, dataset_config: dict) -> dict:
        """Return kwargs used to construct the streamed toy provider."""
        raise NotImplementedError

    def stream_batches(self, dataset_config: dict) -> Iterator[StreamBatch]:
        """Yield streamed training/evaluation batches."""
        raise NotImplementedError

    def base_state(self, target_stat_mean: np.ndarray) -> np.ndarray:
        """Return the initial model state for boosting."""
        raise NotImplementedError

    def project_state(self, state_cpu: np.ndarray) -> np.ndarray:
        """Project state values back into the valid domain if needed."""
        return state_cpu

    def predict_from_state(self, state_cpu: np.ndarray) -> np.ndarray:
        """Map the stored state to the user-facing prediction."""
        return self.project_state(state_cpu)

    def dual_prediction(self, state_cpu: np.ndarray) -> np.ndarray:
        """Map the stored state to the dual prediction used in residuals."""
        return self.predict_from_state(state_cpu)

    def preconditioned_target(self, state_cpu: np.ndarray, target_stats_cpu: np.ndarray) -> np.ndarray:
        """Return the regression target fitted by the tree."""
        return target_stats_cpu - self.dual_prediction(state_cpu)

    def apply_update(self, state_cpu: np.ndarray, tree_output_cpu: np.ndarray, learning_rate: float) -> np.ndarray:
        """Apply one additive tree update to the stored state in place."""
        state_cpu += learning_rate * tree_output_cpu
        state_cpu[...] = self.project_state(state_cpu)
        return state_cpu

    def monitoring_loss(
        self,
        state_cpu: np.ndarray,
        target_stats_cpu: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Return summed monitoring loss and its normalization weight."""
        raise NotImplementedError



def class_weight_vector(tree_config: dict, dim: int) -> tuple[np.ndarray | None, bool]:
    configured = tree_config.get("class_weights")
    if configured is None:
        return None, False
    class_weights = np.asarray(configured, dtype=np.float32)
    if class_weights.shape != (dim,):
        raise ValueError("class_weights must have length prediction_dim.")
    if np.any(class_weights < 0.0):
        raise ValueError("class_weights must be non-negative.")
    return class_weights, True
