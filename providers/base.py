from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StreamBatch:
    x: np.ndarray
    target_stats: np.ndarray
    sample_weight: np.ndarray | None = None


class PlotConfigProvider:
    def plot_config(self, plot_mode: str = "auto", n_bins: int = 80) -> dict:
        raise NotImplementedError
