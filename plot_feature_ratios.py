from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_feature_weighted_hist_plots(
    training_id: str,
    provider_class,
    provider_kwargs: dict,
    predictor,
    n_features: int,
    n_classes: int,
    n_bins: int = 80,
) -> None:
    """
    For each feature j, make a plot under ./plots/<training_id>/feature<j>.png
    with, for every class c:

      - dashed: true class-c density
      - solid: full sample weighted by predicted p(c|x)

    Truth and prediction of the same class are forced to use exactly the same color.
    """

    out_dir = Path("plots") / training_id
    out_dir.mkdir(parents=True, exist_ok=True)

    x_batches = []
    y_batches = []
    pred_batches = []

    for x_cpu, y_cpu in provider_class(**provider_kwargs):
        pred_cpu = predictor(x_cpu)
        x_batches.append(np.asarray(x_cpu, dtype=np.float32))
        y_batches.append(np.asarray(y_cpu, dtype=np.float32))
        pred_batches.append(np.asarray(pred_cpu, dtype=np.float32))

    x_all = np.concatenate(x_batches, axis=0)
    y_all = np.concatenate(y_batches, axis=0)
    pred_all = np.concatenate(pred_batches, axis=0).astype(np.float64, copy=False)

    pred_all = np.clip(pred_all, 1e-8, None)
    pred_all /= pred_all.sum(axis=1, keepdims=True)

    # Fixed color table. No implicit matplotlib cycling anywhere.
    cmap = plt.get_cmap("tab10")
    class_colors = [cmap(c % 10) for c in range(n_classes)]

    for j in range(n_features):
        fig, ax = plt.subplots(figsize=(7, 5))
        xj = x_all[:, j]

        for c in range(n_classes):
            color = class_colors[c]
            truth_mask = y_all[:, c] > 0.5

            # dashed: true class c
            ax.hist(
                xj[truth_mask],
                bins=n_bins,
                density=True,
                histtype="step",
                linestyle="--",
                linewidth=2.0,
                color=color,
                label=f"class {c} truth",
            )

            # solid: full sample weighted to class c
            ax.hist(
                xj,
                bins=n_bins,
                density=True,
                histtype="step",
                linewidth=2.0,
                color=color,
                weights=pred_all[:, c],
                label=f"all -> class {c} weighted",
            )

        ax.set_xlabel(f"feature {j}")
        ax.set_ylabel("density")
        ax.set_title(f"{training_id} : feature {j}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"feature{j}.png", dpi=150)
        plt.close(fig)
