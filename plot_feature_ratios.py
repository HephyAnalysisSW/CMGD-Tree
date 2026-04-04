from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _collect_batches(provider_class, provider_kwargs: dict, predictor):
    x_batches = []
    y_batches = []
    pred_batches = []

    for batch in provider_class(**provider_kwargs):
        if hasattr(batch, "x"):
            x_cpu = batch.x
            y_cpu = batch.target_stats
        else:
            x_cpu, y_cpu = batch
        pred_cpu = predictor(x_cpu)
        x_batches.append(np.asarray(x_cpu, dtype=np.float32))
        y_batches.append(np.asarray(y_cpu, dtype=np.float32))
        pred_batches.append(np.asarray(pred_cpu, dtype=np.float32))

    return (
        np.concatenate(x_batches, axis=0),
        np.concatenate(y_batches, axis=0),
        np.concatenate(pred_batches, axis=0).astype(np.float64, copy=False),
    )


def _plot_class_density(
    out_dir: Path,
    training_id: str,
    x_all: np.ndarray,
    y_all: np.ndarray,
    pred_all: np.ndarray,
    n_classes: int,
    n_bins: int,
    feature_indices: list[int],
):
    truth_all = np.clip(y_all.astype(np.float64, copy=False), 0.0, None)
    pred_all = np.clip(pred_all, 0.0, None)
    cmap = plt.get_cmap("tab10")
    class_colors = [cmap(c % 10) for c in range(n_classes)]

    for j in feature_indices:
        fig, ax = plt.subplots(figsize=(7, 5))
        xj = x_all[:, j]

        for c in range(n_classes):
            color = class_colors[c]
            truth_weight = truth_all[:, c]
            pred_weight = pred_all[:, c]
            if np.sum(truth_weight) <= 0.0 and np.sum(pred_weight) <= 0.0:
                continue
            ax.hist(
                xj,
                bins=n_bins,
                density=True,
                histtype="step",
                linestyle="--",
                linewidth=2.0,
                color=color,
                weights=truth_weight if np.sum(truth_weight) > 0.0 else None,
                label=f"target {c} truth-weighted",
            )
            ax.hist(
                xj,
                bins=n_bins,
                density=True,
                histtype="step",
                linewidth=2.0,
                color=color,
                weights=pred_weight if np.sum(pred_weight) > 0.0 else None,
                label=f"target {c} pred-weighted",
            )

        ax.set_xlabel(f"feature {j}")
        ax.set_ylabel("density")
        ax.set_title(f"{training_id} : feature {j}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"feature{j}.png", dpi=150)
        plt.close(fig)


def _plot_feature_target_mean(
    out_dir: Path,
    training_id: str,
    x_all: np.ndarray,
    y_all: np.ndarray,
    pred_all: np.ndarray,
    n_bins: int,
    pairs: list[tuple[int, int]],
):
    for feature_idx, target_idx in pairs:
        fig, ax = plt.subplots(figsize=(7, 5))
        xj = x_all[:, feature_idx]
        yk = y_all[:, target_idx]
        muk = pred_all[:, target_idx]

        y_max = np.percentile(yk, 99.0) if yk.size else 1.0
        y_max = max(y_max, 1.0)
        ax.hist2d(
            xj,
            np.clip(yk, 0.0, y_max),
            bins=(n_bins, max(20, n_bins // 2)),
            cmap="Blues",
        )

        x_edges = np.linspace(float(np.min(xj)), float(np.max(xj)), n_bins + 1)
        centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        mean_obs = np.full((n_bins,), np.nan, dtype=np.float64)
        mean_pred = np.full((n_bins,), np.nan, dtype=np.float64)
        for idx in range(n_bins):
            if idx == n_bins - 1:
                mask = (xj >= x_edges[idx]) & (xj <= x_edges[idx + 1])
            else:
                mask = (xj >= x_edges[idx]) & (xj < x_edges[idx + 1])
            if np.any(mask):
                mean_obs[idx] = float(np.mean(yk[mask]))
                mean_pred[idx] = float(np.mean(muk[mask]))

        ax.plot(centers, mean_obs, color="black", linewidth=2.0, label="observed mean")
        ax.plot(centers, mean_pred, color="tab:red", linewidth=2.0, label="predicted mean")
        ax.set_xlabel(f"feature {feature_idx}")
        ax.set_ylabel(f"target {target_idx}")
        ax.set_title(f"{training_id} : x[{feature_idx}] vs y[{target_idx}]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"feature{feature_idx}_target{target_idx}.png", dpi=150)
        plt.close(fig)


def make_family_diagnostic_plots(
    training_id: str,
    provider_class,
    provider_kwargs: dict,
    predictor,
    n_classes: int,
    plot_config: dict,
    n_bins: int = 80,
) -> None:
    out_dir = Path("plots") / training_id
    out_dir.mkdir(parents=True, exist_ok=True)
    x_all, y_all, pred_all = _collect_batches(provider_class, provider_kwargs, predictor)

    if "modes" in plot_config:
        for sub_config in plot_config["modes"]:
            make_family_diagnostic_plots(
                training_id=training_id,
                provider_class=provider_class,
                provider_kwargs=provider_kwargs,
                predictor=predictor,
                n_classes=n_classes,
                plot_config=sub_config,
                n_bins=n_bins,
            )
        return

    if plot_config.get("mode") == "class_density":
        _plot_class_density(
            out_dir=out_dir,
            training_id=training_id,
            x_all=x_all,
            y_all=y_all,
            pred_all=pred_all,
            n_classes=n_classes,
            n_bins=n_bins,
            feature_indices=plot_config.get("feature_indices", list(range(min(4, x_all.shape[1])))),
        )
        return

    if plot_config.get("mode") == "feature_target_mean":
        _plot_feature_target_mean(
            out_dir=out_dir,
            training_id=training_id,
            x_all=x_all,
            y_all=y_all,
            pred_all=pred_all,
            n_bins=n_bins,
            pairs=plot_config.get("pairs", []),
        )
        return

    raise ValueError(f"Unknown plot mode '{plot_config.get('mode')}'.")
