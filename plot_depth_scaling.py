from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot depth-scaling benchmark results.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_results(results_dir: Path):
    results_by_depth: dict[int, list[dict]] = {}
    for path in sorted(results_dir.glob("depth_*_n_batches_*.txt")):
        with path.open("r", encoding="utf-8") as fin:
            payload = json.load(fin)
        depth = int(payload["config"]["max_depth"])
        n_events = int(payload["config"]["train_batch_size"]) * int(payload["config"]["train_n_batches"])
        results_by_depth.setdefault(depth, []).append(
            {
                "n_events": n_events,
                "ours_train": float(payload["ours"]["train_wall"]),
                "ours_infer": float(payload["ours"]["fresh_wall"]),
                "ours_train_mse": float(payload["ours"]["train_mse"]),
                "xgb_train": float(payload["xgboost"]["train_wall"]),
                "xgb_infer": float(payload["xgboost"]["fresh_wall"]),
                "xgb_train_mse": float(payload["xgboost"]["train_mse"]),
            }
        )
    for depth in results_by_depth:
        results_by_depth[depth].sort(key=lambda item: item["n_events"])
    return results_by_depth


def make_plot(depth: int, rows: list[dict], output_dir: Path):
    n_events = [row["n_events"] for row in rows]
    ours_train = [row["ours_train"] for row in rows]
    ours_infer = [row["ours_infer"] for row in rows]
    xgb_train = [row["xgb_train"] for row in rows]
    xgb_infer = [row["xgb_infer"] for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    method_handles = []
    linestyle_handles = []

    method_handles.append(ax.plot(n_events, xgb_train, color="tab:orange", linestyle="-", marker="s", label="xgboost")[0])
    ax.plot(n_events, xgb_infer, color="tab:orange", linestyle="--", marker="s")
    method_handles.append(ax.plot(n_events, ours_train, color="tab:blue", linestyle="-", marker="o", label="mgf-gpu")[0])
    ax.plot(n_events, ours_infer, color="tab:blue", linestyle="--", marker="o")

    linestyle_handles.append(ax.plot([], [], color="black", linestyle="-", label="training")[0])
    linestyle_handles.append(ax.plot([], [], color="black", linestyle="--", label="inference")[0])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N events")
    ax.set_ylabel("Wall time [s]")
    ax.set_title(f"Depth {depth} scaling")
    ax.grid(True, which="both", alpha=0.25)

    legend_methods = ax.legend(handles=method_handles, loc="upper left", title="method")
    ax.add_artist(legend_methods)
    ax.legend(handles=linestyle_handles, loc="lower right", title="curve")

    fig.tight_layout()
    png_path = output_dir / f"depth_{depth}_scaling.png"
    pdf_path = output_dir / f"depth_{depth}_scaling.pdf"
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)


def make_metric_plot(depth: int, rows: list[dict], output_dir: Path):
    n_events = [row["n_events"] for row in rows]
    ours_train_mse = [row["ours_train_mse"] for row in rows]
    xgb_train_mse = [row["xgb_train_mse"] for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(n_events, xgb_train_mse, color="tab:orange", linestyle="-", marker="s", label="xgboost")
    ax.plot(n_events, ours_train_mse, color="tab:blue", linestyle="-", marker="o", label="mgf-gpu")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N events")
    ax.set_ylabel("Train MSE")
    ax.set_title(f"Depth {depth} train MSE")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", title="method")

    fig.tight_layout()
    png_path = output_dir / f"depth_{depth}_train_mse.png"
    pdf_path = output_dir / f"depth_{depth}_train_mse.pdf"
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_depth = load_results(results_dir)
    if not results_by_depth:
        raise FileNotFoundError(f"No result files found under {results_dir}")

    for depth, rows in sorted(results_by_depth.items()):
        make_plot(depth, rows, output_dir)
        make_metric_plot(depth, rows, output_dir)


if __name__ == "__main__":
    main()
