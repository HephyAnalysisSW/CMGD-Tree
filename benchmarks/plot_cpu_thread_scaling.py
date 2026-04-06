from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CPU thread-scaling benchmark results.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_results(results_dir: Path):
    results_by_depth: dict[int, dict[int, list[dict]]] = {}
    for path in sorted(results_dir.glob("depth_*_threads_*_n_batches_*.txt")):
        with path.open("r", encoding="utf-8") as fin:
            payload = json.load(fin)
        depth = int(payload["config"]["max_depth"])
        cpu_threads = int(payload["config"]["cpu_threads"])
        n_events = int(payload["config"]["train_batch_size"]) * int(payload["config"]["train_n_batches"])
        results_by_depth.setdefault(depth, {}).setdefault(cpu_threads, []).append(
            {
                "n_events": n_events,
                "train_wall": float(payload["ours"]["train_wall"]),
                "fresh_wall": float(payload["ours"]["fresh_wall"]),
                "train_mse": float(payload["ours"]["train_mse"]),
            }
        )
    for depth in results_by_depth:
        for cpu_threads in results_by_depth[depth]:
            results_by_depth[depth][cpu_threads].sort(key=lambda item: item["n_events"])
    return results_by_depth


def make_timing_plot(depth: int, results_by_threads: dict[int, list[dict]], output_dir: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    thread_handles = []
    style_handles = []

    for color, cpu_threads in zip(colors, sorted(results_by_threads)):
        rows = results_by_threads[cpu_threads]
        n_events = [row["n_events"] for row in rows]
        train_wall = [row["train_wall"] for row in rows]
        fresh_wall = [row["fresh_wall"] for row in rows]
        thread_handles.append(
            ax.plot(n_events, train_wall, color=color, linestyle="-", marker="o", label=f"{cpu_threads} threads")[0]
        )
        ax.plot(n_events, fresh_wall, color=color, linestyle="--", marker="o")

    style_handles.append(ax.plot([], [], color="black", linestyle="-", label="training")[0])
    style_handles.append(ax.plot([], [], color="black", linestyle="--", label="fresh inference")[0])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N events")
    ax.set_ylabel("Wall time [s]")
    ax.set_title(f"CPU thread scaling, depth {depth}")
    ax.grid(True, which="both", alpha=0.25)

    legend_threads = ax.legend(handles=thread_handles, loc="upper left", title="threads")
    ax.add_artist(legend_threads)
    ax.legend(handles=style_handles, loc="lower right", title="curve")

    fig.tight_layout()
    png_path = output_dir / f"depth_{depth}_cpu_thread_scaling.png"
    pdf_path = output_dir / f"depth_{depth}_cpu_thread_scaling.pdf"
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)


def make_metric_plot(depth: int, results_by_threads: dict[int, list[dict]], output_dir: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    for color, cpu_threads in zip(colors, sorted(results_by_threads)):
        rows = results_by_threads[cpu_threads]
        n_events = [row["n_events"] for row in rows]
        train_mse = [row["train_mse"] for row in rows]
        ax.plot(n_events, train_mse, color=color, linestyle="-", marker="o", label=f"{cpu_threads} threads")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N events")
    ax.set_ylabel("Train MSE")
    ax.set_title(f"CPU thread scaling, depth {depth}, train MSE")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", title="threads")

    fig.tight_layout()
    png_path = output_dir / f"depth_{depth}_cpu_thread_train_mse.png"
    pdf_path = output_dir / f"depth_{depth}_cpu_thread_train_mse.pdf"
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

    for depth, results_by_threads in sorted(results_by_depth.items()):
        make_timing_plot(depth, results_by_threads, output_dir)
        make_metric_plot(depth, results_by_threads, output_dir)


if __name__ == "__main__":
    main()
