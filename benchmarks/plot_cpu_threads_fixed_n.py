from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CPU thread scaling at fixed event counts.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_results(results_dir: Path):
    results_by_key: dict[tuple[int, int], list[dict]] = {}
    for path in sorted(results_dir.glob("depth_*_threads_*_n_batches_*.txt")):
        with path.open("r", encoding="utf-8") as fin:
            payload = json.load(fin)
        depth = int(payload["config"]["max_depth"])
        n_batches = int(payload["config"]["train_n_batches"])
        cpu_threads = int(payload["config"]["cpu_threads"])
        n_events = int(payload["config"]["train_batch_size"]) * n_batches
        results_by_key.setdefault((depth, n_events), []).append(
            {
                "cpu_threads": cpu_threads,
                "train_wall": float(payload["ours"]["train_wall"]),
                "fresh_wall": float(payload["ours"]["fresh_wall"]),
                "train_mse": float(payload["ours"]["train_mse"]),
            }
        )
    for key in results_by_key:
        results_by_key[key].sort(key=lambda item: item["cpu_threads"])
    return results_by_key


def make_timing_plot(depth: int, n_events: int, rows: list[dict], output_dir: Path):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    cpu_threads = [row["cpu_threads"] for row in rows]
    train_wall = [row["train_wall"] for row in rows]
    fresh_wall = [row["fresh_wall"] for row in rows]

    ax.plot(cpu_threads, train_wall, color="tab:blue", marker="o", linestyle="-", label="training")
    ax.plot(cpu_threads, fresh_wall, color="tab:orange", marker="o", linestyle="--", label="fresh inference")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(cpu_threads, labels=[str(x) for x in cpu_threads])
    ax.set_xlabel("CPU threads")
    ax.set_ylabel("Wall time [s]")
    ax.set_title(f"CPU thread scaling, depth {depth}, N={n_events}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    stem = f"depth_{depth}_n_events_{n_events}_cpu_threads"
    fig.savefig(output_dir / f"{stem}.png", dpi=160)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def make_metric_plot(depth: int, n_events: int, rows: list[dict], output_dir: Path):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    cpu_threads = [row["cpu_threads"] for row in rows]
    train_mse = [row["train_mse"] for row in rows]

    ax.plot(cpu_threads, train_mse, color="tab:green", marker="o", linestyle="-")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(cpu_threads, labels=[str(x) for x in cpu_threads])
    ax.set_xlabel("CPU threads")
    ax.set_ylabel("Train MSE")
    ax.set_title(f"CPU thread scaling, depth {depth}, N={n_events}, train MSE")
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    stem = f"depth_{depth}_n_events_{n_events}_cpu_threads_train_mse"
    fig.savefig(output_dir / f"{stem}.png", dpi=160)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_key = load_results(results_dir)
    if not results_by_key:
        raise FileNotFoundError(f"No result files found under {results_dir}")

    for (depth, n_events), rows in sorted(results_by_key.items()):
        make_timing_plot(depth, n_events, rows, output_dir)
        make_metric_plot(depth, n_events, rows, output_dir)


if __name__ == "__main__":
    main()
