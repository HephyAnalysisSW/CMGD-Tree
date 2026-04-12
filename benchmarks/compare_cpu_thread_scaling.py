from __future__ import annotations

import ast
import json
import time
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from numba import set_num_threads

from core.cpu_single_tree_trainer import CpuSingleTreeTrainer
from families import family_from_configs


COMPARE_CONFIG = {
    "n_features": 4,
    "n_classes": 4,
    "train_batch_size": 65536,
    "train_n_batches": 12,
    "fresh_batch_size": 65536,
    "fresh_n_batches": 12,
    "seed": 0,
    "feature_offset_scale": 2.5,
    "feature_noise": 1.0,
    "max_depth": 3,
    "max_leaves": 8,
    "min_samples_leaf": 512,
    "min_split_loss": 1e-3,
    "learning_rate": 1.0,
    "n_boost_rounds": 100,
    "cpu_threads": 1,
    "cpu_predictor": "numba_parallel",
}


def cast_override(value_text: str, default_value):
    if isinstance(default_value, bool):
        lowered = value_text.lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean value from '{value_text}'.")
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(value_text)
    if isinstance(default_value, float):
        return float(value_text)
    if isinstance(default_value, str):
        return value_text
    try:
        return ast.literal_eval(value_text)
    except Exception:
        return value_text


def build_model():
    tree_config = {
        "max_bin": 64,
        "cut_sample_rows": 200000,
        "grow_policy": "depthwise",
        "max_depth": COMPARE_CONFIG["max_depth"],
        "max_leaves": COMPARE_CONFIG["max_leaves"],
        "min_samples_leaf": COMPARE_CONFIG["min_samples_leaf"],
        "min_split_loss": COMPARE_CONFIG["min_split_loss"],
        "reg_lambda": 0.0,
        "family": "normal_identity",
        "class_weights": None,
    }
    dataset_config = {
        "data_provider": "gaussian_class_toy",
        "n_features": COMPARE_CONFIG["n_features"],
        "n_classes": COMPARE_CONFIG["n_classes"],
        "batch_size": COMPARE_CONFIG["train_batch_size"],
        "n_batches": COMPARE_CONFIG["train_n_batches"],
        "seed": COMPARE_CONFIG["seed"],
        "feature_offset_scale": COMPARE_CONFIG["feature_offset_scale"],
        "feature_noise": COMPARE_CONFIG["feature_noise"],
    }
    training_config = {
        "threads_per_block": 128,
        "training_backend": "cpu",
        "cpu_threads": COMPARE_CONFIG["cpu_threads"],
        "predict_method": "cpu",
        "cpu_predictor": COMPARE_CONFIG["cpu_predictor"],
        "n_boost_rounds": COMPARE_CONFIG["n_boost_rounds"],
        "learning_rate": COMPARE_CONFIG["learning_rate"],
        "fresh_inference_batch_size": COMPARE_CONFIG["fresh_batch_size"],
        "fresh_inference_n_batches": COMPARE_CONFIG["fresh_n_batches"],
    }
    set_num_threads(training_config["cpu_threads"])
    family = family_from_configs(tree_config, dataset_config)
    trainer = CpuSingleTreeTrainer(tree_config, dataset_config, training_config, family)
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    model, _provider_kwargs, train_metric, _mean_sum_prob, _loss_history = trainer.run(profile=False)
    train_wall = time.perf_counter() - wall_start
    train_cpu = time.process_time() - cpu_start

    fresh_sum = 0.0
    fresh_rows = 0
    fresh_dataset_config = trainer.fresh_inference_dataset_config()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for batch in family.stream_batches(fresh_dataset_config):
        pred = model.predict_batch(
            batch.x,
            predict_method="cpu",
            predict_from_state=family.predict_from_state,
            cpu_predictor=COMPARE_CONFIG["cpu_predictor"],
        )
        fresh_sum += float(pred.sum())
        fresh_rows += pred.shape[0]
    fresh_wall = time.perf_counter() - wall_start
    fresh_cpu = time.process_time() - cpu_start

    total_nodes = sum(len(tree.nodes) for tree in model.trees)
    total_leaves = sum(tree.n_leaves for tree in model.trees)
    return {
        "train_wall": train_wall,
        "train_cpu": train_cpu,
        "train_mse": train_metric,
        "fresh_wall": fresh_wall,
        "fresh_cpu": fresh_cpu,
        "fresh_mean_sum": fresh_sum / max(fresh_rows, 1),
        "total_nodes": total_nodes,
        "total_leaves": total_leaves,
        "n_trees": len(model.trees),
    }


def print_stats(stats: dict):
    print("[cpu_thread_scaling]")
    print(
        f"  training: wall_s={stats['train_wall']:.3f} cpu_s={stats['train_cpu']:.3f} "
        f"cpu_pct_est={100.0 * stats['train_cpu'] / max(stats['train_wall'], 1e-9):.1f} train_mse={stats['train_mse']:.6f}"
    )
    print(
        f"  fresh_inference: wall_s={stats['fresh_wall']:.3f} cpu_s={stats['fresh_cpu']:.3f} "
        f"cpu_pct_est={100.0 * stats['fresh_cpu'] / max(stats['fresh_wall'], 1e-9):.1f} "
        f"mean_sum_prediction={stats['fresh_mean_sum']:.6f}"
    )
    print(
        f"  tree_size: n_trees={stats['n_trees']} total_nodes={stats['total_nodes']} total_leaves={stats['total_leaves']}"
    )


def write_results(result_path: str, stats: dict) -> None:
    payload = {
        "config": COMPARE_CONFIG,
        "ours": stats,
    }
    with open(result_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)
        fout.write("\n")


PARSER = argparse.ArgumentParser(description="Benchmark CPU training and fresh inference versus thread count.")
PARSER.add_argument("--modify", nargs="*", default=[])
PARSER.add_argument("--result-path", default=None)
ARGS, _UNKNOWN_ARGS = PARSER.parse_known_args()

if len(ARGS.modify) % 2 != 0:
    raise ValueError("--modify expects key value pairs.")
for KEY, VALUE_TEXT in zip(ARGS.modify[0::2], ARGS.modify[1::2]):
    if KEY not in COMPARE_CONFIG:
        raise KeyError(f"Unknown config key '{KEY}'.")
    COMPARE_CONFIG[KEY] = cast_override(VALUE_TEXT, COMPARE_CONFIG[KEY])

print("Compare config:", COMPARE_CONFIG)
STATS = build_model()
print()
print_stats(STATS)
if ARGS.result_path:
    write_results(ARGS.result_path, STATS)
