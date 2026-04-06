from __future__ import annotations

import ast
import json
import time

import numpy as np
import xgboost as xgb

from gpu_single_tree_trainer import GpuSingleTreeTrainer
from normal_identity_family import GaussianClassToyStream, family_from_configs


COMPARE_CONFIG = {
    "n_features": 4,
    "n_classes": 4,
    "train_batch_size": 65536,
    "train_n_batches": 12,
    "fresh_batch_size": 262144,
    "fresh_n_batches": 64,
    "seed": 0,
    "feature_offset_scale": 2.5,
    "feature_noise": 1.0,
    "max_depth": 2,
    "max_leaves": 4,
    "min_samples_leaf": 512,
    "min_split_loss": 1e-3,
    "learning_rate": 1.0,
    "n_boost_rounds": 100,
    "cpu_predictor": "leaf_mask",
    "xgb_n_jobs": 1,
    "xgb_multi_strategy": "multi_output_tree",
}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Apples-to-apples single-core fresh inference comparison.")
    parser.add_argument("--modify", nargs="*", default=[])
    parser.add_argument("--result-path", default=None)
    args, _unknown = parser.parse_known_args()
    if len(args.modify) % 2 != 0:
        raise ValueError("--modify expects key value pairs.")
    for key, value_text in zip(args.modify[0::2], args.modify[1::2]):
        if key not in COMPARE_CONFIG:
            raise KeyError(f"Unknown config key '{key}'.")
        COMPARE_CONFIG[key] = cast_override(value_text, COMPARE_CONFIG[key])
    return args


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


def make_stream(batch_size: int, n_batches: int):
    return GaussianClassToyStream(
        n_features=COMPARE_CONFIG["n_features"],
        n_classes=COMPARE_CONFIG["n_classes"],
        batch_size=batch_size,
        n_batches=n_batches,
        feature_offset_scale=COMPARE_CONFIG["feature_offset_scale"],
        feature_noise=COMPARE_CONFIG["feature_noise"],
        seed=COMPARE_CONFIG["seed"],
    )


def make_train_arrays():
    x_batches = []
    y_batches = []
    for batch in make_stream(COMPARE_CONFIG["train_batch_size"], COMPARE_CONFIG["train_n_batches"]):
        x_batches.append(batch.x)
        y_batches.append(batch.target_stats)
    return np.concatenate(x_batches, axis=0), np.concatenate(y_batches, axis=0)


def mse_from_predictions(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.sum((pred - y) ** 2, axis=1)))


def count_xgb_tree_sizes(booster) -> tuple[int, int]:
    total_nodes = 0
    total_leaves = 0
    for tree_dump in booster.get_dump():
        lines = [line for line in tree_dump.splitlines() if line.strip()]
        total_nodes += len(lines)
        total_leaves += sum(":leaf=" in line for line in lines)
    return total_nodes, total_leaves


def build_our_model():
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
        "n_features": COMPARE_CONFIG["n_features"],
        "n_classes": COMPARE_CONFIG["n_classes"],
        "batch_size": COMPARE_CONFIG["train_batch_size"],
        "n_batches": COMPARE_CONFIG["train_n_batches"],
        "seed": COMPARE_CONFIG["seed"],
        "feature_offset_scale": COMPARE_CONFIG["feature_offset_scale"],
        "feature_noise": COMPARE_CONFIG["feature_noise"],
    }
    training_config = {
        "plot_training_id": "single_tree_demo",
        "plot_bins": 80,
        "plot_mode": "all",
        "threads_per_block": 128,
        "predict_method": "cpu",
        "cpu_predictor": COMPARE_CONFIG["cpu_predictor"],
        "n_boost_rounds": COMPARE_CONFIG["n_boost_rounds"],
        "learning_rate": COMPARE_CONFIG["learning_rate"],
        "fresh_inference_batch_size": COMPARE_CONFIG["fresh_batch_size"],
        "fresh_inference_n_batches": COMPARE_CONFIG["fresh_n_batches"],
    }
    family = family_from_configs(tree_config, dataset_config)
    trainer = GpuSingleTreeTrainer(tree_config, dataset_config, training_config, family)
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    model, _provider_kwargs, train_metric, _mean_sum_prob, _loss_history = trainer.run(profile=False)
    train_wall = time.perf_counter() - wall_start
    train_cpu = time.process_time() - cpu_start

    total_nodes = sum(len(tree.nodes) for tree in model.trees)
    total_leaves = sum(tree.n_leaves for tree in model.trees)

    fresh_sum = 0.0
    fresh_rows = 0
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for batch in make_stream(COMPARE_CONFIG["fresh_batch_size"], COMPARE_CONFIG["fresh_n_batches"]):
        pred = model.predict_batch(
            batch.x,
            predict_method="cpu",
            project_prediction=family.project_prediction,
            cpu_predictor=COMPARE_CONFIG["cpu_predictor"],
        )
        fresh_sum += float(np.sum(pred))
        fresh_rows += pred.shape[0]
    fresh_wall = time.perf_counter() - wall_start
    fresh_cpu = time.process_time() - cpu_start

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


def build_xgb_model():
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    train_x, train_y = make_train_arrays()
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=COMPARE_CONFIG["n_boost_rounds"],
        max_depth=COMPARE_CONFIG["max_depth"],
        max_leaves=COMPARE_CONFIG["max_leaves"],
        learning_rate=COMPARE_CONFIG["learning_rate"],
        tree_method="hist",
        device="cpu",
        multi_strategy=COMPARE_CONFIG["xgb_multi_strategy"],
        grow_policy="depthwise",
        min_child_weight=COMPARE_CONFIG["min_samples_leaf"],
        gamma=COMPARE_CONFIG["min_split_loss"],
        reg_lambda=0.0,
        subsample=1.0,
        colsample_bytree=1.0,
        n_jobs=COMPARE_CONFIG["xgb_n_jobs"],
        verbosity=0,
    )
    model.fit(train_x, train_y)
    train_wall = time.perf_counter() - wall_start
    train_cpu = time.process_time() - cpu_start
    booster = model.get_booster()
    pred_train = model.predict(train_x)
    train_mse = mse_from_predictions(pred_train, train_y)
    total_nodes, total_leaves = count_xgb_tree_sizes(booster)

    fresh_sum = 0.0
    fresh_rows = 0
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for batch in make_stream(COMPARE_CONFIG["fresh_batch_size"], COMPARE_CONFIG["fresh_n_batches"]):
        pred = booster.inplace_predict(batch.x)
        fresh_sum += float(np.sum(pred))
        fresh_rows += pred.shape[0]
    fresh_wall = time.perf_counter() - wall_start
    fresh_cpu = time.process_time() - cpu_start

    return {
        "train_wall": train_wall,
        "train_cpu": train_cpu,
        "train_mse": train_mse,
        "fresh_wall": fresh_wall,
        "fresh_cpu": fresh_cpu,
        "fresh_mean_sum": fresh_sum / max(fresh_rows, 1),
        "total_nodes": total_nodes,
        "total_leaves": total_leaves,
        "n_trees": len(booster.get_dump()),
    }


def print_stats(name: str, stats: dict):
    print(f"[{name}]")
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


def write_results(result_path: str, our_stats: dict, xgb_stats: dict) -> None:
    payload = {
        "config": COMPARE_CONFIG,
        "ours": our_stats,
        "xgboost": xgb_stats,
    }
    with open(result_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)
        fout.write("\n")


def main():
    args = parse_args()
    print("Compare config:", COMPARE_CONFIG)
    our_stats = build_our_model()
    xgb_stats = build_xgb_model()
    print()
    print_stats("ours", our_stats)
    print()
    print_stats("xgboost", xgb_stats)
    if args.result_path:
        write_results(args.result_path, our_stats, xgb_stats)


if __name__ == "__main__":
    main()
