from __future__ import annotations

import argparse
import ast
import os

from numba import set_num_threads

from cpu_single_tree_trainer import CpuSingleTreeTrainer
from families import family_class_from_name, family_from_configs
from gpu_single_tree_trainer import GpuSingleTreeTrainer
from plot_feature_ratios import make_family_diagnostic_plots


TREE_CONFIG = {
    "max_bin": 64,
    "cut_sample_rows": 200000,
    "grow_policy": "depthwise",
    "max_depth": 2,
    "max_leaves": 4,
    "min_samples_leaf": 512,
    "min_split_loss": 1e-3,
    "reg_lambda": 0.0,
    "family": "normal_identity",
    "class_weights": None,
}

DATASET_CONFIG = {
    "n_features": 32,
    "n_classes": 4,
    "batch_size": 65536,
    "n_batches": 12,
    "seed": 0,
    "feature_offset_scale": 2.5,
    "feature_noise": 1.0,
}

TRAINING_CONFIG = {
    "plot_training_id": "single_tree_demo",
    "plot_bins": 80,
    "plot_mode": "all",
    "threads_per_block": 128,
    "training_backend": "auto",
    "cpu_threads": 0,
    "predict_method": "cpu",
    "cpu_predictor": "numba_parallel",
    "n_boost_rounds": 2,
    "learning_rate": 1.0,
    "fresh_inference_batch_size": None,
    "fresh_inference_n_batches": None,
}

CONFIG_GROUPS = {
    "tree": TREE_CONFIG,
    "dataset": DATASET_CONFIG,
    "training": TRAINING_CONFIG,
}


def _cast_override(value_text: str, default_value):
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


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Boosted histogram-tree demo with optional profiling, plotting, and tree printing."
    )
    parser.add_argument(
        "--modify",
        nargs="*",
        default=[],
        help="Override config entries as key value pairs, e.g. --modify max_depth 6 predict_method gpu",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile training and evaluation. Plotting and tree printing stay off unless requested.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate diagnostic plots under ./plots/<plot_training_id>/.",
    )
    parser.add_argument(
        "--print-trees",
        action="store_true",
        help="Print the fitted trees after the run.",
    )
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="Compatibility alias for --plot --print-trees.",
    )
    args, _unknown = parser.parse_known_args()

    all_config_keys = {}
    for group_name, group in CONFIG_GROUPS.items():
        for key in group:
            if key in all_config_keys:
                raise ValueError(
                    f"Duplicate config key '{key}' in '{group_name}' and '{all_config_keys[key]}'."
                )
            all_config_keys[key] = group_name

    if len(args.modify) % 2 != 0:
        raise ValueError("--modify expects an even number of arguments: key1 value1 key2 value2 ...")

    for key, value_text in zip(args.modify[0::2], args.modify[1::2]):
        if key not in all_config_keys:
            raise KeyError(f"Unknown config key '{key}'.")
        group = CONFIG_GROUPS[all_config_keys[key]]
        group[key] = _cast_override(value_text, group.get(key))

    if TREE_CONFIG.get("grow_policy") not in {"depthwise", "lossguide"}:
        raise ValueError("grow_policy must be 'depthwise' or 'lossguide'.")
    if TREE_CONFIG.get("family") not in {"normal_identity", "heteroskedastic_normal", "heteroskedastic_normal_ngd", "poisson", "poisson_mgd", "poisson_ngd"}:
        raise ValueError("family must be 'normal_identity', 'heteroskedastic_normal', 'heteroskedastic_normal_ngd', 'poisson', 'poisson_mgd', or 'poisson_ngd'.")
    if TRAINING_CONFIG.get("training_backend") not in {"auto", "gpu", "cpu"}:
        raise ValueError("training_backend must be 'auto', 'gpu', or 'cpu'.")
    if TRAINING_CONFIG.get("predict_method") not in {"cpu", "gpu"}:
        raise ValueError("predict_method must be 'cpu' or 'gpu'.")
    if TRAINING_CONFIG.get("cpu_predictor") not in {"index", "leaf_mask", "numba", "numba_parallel"}:
        raise ValueError("cpu_predictor must be 'index', 'leaf_mask', 'numba', or 'numba_parallel'.")
    if args.full_output:
        args.plot = True
        args.print_trees = True
    return args


def _default_cpu_threads() -> int:
    return 1


def _resolve_training_backend(requested_backend: str) -> str:
    if requested_backend != "auto":
        return requested_backend
    try:
        from numba import cuda

        if cuda.is_available():
            return "gpu"
    except Exception:
        pass
    return "cpu"


def _apply_example_defaults(modified_keys: set[str]) -> None:
    family_cls = family_class_from_name(TREE_CONFIG.get("family", "normal_identity"))
    for group_name, overrides in family_cls.example_defaults().items():
        group = CONFIG_GROUPS[group_name]
        for key, value in overrides.items():
            if key not in modified_keys:
                group[key] = value


ARGS = _parse_args()
MODIFIED_KEYS = set(ARGS.modify[0::2])
_apply_example_defaults(MODIFIED_KEYS)
RESOLVED_TRAINING_BACKEND = _resolve_training_backend(TRAINING_CONFIG.get("training_backend"))
TRAINING_CONFIG["training_backend"] = RESOLVED_TRAINING_BACKEND
if TRAINING_CONFIG.get("cpu_threads", 0) <= 0:
    TRAINING_CONFIG["cpu_threads"] = _default_cpu_threads()
if TRAINING_CONFIG.get("training_backend") == "cpu" and TRAINING_CONFIG.get("predict_method") == "gpu":
    raise ValueError("training_backend=cpu currently requires predict_method=cpu.")
set_num_threads(int(TRAINING_CONFIG.get("cpu_threads")))
FAMILY = family_from_configs(TREE_CONFIG, DATASET_CONFIG)
if TRAINING_CONFIG.get("training_backend") == "gpu":
    TRAINER = GpuSingleTreeTrainer(TREE_CONFIG, DATASET_CONFIG, TRAINING_CONFIG, FAMILY)
else:
    TRAINER = CpuSingleTreeTrainer(TREE_CONFIG, DATASET_CONFIG, TRAINING_CONFIG, FAMILY)


def main():
    print("Training backend:", TRAINING_CONFIG.get("training_backend"))
    print("Device:", TRAINER.device_name)
    print("Tree config:", TREE_CONFIG)
    print("Dataset config:", DATASET_CONFIG)
    print("Training config:", TRAINING_CONFIG)
    print("Family:", FAMILY.name)
    print("Class weights:", None if FAMILY.class_weights is None else FAMILY.class_weights.tolist())
    print(
        f"Building a boosted tree ensemble with grow_policy={TREE_CONFIG.get('grow_policy')}, "
        f"max_depth={TREE_CONFIG.get('max_depth')}, max_leaves={TREE_CONFIG.get('max_leaves')}, "
        f"max_bin={TREE_CONFIG.get('max_bin')}, rounds={TRAINING_CONFIG.get('n_boost_rounds')}"
    )

    model, provider_kwargs, train_metric, mean_sum_prob, loss_history = TRAINER.run(profile=ARGS.profile)
    print()
    print(f"Built boosted ensemble with {len(model.trees)} trees.")
    print(f"Objective: {FAMILY.name}")
    print(f"Initial train {FAMILY.monitor_name}: {loss_history[0]:.6f}")
    print(f"Final streamed train {FAMILY.monitor_name}: {train_metric:.6f}")
    print(f"Mean sum of predictions: {mean_sum_prob:.6f}")
    print(f"Prediction method: {TRAINING_CONFIG.get('predict_method')}")
    if TRAINING_CONFIG.get("predict_method") == "cpu":
        print(f"CPU predictor: {TRAINING_CONFIG.get('cpu_predictor')}")
    return model, provider_kwargs


if __name__ == "__main__":
    model, provider_kwargs = main()
    if ARGS.print_trees:
        print()
        for tree_idx, tree in enumerate(model.trees, start=1):
            print(f"Tree {tree_idx}:")
            tree.print_tree()
            print()
    if ARGS.plot:
        make_family_diagnostic_plots(
            training_id=TRAINING_CONFIG.get("plot_training_id"),
            provider_class=FAMILY.provider_class,
            provider_kwargs=provider_kwargs,
            predictor=lambda x: model.predict_batch(
                x,
                predict_method=TRAINING_CONFIG.get("predict_method"),
                gpu_predictor=lambda batch: TRAINER.predict_model_batch(model, batch),
                predict_from_state=FAMILY.predict_from_state,
                cpu_predictor=TRAINING_CONFIG.get("cpu_predictor"),
            ),
            n_classes=DATASET_CONFIG.get("n_classes"),
            plot_mode=TRAINING_CONFIG.get("plot_mode"),
            n_bins=TRAINING_CONFIG.get("plot_bins"),
        )
        print()
        print(f"Saved validation plots under ./plots/{TRAINING_CONFIG.get('plot_training_id')}/")
