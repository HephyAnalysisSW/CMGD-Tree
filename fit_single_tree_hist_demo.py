from __future__ import annotations

import argparse
import ast

from gpu_single_tree_trainer import GpuSingleTreeTrainer
from normal_identity_family import family_from_configs
from plot_feature_ratios import make_family_diagnostic_plots


TREE_CONFIG = {
    "max_bin": 64,
    "cut_sample_rows": 200000,
    "grow_policy": "depthwise",
    "max_depth": 4,
    "max_leaves": 16,
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
    "threads_per_block": 128,
    "predict_method": "cpu",
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
        description="Single-tree histogram trainer demo with optional profiling."
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
        help="Profile training and evaluation. Plotting is excluded.",
    )
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="Print the full tree and generate plots after the main run.",
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
    if TREE_CONFIG.get("family") not in {"normal_identity", "poisson"}:
        raise ValueError("family must be 'normal_identity' or 'poisson'.")
    if TRAINING_CONFIG.get("predict_method") not in {"cpu", "gpu"}:
        raise ValueError("predict_method must be 'cpu' or 'gpu'.")
    return args


ARGS = _parse_args()
FAMILY = family_from_configs(TREE_CONFIG, DATASET_CONFIG)
TRAINER = GpuSingleTreeTrainer(TREE_CONFIG, DATASET_CONFIG, TRAINING_CONFIG, FAMILY)


def main():
    print("GPU:", TRAINER.device_name)
    print("Tree config:", TREE_CONFIG)
    print("Dataset config:", DATASET_CONFIG)
    print("Training config:", TRAINING_CONFIG)
    print("Family:", FAMILY.name)
    print("Class weights:", FAMILY.class_weights.tolist())
    print(
        f"Building a single tree with grow_policy={TREE_CONFIG.get('grow_policy')}, "
        f"max_depth={TREE_CONFIG.get('max_depth')}, max_leaves={TREE_CONFIG.get('max_leaves')}, "
        f"max_bin={TREE_CONFIG.get('max_bin')}"
    )

    trained_tree, provider_kwargs, train_metric, mean_sum_prob = TRAINER.run(profile=ARGS.profile)
    print()
    print(f"Built tree with {len(trained_tree.nodes)} nodes and {trained_tree.n_leaves} leaves.")
    print(f"Objective: {FAMILY.name}")
    print(f"Root score: {trained_tree.root_score:.6f}")
    print(f"Root objective weight: {trained_tree.root_weight:.6f}")
    if FAMILY.use_weights:
        print(f"Final streamed train weighted {FAMILY.monitor_name}: {train_metric:.6f}")
    else:
        print(f"Final streamed train {FAMILY.monitor_name}: {train_metric:.6f}")
    print(f"Mean sum of class predictions: {mean_sum_prob:.6f}")
    print(f"Prediction method: {TRAINING_CONFIG.get('predict_method')}")
    return trained_tree, provider_kwargs


if __name__ == "__main__":
    trained_tree, provider_kwargs = main()
    if ARGS.full_output or not ARGS.profile:
        print()
        print("Tree:")
        trained_tree.print_tree()
        make_family_diagnostic_plots(
            training_id=TRAINING_CONFIG.get("plot_training_id"),
            provider_class=FAMILY.provider_class,
            provider_kwargs=provider_kwargs,
            predictor=lambda x: trained_tree.predict_batch(
                x,
                predict_method=TRAINING_CONFIG.get("predict_method"),
                gpu_predictor=lambda batch: TRAINER.predict_batch(trained_tree, x=batch),
            ),
            n_classes=DATASET_CONFIG.get("n_classes"),
            plot_config=FAMILY.plot_config(DATASET_CONFIG.get("n_features")),
            n_bins=TRAINING_CONFIG.get("plot_bins"),
        )
        print()
        print(f"Saved validation plots under ./plots/{TRAINING_CONFIG.get('plot_training_id')}/")
