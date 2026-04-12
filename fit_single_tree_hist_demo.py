from __future__ import annotations

import argparse
import ast
from pathlib import Path

import yaml
from numba import set_num_threads

from core.cpu_single_tree_trainer import CpuSingleTreeTrainer
from core.gpu_single_tree_trainer import GpuSingleTreeTrainer
from core.plot_feature_ratios import make_family_diagnostic_plots
from data_providers import build_data_provider
from families import family_from_configs


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


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


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
EXAMPLE_DIR = REPO_ROOT / "configs" / "examples"

PARSER = argparse.ArgumentParser(
    description="Boosted histogram-tree demo with optional profiling, plotting, and tree printing."
)
PARSER.add_argument("--config", default="normal_identity", help="Example name or YAML path.")
PARSER.add_argument("--modify", nargs="*", default=[], help="Override config entries as key value pairs.")
PARSER.add_argument("--profile", action="store_true", help="Profile training and evaluation.")
PARSER.add_argument("--plot", action="store_true", help="Generate diagnostic plots.")
PARSER.add_argument("--print-trees", action="store_true", help="Print the fitted trees after the run.")
PARSER.add_argument("--full-output", action="store_true", help="Compatibility alias for --plot --print-trees.")
ARGS, _UNKNOWN_ARGS = PARSER.parse_known_args()

CONFIG_PATH = Path(ARGS.config)
if not CONFIG_PATH.exists():
    CONFIG_PATH = EXAMPLE_DIR / f"{ARGS.config}.yaml"
CONFIG_PATH = CONFIG_PATH.resolve()

DEFAULTS = load_yaml(DEFAULT_CONFIG_PATH)
CONFIG = load_yaml(CONFIG_PATH)
GROUP_NAMES = ("tree", "dataset", "training", "plot")
GROUP_DEFAULTS = {name: dict(DEFAULTS.get(name, {})) for name in GROUP_NAMES}
CONFIG_GROUPS = {name: dict(CONFIG.get(name, {})) for name in GROUP_NAMES}
CONFIG_KEYS = {}

for GROUP_NAME in GROUP_NAMES:
    for KEY in set(GROUP_DEFAULTS[GROUP_NAME]) | set(CONFIG_GROUPS[GROUP_NAME]):
        CONFIG_KEYS[KEY] = GROUP_NAME

if len(ARGS.modify) % 2 != 0:
    raise ValueError("--modify expects key value pairs.")

for KEY, VALUE_TEXT in zip(ARGS.modify[0::2], ARGS.modify[1::2]):
    if KEY not in CONFIG_KEYS:
        raise KeyError(f"Unknown config key '{KEY}'.")
    GROUP_NAME = CONFIG_KEYS[KEY]
    DEFAULT_VALUE = CONFIG_GROUPS[GROUP_NAME].get(KEY, GROUP_DEFAULTS[GROUP_NAME].get(KEY))
    CONFIG_GROUPS[GROUP_NAME][KEY] = cast_override(VALUE_TEXT, DEFAULT_VALUE)

if ARGS.full_output:
    ARGS.plot = True
    ARGS.print_trees = True
if ARGS.plot:
    CONFIG_GROUPS["plot"]["enabled"] = True

TREE_CONFIG = dict(GROUP_DEFAULTS["tree"])
TREE_CONFIG.update(CONFIG_GROUPS["tree"])
DATASET_CONFIG = dict(GROUP_DEFAULTS["dataset"])
DATASET_CONFIG.update(CONFIG_GROUPS["dataset"])
TRAINING_CONFIG = dict(GROUP_DEFAULTS["training"])
TRAINING_CONFIG.update(CONFIG_GROUPS["training"])
PLOT_CONFIG = dict(GROUP_DEFAULTS["plot"])
PLOT_CONFIG.update(CONFIG_GROUPS["plot"])

if TRAINING_CONFIG.get("training_backend") == "auto":
    try:
        from numba import cuda

        TRAINING_CONFIG["training_backend"] = "gpu" if cuda.is_available() else "cpu"
    except Exception:
        TRAINING_CONFIG["training_backend"] = "cpu"
if TRAINING_CONFIG.get("cpu_threads", 0) <= 0:
    TRAINING_CONFIG["cpu_threads"] = 1

set_num_threads(int(TRAINING_CONFIG.get("cpu_threads")))
FAMILY = family_from_configs(TREE_CONFIG, DATASET_CONFIG)
if TRAINING_CONFIG.get("training_backend") == "gpu":
    TRAINER = GpuSingleTreeTrainer(TREE_CONFIG, DATASET_CONFIG, TRAINING_CONFIG, FAMILY)
else:
    TRAINER = CpuSingleTreeTrainer(TREE_CONFIG, DATASET_CONFIG, TRAINING_CONFIG, FAMILY)

print("Config file:", CONFIG_PATH)
print("Training backend:", TRAINING_CONFIG.get("training_backend"))
print("Device:", TRAINER.device_name)
print("Tree config:", TREE_CONFIG)
print("Dataset config:", DATASET_CONFIG)
print("Training config:", TRAINING_CONFIG)
print("Plot config:", PLOT_CONFIG)
print("Family:", FAMILY.name)
print("Class weights:", None if FAMILY.class_weights is None else FAMILY.class_weights.tolist())

MODEL, DATA_PROVIDER_KWARGS, TRAIN_METRIC, MEAN_SUM_PROB, LOSS_HISTORY = TRAINER.run(profile=ARGS.profile)
print()
print(f"Built boosted ensemble with {len(MODEL.trees)} trees.")
print(f"Objective: {FAMILY.name}")
print(f"Initial train {FAMILY.monitor_name}: {LOSS_HISTORY[0]:.6f}")
print(f"Final streamed train {FAMILY.monitor_name}: {TRAIN_METRIC:.6f}")
print(f"Mean sum of predictions: {MEAN_SUM_PROB:.6f}")
print(f"Prediction method: {TRAINING_CONFIG.get('predict_method')}")
if TRAINING_CONFIG.get("predict_method") == "cpu":
    print(f"CPU predictor: {TRAINING_CONFIG.get('cpu_predictor')}")

if ARGS.print_trees:
    print()
    for TREE_INDEX, TREE in enumerate(MODEL.trees, start=1):
        print(f"Tree {TREE_INDEX}:")
        TREE.print_tree()
        print()

if PLOT_CONFIG.get("enabled"):
    PLOT_PROVIDER = build_data_provider(DATASET_CONFIG, class_weights=FAMILY.class_weights)
    make_family_diagnostic_plots(
        training_id=PLOT_CONFIG.get("training_id"),
        provider=PLOT_PROVIDER,
        predictor=lambda x: MODEL.predict_batch(
            x,
            predict_method=TRAINING_CONFIG.get("predict_method"),
            gpu_predictor=lambda batch: TRAINER.predict_model_batch(MODEL, batch),
            predict_from_state=FAMILY.predict_from_state,
            cpu_predictor=TRAINING_CONFIG.get("cpu_predictor"),
        ),
        n_classes=DATASET_CONFIG.get("n_classes"),
        plot_config=PLOT_CONFIG,
    )
    print()
    print(f"Saved validation plots under ./plots/{PLOT_CONFIG.get('training_id')}/")
