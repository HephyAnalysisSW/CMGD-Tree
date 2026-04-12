from __future__ import annotations

import argparse
import ast
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required for YAML configs. Install it with `python -m pip install pyyaml`.") from exc
from numba import set_num_threads

from core.cpu_single_tree_trainer import CpuSingleTreeTrainer
from core.gpu_single_tree_trainer import GpuSingleTreeTrainer
from core.plot_feature_ratios import make_family_diagnostic_plots
from data_providers import DATA_PROVIDERS, build_data_provider
from families import family_from_configs


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Top level config in '{path}' must be a mapping.")
    return config


def _merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


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


def _resolve_config_path(config_arg: str, repo_root: Path) -> Path:
    config_path = Path(config_arg)
    if config_path.exists():
        return config_path.resolve()
    if not config_path.suffix:
        candidate = repo_root / "configs" / "examples" / f"{config_arg}.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find config '{config_arg}'.")


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


def _validate_config(tree_config: dict, dataset_config: dict, training_config: dict, plot_config: dict) -> None:
    if tree_config.get("grow_policy") not in {"depthwise", "lossguide"}:
        raise ValueError("grow_policy must be 'depthwise' or 'lossguide'.")
    if tree_config.get("family") not in {
        "normal_identity",
        "heteroskedastic_normal",
        "heteroskedastic_normal_ngd",
        "gamma",
        "gamma_mgd",
        "negative_binomial",
        "negative_binomial_mgd",
        "poisson",
        "poisson_mgd",
        "poisson_ngd",
    }:
        raise ValueError(
            "family must be 'normal_identity', 'heteroskedastic_normal', 'heteroskedastic_normal_ngd', "
            "'gamma', 'gamma_mgd', 'negative_binomial', 'negative_binomial_mgd', 'poisson', "
            "'poisson_mgd', or 'poisson_ngd'."
        )
    if dataset_config.get("data_provider") not in DATA_PROVIDERS:
        raise ValueError(f"data_provider must be one of {sorted(DATA_PROVIDERS)}.")
    if training_config.get("training_backend") not in {"auto", "gpu", "cpu"}:
        raise ValueError("training_backend must be 'auto', 'gpu', or 'cpu'.")
    if training_config.get("predict_method") not in {"cpu", "gpu"}:
        raise ValueError("predict_method must be 'cpu' or 'gpu'.")
    if training_config.get("cpu_predictor") not in {"index", "leaf_mask", "numba", "numba_parallel"}:
        raise ValueError("cpu_predictor must be 'index', 'leaf_mask', 'numba', or 'numba_parallel'.")
    if not isinstance(plot_config, dict):
        raise ValueError("plot config must be a mapping.")


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_EXAMPLE_CONFIG_PATH = REPO_ROOT / "configs" / "examples" / "normal_identity.yaml"

PARSER = argparse.ArgumentParser(
    description="Boosted histogram-tree demo with optional profiling, plotting, and tree printing."
)
PARSER.add_argument(
    "--config",
    default=str(DEFAULT_EXAMPLE_CONFIG_PATH),
    help="YAML config path or example name, e.g. --config poisson or --config configs/examples/poisson.yaml",
)
PARSER.add_argument(
    "--modify",
    nargs="*",
    default=[],
    help="Override config entries as key value pairs, e.g. --modify max_depth 6 predict_method gpu",
)
PARSER.add_argument(
    "--profile",
    action="store_true",
    help="Profile training and evaluation. Plotting and tree printing stay off unless requested.",
)
PARSER.add_argument(
    "--plot",
    action="store_true",
    help="Generate diagnostic plots under ./plots/<training_id>/.",
)
PARSER.add_argument(
    "--print-trees",
    action="store_true",
    help="Print the fitted trees after the run.",
)
PARSER.add_argument(
    "--full-output",
    action="store_true",
    help="Compatibility alias for --plot --print-trees.",
)

ARGS, _UNKNOWN_ARGS = PARSER.parse_known_args()
USER_CONFIG_PATH = _resolve_config_path(ARGS.config, REPO_ROOT)
CONFIG = _merge_dicts(_load_yaml(DEFAULT_CONFIG_PATH), _load_yaml(USER_CONFIG_PATH))
TREE_CONFIG = dict(CONFIG.get("tree", {}))
DATASET_CONFIG = dict(CONFIG.get("dataset", {}))
TRAINING_CONFIG = dict(CONFIG.get("training", {}))
PLOT_CONFIG = dict(CONFIG.get("plot", {}))
CONFIG_GROUPS = {
    "tree": TREE_CONFIG,
    "dataset": DATASET_CONFIG,
    "training": TRAINING_CONFIG,
    "plot": PLOT_CONFIG,
}

ALL_CONFIG_KEYS = {}
for GROUP_NAME, GROUP_CONFIG in CONFIG_GROUPS.items():
    for KEY in GROUP_CONFIG:
        if KEY in ALL_CONFIG_KEYS:
            raise ValueError(f"Duplicate config key '{KEY}' in '{GROUP_NAME}' and '{ALL_CONFIG_KEYS[KEY]}'.")
        ALL_CONFIG_KEYS[KEY] = GROUP_NAME

if len(ARGS.modify) % 2 != 0:
    raise ValueError("--modify expects an even number of arguments: key1 value1 key2 value2 ...")

for KEY, VALUE_TEXT in zip(ARGS.modify[0::2], ARGS.modify[1::2]):
    if KEY not in ALL_CONFIG_KEYS:
        raise KeyError(f"Unknown config key '{KEY}'.")
    GROUP = CONFIG_GROUPS[ALL_CONFIG_KEYS[KEY]]
    GROUP[KEY] = _cast_override(VALUE_TEXT, GROUP.get(KEY))

if ARGS.full_output:
    ARGS.plot = True
    ARGS.print_trees = True
if ARGS.plot:
    PLOT_CONFIG["enabled"] = True

_validate_config(TREE_CONFIG, DATASET_CONFIG, TRAINING_CONFIG, PLOT_CONFIG)
TRAINING_CONFIG["training_backend"] = _resolve_training_backend(TRAINING_CONFIG.get("training_backend"))
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

print("Config file:", USER_CONFIG_PATH)
print("Training backend:", TRAINING_CONFIG.get("training_backend"))
print("Device:", TRAINER.device_name)
print("Tree config:", TREE_CONFIG)
print("Dataset config:", DATASET_CONFIG)
print("Training config:", TRAINING_CONFIG)
print("Plot config:", PLOT_CONFIG)
print("Family:", FAMILY.name)
print("Class weights:", None if FAMILY.class_weights is None else FAMILY.class_weights.tolist())
print(
    f"Building a boosted tree ensemble with grow_policy={TREE_CONFIG.get('grow_policy')}, "
    f"max_depth={TREE_CONFIG.get('max_depth')}, max_leaves={TREE_CONFIG.get('max_leaves')}, "
    f"max_bin={TREE_CONFIG.get('max_bin')}, rounds={TRAINING_CONFIG.get('n_boost_rounds')}"
)

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
    for TREE_IDX, TREE in enumerate(MODEL.trees, start=1):
        print(f"Tree {TREE_IDX}:")
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
