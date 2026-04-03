from __future__ import annotations

import argparse
import ast
import os
import time
from dataclasses import dataclass

import cupy as cp
import numpy as np
from numba import cuda

from synthetic_provider import GaussianClassStreamProvider
from plot_feature_ratios import make_feature_weighted_hist_plots

try:
    import psutil
except Exception:
    psutil = None

try:
    import resource
except Exception:
    resource = None


# -----------------------------------------------------------------------------
# Hard-coded defaults. Later these can move into YAML files.
# -----------------------------------------------------------------------------
TREE_CONFIG = {
    "max_bin": 64,
    "cut_sample_rows": 200000,
    "grow_policy": "depthwise",        # "depthwise" or "lossguide"
    "max_depth": 4,
    "max_leaves": 16,
    "min_samples_leaf": 512,
    "min_split_loss": 1e-3,
    "reg_lambda": 0.0,
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
    "predict_method": "cpu",          # "cpu" or "gpu"
}


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
args, _unknown = parser.parse_known_args()


CONFIG_GROUPS = {
    "tree": TREE_CONFIG,
    "dataset": DATASET_CONFIG,
    "training": TRAINING_CONFIG,
}


all_config_keys = {}
for group_name, group in CONFIG_GROUPS.items():
    for key in group:
        if key in all_config_keys:
            raise ValueError(f"Duplicate config key '{key}' in '{group_name}' and '{all_config_keys[key]}'.")
        all_config_keys[key] = group_name

if len(args.modify) % 2 != 0:
    raise ValueError("--modify expects an even number of arguments: key1 value1 key2 value2 ...")


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


for key, value_text in zip(args.modify[0::2], args.modify[1::2]):
    if key not in all_config_keys:
        raise KeyError(f"Unknown config key '{key}'.")
    group_name = all_config_keys[key]
    group = CONFIG_GROUPS[group_name]
    group[key] = _cast_override(value_text, group[key])


MAX_BIN = TREE_CONFIG["max_bin"]
CUT_SAMPLE_ROWS = TREE_CONFIG["cut_sample_rows"]
GROW_POLICY = TREE_CONFIG["grow_policy"]
MAX_DEPTH = TREE_CONFIG["max_depth"]
MAX_LEAVES = TREE_CONFIG["max_leaves"]
MIN_SAMPLES_LEAF = TREE_CONFIG["min_samples_leaf"]
MIN_SPLIT_LOSS = TREE_CONFIG["min_split_loss"]
REG_LAMBDA = TREE_CONFIG["reg_lambda"]
CLASS_WEIGHTS_CONFIG = TREE_CONFIG["class_weights"]

N_FEATURES = DATASET_CONFIG["n_features"]
N_CLASSES = DATASET_CONFIG["n_classes"]
BATCH_SIZE = DATASET_CONFIG["batch_size"]
N_BATCHES = DATASET_CONFIG["n_batches"]
SEED = DATASET_CONFIG["seed"]
FEATURE_OFFSET_SCALE = DATASET_CONFIG["feature_offset_scale"]
FEATURE_NOISE = DATASET_CONFIG["feature_noise"]

PLOT_TRAINING_ID = TRAINING_CONFIG["plot_training_id"]
PLOT_BINS = TRAINING_CONFIG["plot_bins"]
THREADS_PER_BLOCK = TRAINING_CONFIG["threads_per_block"]
PREDICT_METHOD = TRAINING_CONFIG["predict_method"]
MAX_CLASS_CAPACITY = 16
BIN_NP_DTYPE = np.uint8 if MAX_BIN <= 256 else np.uint16
BIN_CP_DTYPE = cp.uint8 if MAX_BIN <= 256 else cp.uint16

if GROW_POLICY not in {"depthwise", "lossguide"}:
    raise ValueError("grow_policy must be 'depthwise' or 'lossguide'.")

if PREDICT_METHOD not in {"cpu", "gpu"}:
    raise ValueError("predict_method must be 'cpu' or 'gpu'.")

if CLASS_WEIGHTS_CONFIG is None:
    USE_WEIGHTED_OBJECTIVE = False
    CLASS_WEIGHTS_CPU = np.ones(N_CLASSES, dtype=np.float32)
else:
    USE_WEIGHTED_OBJECTIVE = True
    CLASS_WEIGHTS_CPU = np.asarray(CLASS_WEIGHTS_CONFIG, dtype=np.float32)
    if CLASS_WEIGHTS_CPU.shape != (N_CLASSES,):
        raise ValueError("class_weights must have length n_classes.")
    if np.any(CLASS_WEIGHTS_CPU < 0.0):
        raise ValueError("class_weights must be non-negative.")

OBJECTIVE_NAME = "weighted_mse" if USE_WEIGHTED_OBJECTIVE else "mse"

provider_kwargs = dict(
    n_features=N_FEATURES,
    n_classes=N_CLASSES,
    batch_size=BATCH_SIZE,
    n_batches=N_BATCHES,
    feature_offset_scale=FEATURE_OFFSET_SCALE,
    feature_noise=FEATURE_NOISE,
    seed=SEED,
)


@dataclass
class Node:
    node_id: int
    depth: int
    is_leaf: bool = True
    expandable: bool = True
    split_feature: int = -1
    split_bin: int = -1
    split_threshold: float = 0.0
    left_child: int = -1
    right_child: int = -1
    value: np.ndarray | None = None
    count: int = 0
    gain: float = -np.inf
    best_left_value: np.ndarray | None = None
    best_right_value: np.ndarray | None = None
    best_left_count: int = 0
    best_right_count: int = 0


@cuda.jit
def quantize_batch(x, cuts, bins_out):
    i, j = cuda.grid(2)
    if i < x.shape[0] and j < x.shape[1]:
        value = x[i, j]
        lo = 0
        hi = cuts.shape[1]
        while lo < hi:
            mid = (lo + hi) // 2
            if value <= cuts[j, mid]:
                hi = mid
            else:
                lo = mid + 1
        bins_out[i, j] = lo


@cuda.jit
def route_rows_to_candidate_slots(
    bins,
    split_feature,
    split_bin,
    left_child,
    right_child,
    is_leaf,
    candidate_slot_of_node,
    out_slot,
):
    i = cuda.grid(1)
    if i < bins.shape[0]:
        node = 0
        while is_leaf[node] == 0:
            feature = split_feature[node]
            threshold_bin = split_bin[node]
            if bins[i, feature] <= threshold_bin:
                node = left_child[node]
            else:
                node = right_child[node]
        out_slot[i] = candidate_slot_of_node[node]


@cuda.jit
def build_candidate_histograms(bins, cls, sample_weight, row_slot, hist_count, hist_sum):
    i = cuda.grid(1)
    if i < bins.shape[0]:
        slot = row_slot[i]
        if slot >= 0:
            weight = sample_weight[i]
            cls_i = cls[i]
            for f in range(bins.shape[1]):
                b = bins[i, f]
                cuda.atomic.add(hist_count, (slot, f, b), 1)
                cuda.atomic.add(hist_sum, (slot, f, b, cls_i), weight)


@cuda.jit
def build_candidate_histograms_unweighted(bins, cls, row_slot, hist_count, hist_sum):
    i = cuda.grid(1)
    if i < bins.shape[0]:
        slot = row_slot[i]
        if slot >= 0:
            cls_i = cls[i]
            for f in range(bins.shape[1]):
                b = bins[i, f]
                cuda.atomic.add(hist_count, (slot, f, b), 1)
                cuda.atomic.add(hist_sum, (slot, f, b, cls_i), 1.0)


@cuda.jit
def evaluate_feature_splits(
    hist_count,
    hist_sum,
    min_samples_leaf,
    reg_lambda,
    slot_parent_count,
    slot_parent_sum,
    feature_best_gain,
    feature_best_bin,
    feature_best_left_count,
    feature_best_right_count,
    feature_best_left_sum,
    feature_best_right_sum,
):
    slot, feature = cuda.grid(2)

    if slot < hist_count.shape[0] and feature < hist_count.shape[1]:
        parent_count = 0
        parent_score_num = 0.0

        for b in range(hist_count.shape[2]):
            parent_count += hist_count[slot, feature, b]

        if feature == 0:
            slot_parent_count[slot] = parent_count
            for c in range(hist_sum.shape[3]):
                s = 0.0
                for b in range(hist_sum.shape[2]):
                    s += hist_sum[slot, feature, b, c]
                slot_parent_sum[slot, c] = s

        if parent_count <= 0:
            feature_best_gain[slot, feature] = -1.0e30
            feature_best_bin[slot, feature] = -1
            feature_best_left_count[slot, feature] = 0
            feature_best_right_count[slot, feature] = 0
            for c in range(hist_sum.shape[3]):
                feature_best_left_sum[slot, feature, c] = 0.0
                feature_best_right_sum[slot, feature, c] = 0.0
            return

        parent_sum = cuda.local.array(16, dtype=np.float32)
        left_sum = cuda.local.array(16, dtype=np.float32)
        best_left_sum = cuda.local.array(16, dtype=np.float32)
        best_right_sum = cuda.local.array(16, dtype=np.float32)

        n_classes = hist_sum.shape[3]
        parent_weight = 0.0
        for c in range(n_classes):
            s = 0.0
            for b in range(hist_sum.shape[2]):
                s += hist_sum[slot, feature, b, c]
            parent_sum[c] = s
            left_sum[c] = 0.0
            parent_score_num += s * s
            parent_weight += s

        parent_score = parent_score_num / (parent_weight + reg_lambda)

        left_count = 0
        best_gain = -1.0e30
        best_bin = -1
        best_left_count = 0
        best_right_count = 0

        for split_bin in range(hist_count.shape[2] - 1):
            left_count += hist_count[slot, feature, split_bin]

            for c in range(n_classes):
                left_sum[c] += hist_sum[slot, feature, split_bin, c]

            right_count = parent_count - left_count
            left_weight = 0.0
            for c in range(n_classes):
                left_weight += left_sum[c]
            right_weight = parent_weight - left_weight

            if left_count < min_samples_leaf or right_count < min_samples_leaf or left_weight <= 0.0 or right_weight <= 0.0:
                continue

            left_score_num = 0.0
            right_score_num = 0.0
            for c in range(n_classes):
                right_sum_c = parent_sum[c] - left_sum[c]
                left_score_num += left_sum[c] * left_sum[c]
                right_score_num += right_sum_c * right_sum_c

            gain = (
                left_score_num / (left_weight + reg_lambda)
                + right_score_num / (right_weight + reg_lambda)
                - parent_score
            )

            if gain > best_gain:
                best_gain = gain
                best_bin = split_bin
                best_left_count = left_count
                best_right_count = right_count
                for c in range(n_classes):
                    best_left_sum[c] = left_sum[c]
                    best_right_sum[c] = parent_sum[c] - left_sum[c]

        feature_best_gain[slot, feature] = best_gain
        feature_best_bin[slot, feature] = best_bin
        feature_best_left_count[slot, feature] = best_left_count
        feature_best_right_count[slot, feature] = best_right_count
        for c in range(n_classes):
            feature_best_left_sum[slot, feature, c] = best_left_sum[c]
            feature_best_right_sum[slot, feature, c] = best_right_sum[c]


@cuda.jit
def evaluate_feature_splits_unweighted(
    hist_count,
    hist_sum,
    min_samples_leaf,
    reg_lambda,
    slot_parent_count,
    slot_parent_sum,
    feature_best_gain,
    feature_best_bin,
    feature_best_left_count,
    feature_best_right_count,
    feature_best_left_sum,
    feature_best_right_sum,
):
    slot, feature = cuda.grid(2)

    if slot < hist_count.shape[0] and feature < hist_count.shape[1]:
        parent_count = 0
        parent_score_num = 0.0

        for b in range(hist_count.shape[2]):
            parent_count += hist_count[slot, feature, b]

        if feature == 0:
            slot_parent_count[slot] = parent_count
            for c in range(hist_sum.shape[3]):
                s = 0.0
                for b in range(hist_sum.shape[2]):
                    s += hist_sum[slot, feature, b, c]
                slot_parent_sum[slot, c] = s

        if parent_count <= 0:
            feature_best_gain[slot, feature] = -1.0e30
            feature_best_bin[slot, feature] = -1
            feature_best_left_count[slot, feature] = 0
            feature_best_right_count[slot, feature] = 0
            for c in range(hist_sum.shape[3]):
                feature_best_left_sum[slot, feature, c] = 0.0
                feature_best_right_sum[slot, feature, c] = 0.0
            return

        parent_sum = cuda.local.array(16, dtype=np.float32)
        left_sum = cuda.local.array(16, dtype=np.float32)
        best_left_sum = cuda.local.array(16, dtype=np.float32)
        best_right_sum = cuda.local.array(16, dtype=np.float32)

        n_classes = hist_sum.shape[3]
        for c in range(n_classes):
            s = 0.0
            for b in range(hist_sum.shape[2]):
                s += hist_sum[slot, feature, b, c]
            parent_sum[c] = s
            left_sum[c] = 0.0
            parent_score_num += s * s

        parent_score = parent_score_num / (parent_count + reg_lambda)

        left_count = 0
        best_gain = -1.0e30
        best_bin = -1
        best_left_count = 0
        best_right_count = 0

        for split_bin in range(hist_count.shape[2] - 1):
            left_count += hist_count[slot, feature, split_bin]
            right_count = parent_count - left_count

            for c in range(n_classes):
                left_sum[c] += hist_sum[slot, feature, split_bin, c]

            if left_count < min_samples_leaf or right_count < min_samples_leaf:
                continue

            left_score_num = 0.0
            right_score_num = 0.0
            for c in range(n_classes):
                right_sum_c = parent_sum[c] - left_sum[c]
                left_score_num += left_sum[c] * left_sum[c]
                right_score_num += right_sum_c * right_sum_c

            gain = (
                left_score_num / (left_count + reg_lambda)
                + right_score_num / (right_count + reg_lambda)
                - parent_score
            )

            if gain > best_gain:
                best_gain = gain
                best_bin = split_bin
                best_left_count = left_count
                best_right_count = right_count
                for c in range(n_classes):
                    best_left_sum[c] = left_sum[c]
                    best_right_sum[c] = parent_sum[c] - left_sum[c]

        feature_best_gain[slot, feature] = best_gain
        feature_best_bin[slot, feature] = best_bin
        feature_best_left_count[slot, feature] = best_left_count
        feature_best_right_count[slot, feature] = best_right_count
        for c in range(n_classes):
            feature_best_left_sum[slot, feature, c] = best_left_sum[c]
            feature_best_right_sum[slot, feature, c] = best_right_sum[c]


@cuda.jit
def reduce_feature_bests(
    feature_best_gain,
    feature_best_bin,
    feature_best_left_count,
    feature_best_right_count,
    feature_best_left_sum,
    feature_best_right_sum,
    slot_best_gain,
    slot_best_feature,
    slot_best_bin,
    slot_best_left_count,
    slot_best_right_count,
    slot_best_left_sum,
    slot_best_right_sum,
):
    slot = cuda.grid(1)
    if slot < feature_best_gain.shape[0]:
        best_gain = -1.0e30
        best_feature = -1
        best_bin = -1
        best_left_count = 0
        best_right_count = 0

        for feature in range(feature_best_gain.shape[1]):
            gain = feature_best_gain[slot, feature]
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_bin = feature_best_bin[slot, feature]
                best_left_count = feature_best_left_count[slot, feature]
                best_right_count = feature_best_right_count[slot, feature]
                for c in range(feature_best_left_sum.shape[2]):
                    slot_best_left_sum[slot, c] = feature_best_left_sum[slot, feature, c]
                    slot_best_right_sum[slot, c] = feature_best_right_sum[slot, feature, c]

        slot_best_gain[slot] = best_gain
        slot_best_feature[slot] = best_feature
        slot_best_bin[slot] = best_bin
        slot_best_left_count[slot] = best_left_count
        slot_best_right_count[slot] = best_right_count


@cuda.jit
def predict_rows_gpu_kernel(
    x,
    split_feature,
    split_threshold,
    left_child,
    right_child,
    is_leaf,
    leaf_value,
    pred_out,
):
    i = cuda.grid(1)
    if i < x.shape[0]:
        node = 0
        while is_leaf[node] == 0:
            feature = split_feature[node]
            threshold = split_threshold[node]
            if x[i, feature] <= threshold:
                node = left_child[node]
            else:
                node = right_child[node]
        for c in range(pred_out.shape[1]):
            pred_out[i, c] = leaf_value[node, c]


@cuda.jit
def predict_rows_gpu_bins_kernel(
    bins,
    split_feature,
    split_bin,
    left_child,
    right_child,
    is_leaf,
    leaf_value,
    pred_out,
):
    i = cuda.grid(1)
    if i < bins.shape[0]:
        node = 0
        while is_leaf[node] == 0:
            feature = split_feature[node]
            threshold_bin = split_bin[node]
            if bins[i, feature] <= threshold_bin:
                node = left_child[node]
            else:
                node = right_child[node]
        for c in range(pred_out.shape[1]):
            pred_out[i, c] = leaf_value[node, c]


def weighted_mse_objective_leaf_value(sum_y: np.ndarray, weight_sum: float, reg_lambda: float) -> np.ndarray:
    return sum_y / (weight_sum + reg_lambda)


def mse_objective_leaf_value(sum_y: np.ndarray, count: int, reg_lambda: float) -> np.ndarray:
    return sum_y / (count + reg_lambda)


def weighted_mse_objective_leaf_score(sum_y: np.ndarray, weight_sum: float, reg_lambda: float) -> float:
    if weight_sum <= 0:
        return -np.inf
    return float(np.dot(sum_y, sum_y) / (weight_sum + reg_lambda))


def mse_objective_leaf_score(sum_y: np.ndarray, count: int, reg_lambda: float) -> float:
    if count <= 0:
        return -np.inf
    return float(np.dot(sum_y, sum_y) / (count + reg_lambda))


def objective_leaf_value(sum_y: np.ndarray, weight_sum: float, reg_lambda: float) -> np.ndarray:
    if OBJECTIVE_NAME == "mse":
        return mse_objective_leaf_value(sum_y, int(weight_sum), reg_lambda)
    if OBJECTIVE_NAME == "weighted_mse":
        return weighted_mse_objective_leaf_value(sum_y, weight_sum, reg_lambda)
    raise ValueError(f"Unsupported objective '{OBJECTIVE_NAME}'.")


def objective_leaf_score(sum_y: np.ndarray, weight_sum: float, reg_lambda: float) -> float:
    if OBJECTIVE_NAME == "mse":
        return mse_objective_leaf_score(sum_y, int(weight_sum), reg_lambda)
    if OBJECTIVE_NAME == "weighted_mse":
        return weighted_mse_objective_leaf_score(sum_y, weight_sum, reg_lambda)
    raise ValueError(f"Unsupported objective '{OBJECTIVE_NAME}'.")


def _rss_bytes():
    if psutil is not None:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    if resource is not None:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if os.uname().sysname.lower() == "darwin":
            return rss
        return rss * 1024
    return 0


def _gpu_snapshot():
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    mempool = cp.get_default_memory_pool()
    pinned_pool = cp.get_default_pinned_memory_pool()
    return {
        "gpu_used_bytes": int(total_bytes - free_bytes),
        "gpu_total_bytes": int(total_bytes),
        "gpu_pool_used_bytes": int(mempool.used_bytes()),
        "gpu_pool_total_bytes": int(mempool.total_bytes()),
        "gpu_pinned_free_blocks": int(pinned_pool.n_free_blocks()),
    }


def _start_profile(stage_name: str):
    gpu = _gpu_snapshot()
    return {
        "stage": stage_name,
        "wall_start": time.perf_counter(),
        "cpu_start": time.process_time(),
        "rss_start": _rss_bytes(),
        "rss_max": _rss_bytes(),
        "gpu_used_start": gpu["gpu_used_bytes"],
        "gpu_used_max": gpu["gpu_used_bytes"],
        "gpu_pool_used_start": gpu["gpu_pool_used_bytes"],
        "gpu_pool_used_max": gpu["gpu_pool_used_bytes"],
        "gpu_pool_total_start": gpu["gpu_pool_total_bytes"],
        "gpu_pool_total_max": gpu["gpu_pool_total_bytes"],
    }


def _update_profile(profile_state: dict):
    rss = _rss_bytes()
    gpu = _gpu_snapshot()
    if rss > profile_state["rss_max"]:
        profile_state["rss_max"] = rss
    if gpu["gpu_used_bytes"] > profile_state["gpu_used_max"]:
        profile_state["gpu_used_max"] = gpu["gpu_used_bytes"]
    if gpu["gpu_pool_used_bytes"] > profile_state["gpu_pool_used_max"]:
        profile_state["gpu_pool_used_max"] = gpu["gpu_pool_used_bytes"]
    if gpu["gpu_pool_total_bytes"] > profile_state["gpu_pool_total_max"]:
        profile_state["gpu_pool_total_max"] = gpu["gpu_pool_total_bytes"]


def _finish_profile(profile_state: dict):
    profile_state["wall_end"] = time.perf_counter()
    profile_state["cpu_end"] = time.process_time()
    profile_state["rss_end"] = _rss_bytes()
    gpu = _gpu_snapshot()
    profile_state["gpu_used_end"] = gpu["gpu_used_bytes"]
    profile_state["gpu_pool_used_end"] = gpu["gpu_pool_used_bytes"]
    profile_state["gpu_pool_total_end"] = gpu["gpu_pool_total_bytes"]
    _update_profile(profile_state)


def _mb(value_bytes: int) -> float:
    return float(value_bytes) / (1024.0 * 1024.0)


def _print_profile(profile_state: dict):
    wall = profile_state["wall_end"] - profile_state["wall_start"]
    cpu = profile_state["cpu_end"] - profile_state["cpu_start"]
    cpu_pct = 100.0 * cpu / wall if wall > 0.0 else 0.0

    print()
    print(f"[profile:{profile_state['stage']}]")
    print(f"  wall_s={wall:.3f} cpu_s={cpu:.3f} cpu_pct_est={cpu_pct:.1f}")
    print(
        "  rss_mb="
        f"start={_mb(profile_state['rss_start']):.1f} "
        f"end={_mb(profile_state['rss_end']):.1f} "
        f"max={_mb(profile_state['rss_max']):.1f}"
    )
    print(
        "  gpu_used_mb="
        f"start={_mb(profile_state['gpu_used_start']):.1f} "
        f"end={_mb(profile_state['gpu_used_end']):.1f} "
        f"max={_mb(profile_state['gpu_used_max']):.1f}"
    )
    print(
        "  gpu_pool_used_mb="
        f"start={_mb(profile_state['gpu_pool_used_start']):.1f} "
        f"end={_mb(profile_state['gpu_pool_used_end']):.1f} "
        f"max={_mb(profile_state['gpu_pool_used_max']):.1f}"
    )
    print(
        "  gpu_pool_reserved_mb="
        f"start={_mb(profile_state['gpu_pool_total_start']):.1f} "
        f"end={_mb(profile_state['gpu_pool_total_end']):.1f} "
        f"max={_mb(profile_state['gpu_pool_total_max']):.1f}"
    )


print("GPU:", cuda.get_current_device().name)
print("Tree config:", TREE_CONFIG)
print("Dataset config:", DATASET_CONFIG)
print("Training config:", TRAINING_CONFIG)
print("Objective:", OBJECTIVE_NAME)
print("Class weights:", CLASS_WEIGHTS_CPU.tolist())
print(
    f"Building a single tree with grow_policy={GROW_POLICY}, "
    f"max_depth={MAX_DEPTH}, max_leaves={MAX_LEAVES}, max_bin={MAX_BIN}"
)

training_profile = _start_profile("training") if args.profile else None


# -----------------------------------------------------------------------------
# Global feature cuts. Not profiled: this is setup / data-side work.
# -----------------------------------------------------------------------------
cut_batches = []
sampled_rows = 0
for x_cpu, _ in GaussianClassStreamProvider(**provider_kwargs):
    take = min(x_cpu.shape[0], CUT_SAMPLE_ROWS - sampled_rows)
    if take > 0:
        cut_batches.append(x_cpu[:take].copy())
        sampled_rows += take
    if sampled_rows >= CUT_SAMPLE_ROWS:
        break

cut_sample = np.concatenate(cut_batches, axis=0)
quantile_levels = np.linspace(0.0, 1.0, MAX_BIN + 1, dtype=np.float64)[1:-1]
cuts_cpu = np.quantile(cut_sample, quantile_levels, axis=0).T.astype(np.float32)
cuts_gpu = cp.asarray(cuts_cpu)

cache_build_profile = _start_profile("cache_build") if args.profile else None
quantized_train_batches = []
cache_x_gpu = None
cache_bins_gpu = None
for x_cpu, y_cpu in GaussianClassStreamProvider(**provider_kwargs):
    if cache_x_gpu is None or cache_x_gpu.shape != x_cpu.shape:
        cache_x_gpu = cp.empty(x_cpu.shape, dtype=cp.float32)
        cache_bins_gpu = cp.empty(x_cpu.shape, dtype=BIN_CP_DTYPE)
    cache_x_gpu.set(x_cpu)
    quant_blocks = (
        (cache_x_gpu.shape[0] + 15) // 16,
        (cache_x_gpu.shape[1] + 15) // 16,
    )
    quantize_batch[quant_blocks, (16, 16)](cache_x_gpu, cuts_gpu, cache_bins_gpu)
    cuda.synchronize()
    bins_cpu = cp.asnumpy(cache_bins_gpu)
    cls_cpu = np.argmax(y_cpu, axis=1).astype(np.int16 if N_CLASSES > 256 else np.uint8, copy=False)
    if USE_WEIGHTED_OBJECTIVE:
        sample_weight_cpu = CLASS_WEIGHTS_CPU[cls_cpu]
        quantized_train_batches.append((bins_cpu, cls_cpu.copy(), sample_weight_cpu.astype(np.float32, copy=False)))
    else:
        quantized_train_batches.append((bins_cpu, cls_cpu.copy()))
    if cache_build_profile is not None:
        _update_profile(cache_build_profile)

if cache_build_profile is not None:
    _finish_profile(cache_build_profile)
    _print_profile(cache_build_profile)


# -----------------------------------------------------------------------------
# Tree state.
# -----------------------------------------------------------------------------
nodes: list[Node] = [Node(node_id=0, depth=0)]
n_leaves = 1
next_node_id = 1
root_score = None
root_weight = None

tree_growth_profile = _start_profile("tree_growth") if args.profile else None


# -----------------------------------------------------------------------------
# Greedy tree growth.
# -----------------------------------------------------------------------------
while True:
    candidate_node_ids = [
        node.node_id
        for node in nodes
        if node.is_leaf and node.expandable and node.depth < MAX_DEPTH
    ]
    if not candidate_node_ids:
        break

    if GROW_POLICY == "depthwise":
        frontier_depth = min(nodes[node_id].depth for node_id in candidate_node_ids)
        candidate_node_ids = [
            node_id
            for node_id in candidate_node_ids
            if nodes[node_id].depth == frontier_depth
        ]

    candidate_slot_of_node_cpu = np.full(len(nodes), -1, dtype=np.int32)
    for slot, node_id in enumerate(candidate_node_ids):
        candidate_slot_of_node_cpu[node_id] = slot
    candidate_slot_of_node_gpu = cp.asarray(candidate_slot_of_node_cpu)

    split_feature_cpu = np.array([node.split_feature for node in nodes], dtype=np.int32)
    split_bin_cpu = np.array([node.split_bin for node in nodes], dtype=np.int32)
    left_child_cpu = np.array([node.left_child for node in nodes], dtype=np.int32)
    right_child_cpu = np.array([node.right_child for node in nodes], dtype=np.int32)
    is_leaf_cpu = np.array([1 if node.is_leaf else 0 for node in nodes], dtype=np.int8)

    split_feature_gpu = cp.asarray(split_feature_cpu)
    split_bin_gpu = cp.asarray(split_bin_cpu)
    left_child_gpu = cp.asarray(left_child_cpu)
    right_child_gpu = cp.asarray(right_child_cpu)
    is_leaf_gpu = cp.asarray(is_leaf_cpu)

    hist_count_gpu = cp.zeros((len(candidate_node_ids), N_FEATURES, MAX_BIN), dtype=cp.int32)
    hist_sum_gpu = cp.zeros((len(candidate_node_ids), N_FEATURES, MAX_BIN, N_CLASSES), dtype=cp.float32)

    for batch in quantized_train_batches:
        bins_cpu = batch[0]
        cls_cpu = batch[1]
        bins_gpu = cp.asarray(bins_cpu)
        cls_gpu = cp.asarray(cls_cpu)

        blocks_1d = (bins_gpu.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        row_slot_gpu = cp.empty((bins_gpu.shape[0],), dtype=cp.int32)
        route_rows_to_candidate_slots[blocks_1d, THREADS_PER_BLOCK](
            bins_gpu,
            split_feature_gpu,
            split_bin_gpu,
            left_child_gpu,
            right_child_gpu,
            is_leaf_gpu,
            candidate_slot_of_node_gpu,
            row_slot_gpu,
        )
        if USE_WEIGHTED_OBJECTIVE:
            sample_weight_gpu = cp.asarray(batch[2])
            build_candidate_histograms[blocks_1d, THREADS_PER_BLOCK](
                bins_gpu,
                cls_gpu,
                sample_weight_gpu,
                row_slot_gpu,
                hist_count_gpu,
                hist_sum_gpu,
            )
        else:
            build_candidate_histograms_unweighted[blocks_1d, THREADS_PER_BLOCK](
                bins_gpu,
                cls_gpu,
                row_slot_gpu,
                hist_count_gpu,
                hist_sum_gpu,
            )

        if tree_growth_profile is not None:
            _update_profile(tree_growth_profile)
        if training_profile is not None:
            _update_profile(training_profile)

    n_slots = len(candidate_node_ids)
    slot_parent_count_gpu = cp.zeros((n_slots,), dtype=cp.int32)
    slot_parent_sum_gpu = cp.zeros((n_slots, N_CLASSES), dtype=cp.float32)

    feature_best_gain_gpu = cp.full((n_slots, N_FEATURES), -1.0e30, dtype=cp.float32)
    feature_best_bin_gpu = cp.full((n_slots, N_FEATURES), -1, dtype=cp.int32)
    feature_best_left_count_gpu = cp.zeros((n_slots, N_FEATURES), dtype=cp.int32)
    feature_best_right_count_gpu = cp.zeros((n_slots, N_FEATURES), dtype=cp.int32)
    feature_best_left_sum_gpu = cp.zeros((n_slots, N_FEATURES, N_CLASSES), dtype=cp.float32)
    feature_best_right_sum_gpu = cp.zeros((n_slots, N_FEATURES, N_CLASSES), dtype=cp.float32)

    eval_blocks = (
        (n_slots + 7) // 8,
        (N_FEATURES + 7) // 8,
    )
    if USE_WEIGHTED_OBJECTIVE:
        evaluate_feature_splits[eval_blocks, (8, 8)](
            hist_count_gpu,
            hist_sum_gpu,
            MIN_SAMPLES_LEAF,
            REG_LAMBDA,
            slot_parent_count_gpu,
            slot_parent_sum_gpu,
            feature_best_gain_gpu,
            feature_best_bin_gpu,
            feature_best_left_count_gpu,
            feature_best_right_count_gpu,
            feature_best_left_sum_gpu,
            feature_best_right_sum_gpu,
        )
    else:
        evaluate_feature_splits_unweighted[eval_blocks, (8, 8)](
            hist_count_gpu,
            hist_sum_gpu,
            MIN_SAMPLES_LEAF,
            REG_LAMBDA,
            slot_parent_count_gpu,
            slot_parent_sum_gpu,
            feature_best_gain_gpu,
            feature_best_bin_gpu,
            feature_best_left_count_gpu,
            feature_best_right_count_gpu,
            feature_best_left_sum_gpu,
            feature_best_right_sum_gpu,
        )

    slot_best_gain_gpu = cp.full((n_slots,), -1.0e30, dtype=cp.float32)
    slot_best_feature_gpu = cp.full((n_slots,), -1, dtype=cp.int32)
    slot_best_bin_gpu = cp.full((n_slots,), -1, dtype=cp.int32)
    slot_best_left_count_gpu = cp.zeros((n_slots,), dtype=cp.int32)
    slot_best_right_count_gpu = cp.zeros((n_slots,), dtype=cp.int32)
    slot_best_left_sum_gpu = cp.zeros((n_slots, N_CLASSES), dtype=cp.float32)
    slot_best_right_sum_gpu = cp.zeros((n_slots, N_CLASSES), dtype=cp.float32)

    reduce_blocks = (n_slots + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    reduce_feature_bests[reduce_blocks, THREADS_PER_BLOCK](
        feature_best_gain_gpu,
        feature_best_bin_gpu,
        feature_best_left_count_gpu,
        feature_best_right_count_gpu,
        feature_best_left_sum_gpu,
        feature_best_right_sum_gpu,
        slot_best_gain_gpu,
        slot_best_feature_gpu,
        slot_best_bin_gpu,
        slot_best_left_count_gpu,
        slot_best_right_count_gpu,
        slot_best_left_sum_gpu,
        slot_best_right_sum_gpu,
    )

    cuda.synchronize()

    if tree_growth_profile is not None:
        _update_profile(tree_growth_profile)
    if training_profile is not None:
        _update_profile(training_profile)

    slot_parent_count = cp.asnumpy(slot_parent_count_gpu)
    slot_parent_sum = cp.asnumpy(slot_parent_sum_gpu)
    slot_best_gain = cp.asnumpy(slot_best_gain_gpu)
    slot_best_feature = cp.asnumpy(slot_best_feature_gpu)
    slot_best_bin = cp.asnumpy(slot_best_bin_gpu)
    slot_best_left_count = cp.asnumpy(slot_best_left_count_gpu)
    slot_best_right_count = cp.asnumpy(slot_best_right_count_gpu)
    slot_best_left_sum = cp.asnumpy(slot_best_left_sum_gpu)
    slot_best_right_sum = cp.asnumpy(slot_best_right_sum_gpu)

    split_plans = []
    for slot, node_id in enumerate(candidate_node_ids):
        node = nodes[node_id]
        node.count = int(slot_parent_count[slot])
        parent_sum = slot_parent_sum[slot].astype(np.float64)
        node_weight = float(np.sum(parent_sum)) if USE_WEIGHTED_OBJECTIVE else float(node.count)
        node.value = objective_leaf_value(parent_sum.astype(np.float32), node_weight, REG_LAMBDA)
        parent_score = objective_leaf_score(parent_sum, node_weight, REG_LAMBDA)
        if root_score is None and node_id == 0:
            root_score = parent_score
            root_weight = node_weight

        node.gain = -np.inf
        node.best_left_value = None
        node.best_right_value = None
        node.best_left_count = 0
        node.best_right_count = 0
        node.split_feature = -1
        node.split_bin = -1
        node.split_threshold = 0.0

        if node.count < 2 * MIN_SAMPLES_LEAF:
            node.expandable = False
            continue

        best_feature = int(slot_best_feature[slot])
        best_bin = int(slot_best_bin[slot])
        best_gain = float(slot_best_gain[slot])
        best_left_count = int(slot_best_left_count[slot])
        best_right_count = int(slot_best_right_count[slot])

        if best_feature < 0 or best_bin < 0 or best_gain < MIN_SPLIT_LOSS:
            node.expandable = False
            continue

        best_left_sum = slot_best_left_sum[slot].astype(np.float32)
        best_right_sum = slot_best_right_sum[slot].astype(np.float32)

        node.gain = best_gain
        node.split_feature = best_feature
        node.split_bin = best_bin
        node.split_threshold = float(cuts_cpu[best_feature, best_bin])
        node.best_left_count = best_left_count
        node.best_right_count = best_right_count
        node.best_left_value = objective_leaf_value(
            best_left_sum,
            float(np.sum(best_left_sum)) if USE_WEIGHTED_OBJECTIVE else float(best_left_count),
            REG_LAMBDA,
        )
        node.best_right_value = objective_leaf_value(
            best_right_sum,
            float(np.sum(best_right_sum)) if USE_WEIGHTED_OBJECTIVE else float(best_right_count),
            REG_LAMBDA,
        )

        split_plans.append((node.gain, node_id))

    if not split_plans:
        break

    split_plans.sort(reverse=True)
    selected_node_ids = []
    if GROW_POLICY == "lossguide":
        best_gain, best_node_id = split_plans[0]
        if best_gain >= MIN_SPLIT_LOSS and n_leaves < MAX_LEAVES:
            selected_node_ids.append(best_node_id)
    else:
        budget = MAX_LEAVES - n_leaves
        for gain, node_id in split_plans:
            if budget <= 0:
                break
            if gain < MIN_SPLIT_LOSS:
                continue
            selected_node_ids.append(node_id)
            budget -= 1

    if not selected_node_ids:
        break

    for node_id in selected_node_ids:
        node = nodes[node_id]
        if node.depth >= MAX_DEPTH or n_leaves >= MAX_LEAVES:
            node.expandable = False
            continue

        left_node = Node(
            node_id=next_node_id,
            depth=node.depth + 1,
            value=node.best_left_value,
            count=node.best_left_count,
            expandable=(node.depth + 1) < MAX_DEPTH,
        )
        next_node_id += 1
        right_node = Node(
            node_id=next_node_id,
            depth=node.depth + 1,
            value=node.best_right_value,
            count=node.best_right_count,
            expandable=(node.depth + 1) < MAX_DEPTH,
        )
        next_node_id += 1

        node.is_leaf = False
        node.left_child = left_node.node_id
        node.right_child = right_node.node_id
        node.expandable = False

        nodes.append(left_node)
        nodes.append(right_node)
        n_leaves += 1

    if tree_growth_profile is not None:
        _update_profile(tree_growth_profile)
    if training_profile is not None:
        _update_profile(training_profile)

if tree_growth_profile is not None:
    _finish_profile(tree_growth_profile)
    _print_profile(tree_growth_profile)
if training_profile is not None:
    _finish_profile(training_profile)
    _print_profile(training_profile)


leaf_value_cpu = np.zeros((len(nodes), N_CLASSES), dtype=np.float32)
split_feature_cpu = np.array([node.split_feature for node in nodes], dtype=np.int32)
split_bin_cpu = np.array([node.split_bin for node in nodes], dtype=np.int32)
split_threshold_cpu = np.array([node.split_threshold for node in nodes], dtype=np.float32)
left_child_cpu = np.array([node.left_child for node in nodes], dtype=np.int32)
right_child_cpu = np.array([node.right_child for node in nodes], dtype=np.int32)
is_leaf_cpu = np.array([1 if node.is_leaf else 0 for node in nodes], dtype=np.int8)
for node in nodes:
    if node.value is not None:
        leaf_value_cpu[node.node_id] = node.value

split_feature_gpu = cp.asarray(split_feature_cpu)
split_bin_gpu = cp.asarray(split_bin_cpu)
split_threshold_gpu = cp.asarray(split_threshold_cpu)
left_child_gpu = cp.asarray(left_child_cpu)
right_child_gpu = cp.asarray(right_child_cpu)
is_leaf_gpu = cp.asarray(is_leaf_cpu)
leaf_value_gpu = cp.asarray(leaf_value_cpu)


def predict_one(x_row: np.ndarray) -> np.ndarray:
    node_id = 0
    while not nodes[node_id].is_leaf:
        feature = nodes[node_id].split_feature
        threshold = nodes[node_id].split_threshold
        if x_row[feature] <= threshold:
            node_id = nodes[node_id].left_child
        else:
            node_id = nodes[node_id].right_child
    return nodes[node_id].value


def predict_batch_cpu(x: np.ndarray) -> np.ndarray:
    pred = np.empty((x.shape[0], N_CLASSES), dtype=np.float32)
    pending = [(0, np.arange(x.shape[0], dtype=np.int32))]
    while pending:
        node_id, row_idx = pending.pop()
        if row_idx.size == 0:
            continue

        if is_leaf_cpu[node_id]:
            pred[row_idx] = leaf_value_cpu[node_id]
            continue

        feature = split_feature_cpu[node_id]
        threshold = split_threshold_cpu[node_id]
        left_mask = x[row_idx, feature] <= threshold
        pending.append((right_child_cpu[node_id], row_idx[~left_mask]))
        pending.append((left_child_cpu[node_id], row_idx[left_mask]))
    return pred


def predict_batch_gpu(x: np.ndarray) -> np.ndarray:
    x_gpu = cp.asarray(x, dtype=cp.float32)
    pred_gpu = cp.empty((x_gpu.shape[0], N_CLASSES), dtype=cp.float32)
    blocks_1d = (x_gpu.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    predict_rows_gpu_kernel[blocks_1d, THREADS_PER_BLOCK](
        x_gpu,
        split_feature_gpu,
        split_threshold_gpu,
        left_child_gpu,
        right_child_gpu,
        is_leaf_gpu,
        leaf_value_gpu,
        pred_gpu,
    )
    cuda.synchronize()
    return cp.asnumpy(pred_gpu)


def predict_batch_gpu_bins(bins: np.ndarray) -> np.ndarray:
    bins_gpu = cp.asarray(bins)
    pred_gpu = cp.empty((bins_gpu.shape[0], N_CLASSES), dtype=cp.float32)
    blocks_1d = (bins_gpu.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    predict_rows_gpu_bins_kernel[blocks_1d, THREADS_PER_BLOCK](
        bins_gpu,
        split_feature_gpu,
        split_bin_gpu,
        left_child_gpu,
        right_child_gpu,
        is_leaf_gpu,
        leaf_value_gpu,
        pred_gpu,
    )
    cuda.synchronize()
    return cp.asnumpy(pred_gpu)


def predict_batch(x: np.ndarray) -> np.ndarray:
    if PREDICT_METHOD == "gpu":
        return predict_batch_gpu(x)
    return predict_batch_cpu(x)


evaluation_profile = _start_profile("evaluation") if args.profile else None


# -----------------------------------------------------------------------------
# Final streamed training MSE and class-probability sanity check.
# -----------------------------------------------------------------------------
total_count = 0
total_weight = 0.0
total_weighted_se = 0.0
sum_prob = 0.0
if PREDICT_METHOD == "gpu":
    for batch in quantized_train_batches:
        bins_cpu = batch[0]
        cls_cpu = batch[1]
        pred_cpu = predict_batch_gpu_bins(bins_cpu)
        total_count += bins_cpu.shape[0]
        pred_sq = np.sum(pred_cpu * pred_cpu, axis=1)
        target_prob = pred_cpu[np.arange(pred_cpu.shape[0]), cls_cpu]
        if USE_WEIGHTED_OBJECTIVE:
            sample_weight_cpu = batch[2]
            total_weighted_se += float(np.sum(sample_weight_cpu * (1.0 - 2.0 * target_prob + pred_sq)))
            total_weight += float(np.sum(sample_weight_cpu))
        else:
            total_weighted_se += float(np.sum(1.0 - 2.0 * target_prob + pred_sq))
            total_weight += bins_cpu.shape[0]
        sum_prob += float(np.sum(pred_cpu))

        if evaluation_profile is not None:
            _update_profile(evaluation_profile)
else:
    for x_cpu, y_cpu in GaussianClassStreamProvider(**provider_kwargs):
        cls_cpu = np.argmax(y_cpu, axis=1)
        pred_cpu = predict_batch(x_cpu)
        total_count += x_cpu.shape[0]
        pred_sq = np.sum(pred_cpu * pred_cpu, axis=1)
        target_prob = pred_cpu[np.arange(pred_cpu.shape[0]), cls_cpu]
        if USE_WEIGHTED_OBJECTIVE:
            sample_weight_cpu = CLASS_WEIGHTS_CPU[cls_cpu]
            total_weighted_se += float(np.sum(sample_weight_cpu * (1.0 - 2.0 * target_prob + pred_sq)))
            total_weight += float(np.sum(sample_weight_cpu))
        else:
            total_weighted_se += float(np.sum(1.0 - 2.0 * target_prob + pred_sq))
            total_weight += x_cpu.shape[0]
        sum_prob += float(np.sum(pred_cpu))

        if evaluation_profile is not None:
            _update_profile(evaluation_profile)

train_mse = total_weighted_se / max(total_weight, 1.0)
mean_sum_prob = sum_prob / max(total_count, 1)

if evaluation_profile is not None:
    _finish_profile(evaluation_profile)
    _print_profile(evaluation_profile)

print()
print(f"Built tree with {len(nodes)} nodes and {n_leaves} leaves.")
print(f"Root {OBJECTIVE_NAME} score: {root_score:.6f}")
print(f"Root objective weight: {root_weight:.6f}")
if USE_WEIGHTED_OBJECTIVE:
    print(f"Final streamed train weighted MSE: {train_mse:.6f}")
else:
    print(f"Final streamed train MSE: {train_mse:.6f}")
print(f"Mean sum of class predictions: {mean_sum_prob:.6f}")
print(f"Prediction method: {PREDICT_METHOD}")
print()
print("Tree:")


def print_tree(node_id: int, indent: str = "") -> None:
    node = nodes[node_id]
    if node.is_leaf:
        print(
            indent
            + f"leaf id={node.node_id} depth={node.depth} count={node.count} "
            + f"value={np.array2string(node.value, precision=3, suppress_small=True)}"
        )
        return

    print(
        indent
        + f"node id={node.node_id} depth={node.depth} feature={node.split_feature} "
        + f"threshold={node.split_threshold:.5f} gain={node.gain:.6f}"
    )
    print_tree(node.left_child, indent + "  ")
    print_tree(node.right_child, indent + "  ")


print_tree(0)


make_feature_weighted_hist_plots(
    training_id=PLOT_TRAINING_ID,
    provider_class=GaussianClassStreamProvider,
    provider_kwargs=provider_kwargs,
    predictor=predict_batch,
    n_features=N_FEATURES,
    n_classes=N_CLASSES,
    n_bins=PLOT_BINS,
)
print()
print(f"Saved validation plots under ./plots/{PLOT_TRAINING_ID}/")
