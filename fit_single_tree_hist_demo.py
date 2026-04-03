from __future__ import annotations

import argparse
import ast
import os
import time
from dataclasses import dataclass

import cupy as cp
import numpy as np
from numba import cuda

from plot_feature_ratios import make_feature_weighted_hist_plots
from synthetic_provider import GaussianClassStreamProvider

try:
    import psutil
except Exception:
    psutil = None

try:
    import resource
except Exception:
    resource = None


TREE_CONFIG = {
    "max_bin": 64,
    "cut_sample_rows": 200000,
    "grow_policy": "depthwise",
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
    if TRAINING_CONFIG.get("predict_method") not in {"cpu", "gpu"}:
        raise ValueError("predict_method must be 'cpu' or 'gpu'.")
    return args


ARGS = _parse_args()


@dataclass
class Objective:
    name: str
    class_weights: np.ndarray
    use_weights: bool

    @classmethod
    def from_tree_config(cls, tree_config: dict, n_classes: int) -> "Objective":
        configured = tree_config.get("class_weights")
        if configured is None:
            return cls(name="mse", class_weights=np.ones(n_classes, dtype=np.float32), use_weights=False)

        class_weights = np.asarray(configured, dtype=np.float32)
        if class_weights.shape != (n_classes,):
            raise ValueError("class_weights must have length n_classes.")
        if np.any(class_weights < 0.0):
            raise ValueError("class_weights must be non-negative.")
        return cls(name="weighted_mse", class_weights=class_weights, use_weights=True)

    def leaf_value(self, sum_y: np.ndarray, denominator: float, reg_lambda: float) -> np.ndarray:
        return sum_y / (denominator + reg_lambda)

    def leaf_score(self, sum_y: np.ndarray, denominator: float, reg_lambda: float) -> float:
        if denominator <= 0.0:
            return -np.inf
        return float(np.dot(sum_y, sum_y) / (denominator + reg_lambda))

    def cache_batch(self, bins_cpu: np.ndarray, y_cpu: np.ndarray) -> tuple[np.ndarray, ...]:
        cls_cpu = np.argmax(y_cpu, axis=1).astype(np.int16 if y_cpu.shape[1] > 256 else np.uint8, copy=False)
        if self.use_weights:
            sample_weight_cpu = self.class_weights[cls_cpu].astype(np.float32, copy=False)
            return bins_cpu, cls_cpu.copy(), sample_weight_cpu
        return bins_cpu, cls_cpu.copy()

    def denominator_from_sum(self, sum_y: np.ndarray, count: int) -> float:
        if self.use_weights:
            return float(np.sum(sum_y))
        return float(count)

    def mse_from_predictions(self, pred_cpu: np.ndarray, cls_cpu: np.ndarray, sample_weight_cpu: np.ndarray | None = None) -> tuple[float, float]:
        pred_sq = np.sum(pred_cpu * pred_cpu, axis=1)
        target_prob = pred_cpu[np.arange(pred_cpu.shape[0]), cls_cpu]
        per_row = 1.0 - 2.0 * target_prob + pred_sq
        if sample_weight_cpu is None:
            return float(np.sum(per_row)), float(pred_cpu.shape[0])
        return float(np.sum(sample_weight_cpu * per_row)), float(np.sum(sample_weight_cpu))


OBJECTIVE = Objective.from_tree_config(TREE_CONFIG, DATASET_CONFIG.get("n_classes"))


# -----------------------------------------------------------------------------
# GPU support
# -----------------------------------------------------------------------------

def _bin_np_dtype():
    return np.uint8 if TREE_CONFIG.get("max_bin") <= 256 else np.uint16


def _bin_cp_dtype():
    return cp.uint8 if TREE_CONFIG.get("max_bin") <= 256 else cp.uint16


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
def build_candidate_histograms_weighted(bins, cls, sample_weight, row_slot, hist_count, hist_sum):
    i = cuda.grid(1)
    if i < bins.shape[0]:
        slot = row_slot[i]
        if slot >= 0:
            cls_i = cls[i]
            weight = sample_weight[i]
            for f in range(bins.shape[1]):
                b = bins[i, f]
                cuda.atomic.add(hist_count, (slot, f, b), 1)
                cuda.atomic.add(hist_sum, (slot, f, b, cls_i), weight)


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

        for c in range(hist_sum.shape[3]):
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

            for c in range(hist_sum.shape[3]):
                left_sum[c] += hist_sum[slot, feature, split_bin, c]

            if left_count < min_samples_leaf or right_count < min_samples_leaf:
                continue

            left_score_num = 0.0
            right_score_num = 0.0
            for c in range(hist_sum.shape[3]):
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
                for c in range(hist_sum.shape[3]):
                    best_left_sum[c] = left_sum[c]
                    best_right_sum[c] = parent_sum[c] - left_sum[c]

        feature_best_gain[slot, feature] = best_gain
        feature_best_bin[slot, feature] = best_bin
        feature_best_left_count[slot, feature] = best_left_count
        feature_best_right_count[slot, feature] = best_right_count
        for c in range(hist_sum.shape[3]):
            feature_best_left_sum[slot, feature, c] = best_left_sum[c]
            feature_best_right_sum[slot, feature, c] = best_right_sum[c]


@cuda.jit
def evaluate_feature_splits_weighted(
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

        parent_weight = 0.0
        for c in range(hist_sum.shape[3]):
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
            right_count = parent_count - left_count

            for c in range(hist_sum.shape[3]):
                left_sum[c] += hist_sum[slot, feature, split_bin, c]

            left_weight = 0.0
            for c in range(hist_sum.shape[3]):
                left_weight += left_sum[c]
            right_weight = parent_weight - left_weight

            if left_count < min_samples_leaf or right_count < min_samples_leaf or left_weight <= 0.0 or right_weight <= 0.0:
                continue

            left_score_num = 0.0
            right_score_num = 0.0
            for c in range(hist_sum.shape[3]):
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
                for c in range(hist_sum.shape[3]):
                    best_left_sum[c] = left_sum[c]
                    best_right_sum[c] = parent_sum[c] - left_sum[c]

        feature_best_gain[slot, feature] = best_gain
        feature_best_bin[slot, feature] = best_bin
        feature_best_left_count[slot, feature] = best_left_count
        feature_best_right_count[slot, feature] = best_right_count
        for c in range(hist_sum.shape[3]):
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


# -----------------------------------------------------------------------------
# Profiling helpers
# -----------------------------------------------------------------------------

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
    profile_state["rss_max"] = max(profile_state["rss_max"], rss)
    profile_state["gpu_used_max"] = max(profile_state["gpu_used_max"], gpu["gpu_used_bytes"])
    profile_state["gpu_pool_used_max"] = max(profile_state["gpu_pool_used_max"], gpu["gpu_pool_used_bytes"])
    profile_state["gpu_pool_total_max"] = max(profile_state["gpu_pool_total_max"], gpu["gpu_pool_total_bytes"])


def _finish_profile(profile_state: dict):
    profile_state["wall_end"] = time.perf_counter()
    profile_state["cpu_end"] = time.process_time()
    profile_state["rss_end"] = _rss_bytes()
    gpu = _gpu_snapshot()
    profile_state["gpu_used_end"] = gpu["gpu_used_bytes"]
    profile_state["gpu_pool_used_end"] = gpu["gpu_pool_used_bytes"]
    profile_state["gpu_pool_total_end"] = gpu["gpu_pool_total_bytes"]
    _update_profile(profile_state)


def _print_profile(profile_state: dict):
    def mb(value_bytes: int) -> float:
        return float(value_bytes) / (1024.0 * 1024.0)

    wall = profile_state["wall_end"] - profile_state["wall_start"]
    cpu = profile_state["cpu_end"] - profile_state["cpu_start"]
    cpu_pct = 100.0 * cpu / wall if wall > 0.0 else 0.0

    print()
    print(f"[profile:{profile_state['stage']}]")
    print(f"  wall_s={wall:.3f} cpu_s={cpu:.3f} cpu_pct_est={cpu_pct:.1f}")
    print(
        "  rss_mb="
        f"start={mb(profile_state['rss_start']):.1f} "
        f"end={mb(profile_state['rss_end']):.1f} "
        f"max={mb(profile_state['rss_max']):.1f}"
    )
    print(
        "  gpu_used_mb="
        f"start={mb(profile_state['gpu_used_start']):.1f} "
        f"end={mb(profile_state['gpu_used_end']):.1f} "
        f"max={mb(profile_state['gpu_used_max']):.1f}"
    )
    print(
        "  gpu_pool_used_mb="
        f"start={mb(profile_state['gpu_pool_used_start']):.1f} "
        f"end={mb(profile_state['gpu_pool_used_end']):.1f} "
        f"max={mb(profile_state['gpu_pool_used_max']):.1f}"
    )
    print(
        "  gpu_pool_reserved_mb="
        f"start={mb(profile_state['gpu_pool_total_start']):.1f} "
        f"end={mb(profile_state['gpu_pool_total_end']):.1f} "
        f"max={mb(profile_state['gpu_pool_total_max']):.1f}"
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


class SingleTree:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.nodes: list[Node] = [Node(node_id=0, depth=0)]
        self.n_leaves = 1
        self.next_node_id = 1
        self.root_score: float | None = None
        self.root_weight: float | None = None
        self.leaf_value_cpu = None
        self.split_feature_cpu = None
        self.split_bin_cpu = None
        self.split_threshold_cpu = None
        self.left_child_cpu = None
        self.right_child_cpu = None
        self.is_leaf_cpu = None
        self.split_feature_gpu = None
        self.split_bin_gpu = None
        self.split_threshold_gpu = None
        self.left_child_gpu = None
        self.right_child_gpu = None
        self.is_leaf_gpu = None
        self.leaf_value_gpu = None

    def candidate_node_ids(self) -> list[int]:
        node_ids = [
            node.node_id
            for node in self.nodes
            if node.is_leaf and node.expandable and node.depth < TREE_CONFIG.get("max_depth")
        ]
        if TREE_CONFIG.get("grow_policy") == "depthwise" and node_ids:
            frontier_depth = min(self.nodes[node_id].depth for node_id in node_ids)
            node_ids = [node_id for node_id in node_ids if self.nodes[node_id].depth == frontier_depth]
        return node_ids

    def finalize_prediction_state(self):
        self.leaf_value_cpu = np.zeros((len(self.nodes), self.n_classes), dtype=np.float32)
        self.split_feature_cpu = np.array([node.split_feature for node in self.nodes], dtype=np.int32)
        self.split_bin_cpu = np.array([node.split_bin for node in self.nodes], dtype=np.int32)
        self.split_threshold_cpu = np.array([node.split_threshold for node in self.nodes], dtype=np.float32)
        self.left_child_cpu = np.array([node.left_child for node in self.nodes], dtype=np.int32)
        self.right_child_cpu = np.array([node.right_child for node in self.nodes], dtype=np.int32)
        self.is_leaf_cpu = np.array([1 if node.is_leaf else 0 for node in self.nodes], dtype=np.int8)

        for node in self.nodes:
            if node.value is not None:
                self.leaf_value_cpu[node.node_id] = node.value

        self.split_feature_gpu = cp.asarray(self.split_feature_cpu)
        self.split_bin_gpu = cp.asarray(self.split_bin_cpu)
        self.split_threshold_gpu = cp.asarray(self.split_threshold_cpu)
        self.left_child_gpu = cp.asarray(self.left_child_cpu)
        self.right_child_gpu = cp.asarray(self.right_child_cpu)
        self.is_leaf_gpu = cp.asarray(self.is_leaf_cpu)
        self.leaf_value_gpu = cp.asarray(self.leaf_value_cpu)

    def predict_batch_cpu(self, x: np.ndarray) -> np.ndarray:
        pred = np.empty((x.shape[0], self.n_classes), dtype=np.float32)
        pending = [(0, np.arange(x.shape[0], dtype=np.int32))]
        while pending:
            node_id, row_idx = pending.pop()
            if row_idx.size == 0:
                continue
            if self.is_leaf_cpu[node_id]:
                pred[row_idx] = self.leaf_value_cpu[node_id]
                continue
            feature = self.split_feature_cpu[node_id]
            threshold = self.split_threshold_cpu[node_id]
            left_mask = x[row_idx, feature] <= threshold
            pending.append((self.right_child_cpu[node_id], row_idx[~left_mask]))
            pending.append((self.left_child_cpu[node_id], row_idx[left_mask]))
        return pred

    def predict_batch_gpu(self, x: np.ndarray) -> np.ndarray:
        x_gpu = cp.asarray(x, dtype=cp.float32)
        pred_gpu = cp.empty((x_gpu.shape[0], self.n_classes), dtype=cp.float32)
        blocks_1d = (x_gpu.shape[0] + TRAINING_CONFIG.get("threads_per_block") - 1) // TRAINING_CONFIG.get("threads_per_block")
        predict_rows_gpu_kernel[blocks_1d, TRAINING_CONFIG.get("threads_per_block")](
            x_gpu,
            self.split_feature_gpu,
            self.split_threshold_gpu,
            self.left_child_gpu,
            self.right_child_gpu,
            self.is_leaf_gpu,
            self.leaf_value_gpu,
            pred_gpu,
        )
        cuda.synchronize()
        return cp.asnumpy(pred_gpu)

    def predict_batch_gpu_bins(self, bins: np.ndarray) -> np.ndarray:
        bins_gpu = cp.asarray(bins)
        pred_gpu = cp.empty((bins_gpu.shape[0], self.n_classes), dtype=cp.float32)
        blocks_1d = (bins_gpu.shape[0] + TRAINING_CONFIG.get("threads_per_block") - 1) // TRAINING_CONFIG.get("threads_per_block")
        predict_rows_gpu_bins_kernel[blocks_1d, TRAINING_CONFIG.get("threads_per_block")](
            bins_gpu,
            self.split_feature_gpu,
            self.split_bin_gpu,
            self.left_child_gpu,
            self.right_child_gpu,
            self.is_leaf_gpu,
            self.leaf_value_gpu,
            pred_gpu,
        )
        cuda.synchronize()
        return cp.asnumpy(pred_gpu)

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        if TRAINING_CONFIG.get("predict_method") == "gpu":
            return self.predict_batch_gpu(x)
        return self.predict_batch_cpu(x)

    def print_tree(self, node_id: int = 0, indent: str = "") -> None:
        node = self.nodes[node_id]
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
        self.print_tree(node.left_child, indent + "  ")
        self.print_tree(node.right_child, indent + "  ")


def _provider_kwargs():
    return {
        "n_features": DATASET_CONFIG.get("n_features"),
        "n_classes": DATASET_CONFIG.get("n_classes"),
        "batch_size": DATASET_CONFIG.get("batch_size"),
        "n_batches": DATASET_CONFIG.get("n_batches"),
        "feature_offset_scale": DATASET_CONFIG.get("feature_offset_scale"),
        "feature_noise": DATASET_CONFIG.get("feature_noise"),
        "seed": DATASET_CONFIG.get("seed"),
    }


def _build_cuts(provider_kwargs: dict) -> tuple[np.ndarray, cp.ndarray]:
    cut_batches = []
    sampled_rows = 0
    for x_cpu, _ in GaussianClassStreamProvider(**provider_kwargs):
        take = min(x_cpu.shape[0], TREE_CONFIG.get("cut_sample_rows") - sampled_rows)
        if take > 0:
            cut_batches.append(x_cpu[:take].copy())
            sampled_rows += take
        if sampled_rows >= TREE_CONFIG.get("cut_sample_rows"):
            break

    cut_sample = np.concatenate(cut_batches, axis=0)
    quantile_levels = np.linspace(0.0, 1.0, TREE_CONFIG.get("max_bin") + 1, dtype=np.float64)[1:-1]
    cuts_cpu = np.quantile(cut_sample, quantile_levels, axis=0).T.astype(np.float32)
    return cuts_cpu, cp.asarray(cuts_cpu)


def _build_quantized_cache(
    provider_kwargs: dict,
    cuts_cpu: np.ndarray,
    cuts_gpu: cp.ndarray,
    training_profile: dict | None,
) -> list[tuple[np.ndarray, ...]]:
    cache_profile = _start_profile("cache_build") if ARGS.profile else None
    cache_x_gpu = None
    cache_bins_gpu = None
    quantized_train_batches = []

    for x_cpu, y_cpu in GaussianClassStreamProvider(**provider_kwargs):
        if cache_x_gpu is None or cache_x_gpu.shape != x_cpu.shape:
            cache_x_gpu = cp.empty(x_cpu.shape, dtype=cp.float32)
            cache_bins_gpu = cp.empty(x_cpu.shape, dtype=_bin_cp_dtype())

        cache_x_gpu.set(x_cpu)
        quant_blocks = (
            (cache_x_gpu.shape[0] + 15) // 16,
            (cache_x_gpu.shape[1] + 15) // 16,
        )
        quantize_batch[quant_blocks, (16, 16)](cache_x_gpu, cuts_gpu, cache_bins_gpu)
        cuda.synchronize()
        bins_cpu = cp.asnumpy(cache_bins_gpu)
        quantized_train_batches.append(OBJECTIVE.cache_batch(bins_cpu, y_cpu))
        if cache_profile is not None:
            _update_profile(cache_profile)
        if training_profile is not None:
            _update_profile(training_profile)

    if cache_profile is not None:
        _finish_profile(cache_profile)
        _print_profile(cache_profile)
    return quantized_train_batches


def _train_tree(
    tree: SingleTree,
    quantized_train_batches: list[tuple[np.ndarray, ...]],
    cuts_cpu: np.ndarray,
    training_profile: dict | None,
):
    tree_growth_profile = _start_profile("tree_growth") if ARGS.profile else None

    while True:
        candidate_node_ids = tree.candidate_node_ids()
        if not candidate_node_ids:
            break

        candidate_slot_of_node_cpu = np.full(len(tree.nodes), -1, dtype=np.int32)
        for slot, node_id in enumerate(candidate_node_ids):
            candidate_slot_of_node_cpu[node_id] = slot

        split_feature_gpu = cp.asarray(np.array([node.split_feature for node in tree.nodes], dtype=np.int32))
        split_bin_gpu = cp.asarray(np.array([node.split_bin for node in tree.nodes], dtype=np.int32))
        left_child_gpu = cp.asarray(np.array([node.left_child for node in tree.nodes], dtype=np.int32))
        right_child_gpu = cp.asarray(np.array([node.right_child for node in tree.nodes], dtype=np.int32))
        is_leaf_gpu = cp.asarray(np.array([1 if node.is_leaf else 0 for node in tree.nodes], dtype=np.int8))
        candidate_slot_of_node_gpu = cp.asarray(candidate_slot_of_node_cpu)

        hist_count_gpu = cp.zeros(
            (len(candidate_node_ids), DATASET_CONFIG.get("n_features"), TREE_CONFIG.get("max_bin")),
            dtype=cp.int32,
        )
        hist_sum_gpu = cp.zeros(
            (len(candidate_node_ids), DATASET_CONFIG.get("n_features"), TREE_CONFIG.get("max_bin"), DATASET_CONFIG.get("n_classes")),
            dtype=cp.float32,
        )

        for batch in quantized_train_batches:
            bins_gpu = cp.asarray(batch[0])
            cls_gpu = cp.asarray(batch[1])
            blocks_1d = (bins_gpu.shape[0] + TRAINING_CONFIG.get("threads_per_block") - 1) // TRAINING_CONFIG.get("threads_per_block")
            row_slot_gpu = cp.empty((bins_gpu.shape[0],), dtype=cp.int32)
            route_rows_to_candidate_slots[blocks_1d, TRAINING_CONFIG.get("threads_per_block")](
                bins_gpu,
                split_feature_gpu,
                split_bin_gpu,
                left_child_gpu,
                right_child_gpu,
                is_leaf_gpu,
                candidate_slot_of_node_gpu,
                row_slot_gpu,
            )
            if OBJECTIVE.use_weights:
                build_candidate_histograms_weighted[blocks_1d, TRAINING_CONFIG.get("threads_per_block")](
                    bins_gpu,
                    cls_gpu,
                    cp.asarray(batch[2]),
                    row_slot_gpu,
                    hist_count_gpu,
                    hist_sum_gpu,
                )
            else:
                build_candidate_histograms_unweighted[blocks_1d, TRAINING_CONFIG.get("threads_per_block")](
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
        slot_parent_sum_gpu = cp.zeros((n_slots, DATASET_CONFIG.get("n_classes")), dtype=cp.float32)
        feature_best_gain_gpu = cp.full((n_slots, DATASET_CONFIG.get("n_features")), -1.0e30, dtype=cp.float32)
        feature_best_bin_gpu = cp.full((n_slots, DATASET_CONFIG.get("n_features")), -1, dtype=cp.int32)
        feature_best_left_count_gpu = cp.zeros((n_slots, DATASET_CONFIG.get("n_features")), dtype=cp.int32)
        feature_best_right_count_gpu = cp.zeros((n_slots, DATASET_CONFIG.get("n_features")), dtype=cp.int32)
        feature_best_left_sum_gpu = cp.zeros((n_slots, DATASET_CONFIG.get("n_features"), DATASET_CONFIG.get("n_classes")), dtype=cp.float32)
        feature_best_right_sum_gpu = cp.zeros((n_slots, DATASET_CONFIG.get("n_features"), DATASET_CONFIG.get("n_classes")), dtype=cp.float32)

        eval_blocks = ((n_slots + 7) // 8, (DATASET_CONFIG.get("n_features") + 7) // 8)
        if OBJECTIVE.use_weights:
            evaluate_feature_splits_weighted[eval_blocks, (8, 8)](
                hist_count_gpu,
                hist_sum_gpu,
                TREE_CONFIG.get("min_samples_leaf"),
                TREE_CONFIG.get("reg_lambda"),
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
                TREE_CONFIG.get("min_samples_leaf"),
                TREE_CONFIG.get("reg_lambda"),
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
        slot_best_left_sum_gpu = cp.zeros((n_slots, DATASET_CONFIG.get("n_classes")), dtype=cp.float32)
        slot_best_right_sum_gpu = cp.zeros((n_slots, DATASET_CONFIG.get("n_classes")), dtype=cp.float32)

        reduce_blocks = (n_slots + TRAINING_CONFIG.get("threads_per_block") - 1) // TRAINING_CONFIG.get("threads_per_block")
        reduce_feature_bests[reduce_blocks, TRAINING_CONFIG.get("threads_per_block")](
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
            node = tree.nodes[node_id]
            node.count = int(slot_parent_count[slot])
            parent_sum = slot_parent_sum[slot].astype(np.float64)
            node_denominator = OBJECTIVE.denominator_from_sum(parent_sum, node.count)
            node.value = OBJECTIVE.leaf_value(parent_sum.astype(np.float32), node_denominator, TREE_CONFIG.get("reg_lambda"))
            parent_score = OBJECTIVE.leaf_score(parent_sum, node_denominator, TREE_CONFIG.get("reg_lambda"))
            if tree.root_score is None and node_id == 0:
                tree.root_score = parent_score
                tree.root_weight = node_denominator

            node.gain = -np.inf
            node.best_left_value = None
            node.best_right_value = None
            node.best_left_count = 0
            node.best_right_count = 0
            node.split_feature = -1
            node.split_bin = -1
            node.split_threshold = 0.0

            if node.count < 2 * TREE_CONFIG.get("min_samples_leaf"):
                node.expandable = False
                continue

            best_feature = int(slot_best_feature[slot])
            best_bin = int(slot_best_bin[slot])
            best_gain = float(slot_best_gain[slot])
            best_left_count = int(slot_best_left_count[slot])
            best_right_count = int(slot_best_right_count[slot])
            if best_feature < 0 or best_bin < 0 or best_gain < TREE_CONFIG.get("min_split_loss"):
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
            node.best_left_value = OBJECTIVE.leaf_value(
                best_left_sum,
                OBJECTIVE.denominator_from_sum(best_left_sum.astype(np.float64), best_left_count),
                TREE_CONFIG.get("reg_lambda"),
            )
            node.best_right_value = OBJECTIVE.leaf_value(
                best_right_sum,
                OBJECTIVE.denominator_from_sum(best_right_sum.astype(np.float64), best_right_count),
                TREE_CONFIG.get("reg_lambda"),
            )
            split_plans.append((node.gain, node_id))

        if not split_plans:
            break

        split_plans.sort(reverse=True)
        selected_node_ids = []
        if TREE_CONFIG.get("grow_policy") == "lossguide":
            best_gain, best_node_id = split_plans[0]
            if best_gain >= TREE_CONFIG.get("min_split_loss") and tree.n_leaves < TREE_CONFIG.get("max_leaves"):
                selected_node_ids.append(best_node_id)
        else:
            budget = TREE_CONFIG.get("max_leaves") - tree.n_leaves
            for gain, node_id in split_plans:
                if budget <= 0:
                    break
                if gain < TREE_CONFIG.get("min_split_loss"):
                    continue
                selected_node_ids.append(node_id)
                budget -= 1

        if not selected_node_ids:
            break

        for node_id in selected_node_ids:
            node = tree.nodes[node_id]
            if node.depth >= TREE_CONFIG.get("max_depth") or tree.n_leaves >= TREE_CONFIG.get("max_leaves"):
                node.expandable = False
                continue

            left_node = Node(
                node_id=tree.next_node_id,
                depth=node.depth + 1,
                value=node.best_left_value,
                count=node.best_left_count,
                expandable=(node.depth + 1) < TREE_CONFIG.get("max_depth"),
            )
            tree.next_node_id += 1
            right_node = Node(
                node_id=tree.next_node_id,
                depth=node.depth + 1,
                value=node.best_right_value,
                count=node.best_right_count,
                expandable=(node.depth + 1) < TREE_CONFIG.get("max_depth"),
            )
            tree.next_node_id += 1

            node.is_leaf = False
            node.left_child = left_node.node_id
            node.right_child = right_node.node_id
            node.expandable = False
            tree.nodes.append(left_node)
            tree.nodes.append(right_node)
            tree.n_leaves += 1

        if tree_growth_profile is not None:
            _update_profile(tree_growth_profile)
        if training_profile is not None:
            _update_profile(training_profile)

    if tree_growth_profile is not None:
        _finish_profile(tree_growth_profile)
        _print_profile(tree_growth_profile)
def _evaluate_cached_training_stream(tree: SingleTree, quantized_train_batches: list[tuple[np.ndarray, ...]]) -> tuple[float, float]:
    evaluation_profile = _start_profile("evaluation") if ARGS.profile else None
    total_count = 0
    total_denominator = 0.0
    total_error = 0.0
    sum_prob = 0.0

    if TRAINING_CONFIG.get("predict_method") == "gpu":
        for batch in quantized_train_batches:
            pred_cpu = tree.predict_batch_gpu_bins(batch[0])
            error_sum, denominator = OBJECTIVE.mse_from_predictions(
                pred_cpu,
                batch[1],
                batch[2] if OBJECTIVE.use_weights else None,
            )
            total_count += batch[0].shape[0]
            total_error += error_sum
            total_denominator += denominator
            sum_prob += float(np.sum(pred_cpu))
            if evaluation_profile is not None:
                _update_profile(evaluation_profile)
    else:
        for x_cpu, y_cpu in GaussianClassStreamProvider(**_provider_kwargs()):
            cls_cpu = np.argmax(y_cpu, axis=1)
            pred_cpu = tree.predict_batch(x_cpu)
            error_sum, denominator = OBJECTIVE.mse_from_predictions(
                pred_cpu,
                cls_cpu,
                OBJECTIVE.class_weights[cls_cpu] if OBJECTIVE.use_weights else None,
            )
            total_count += x_cpu.shape[0]
            total_error += error_sum
            total_denominator += denominator
            sum_prob += float(np.sum(pred_cpu))
            if evaluation_profile is not None:
                _update_profile(evaluation_profile)

    if evaluation_profile is not None:
        _finish_profile(evaluation_profile)
        _print_profile(evaluation_profile)
        print()
        print("Inference done.")

    return total_error / max(total_denominator, 1.0), sum_prob / max(total_count, 1)


def _profile_fresh_inference(tree: SingleTree, provider_kwargs: dict):
    if not ARGS.profile or TRAINING_CONFIG.get("predict_method") != "gpu":
        return
    fresh_profile = _start_profile("fresh_inference")
    fresh_sum_prob = 0.0
    fresh_count = 0
    for x_cpu, _ in GaussianClassStreamProvider(**provider_kwargs):
        pred_cpu = tree.predict_batch_gpu(x_cpu)
        fresh_sum_prob += float(np.sum(pred_cpu))
        fresh_count += x_cpu.shape[0]
        _update_profile(fresh_profile)
    _finish_profile(fresh_profile)
    _print_profile(fresh_profile)
    print(f"Fresh inference mean sum of class predictions: {fresh_sum_prob / max(fresh_count, 1):.6f}")


# -----------------------------------------------------------------------------
# Plotting section. This will move out next.
# -----------------------------------------------------------------------------

def _emit_plot_artifacts(tree: SingleTree, provider_kwargs: dict):
    if not (ARGS.full_output or not ARGS.profile):
        return
    print()
    print("Tree:")
    tree.print_tree()
    make_feature_weighted_hist_plots(
        training_id=TRAINING_CONFIG.get("plot_training_id"),
        provider_class=GaussianClassStreamProvider,
        provider_kwargs=provider_kwargs,
        predictor=tree.predict_batch,
        n_features=DATASET_CONFIG.get("n_features"),
        n_classes=DATASET_CONFIG.get("n_classes"),
        n_bins=TRAINING_CONFIG.get("plot_bins"),
    )
    print()
    print(f"Saved validation plots under ./plots/{TRAINING_CONFIG.get('plot_training_id')}/")


def main():
    provider_kwargs = _provider_kwargs()
    print("GPU:", cuda.get_current_device().name)
    print("Tree config:", TREE_CONFIG)
    print("Dataset config:", DATASET_CONFIG)
    print("Training config:", TRAINING_CONFIG)
    print("Objective:", OBJECTIVE.name)
    print("Class weights:", OBJECTIVE.class_weights.tolist())
    print(
        f"Building a single tree with grow_policy={TREE_CONFIG.get('grow_policy')}, "
        f"max_depth={TREE_CONFIG.get('max_depth')}, max_leaves={TREE_CONFIG.get('max_leaves')}, "
        f"max_bin={TREE_CONFIG.get('max_bin')}"
    )

    cuts_cpu, cuts_gpu = _build_cuts(provider_kwargs)
    training_profile = _start_profile("training") if ARGS.profile else None
    quantized_train_batches = _build_quantized_cache(provider_kwargs, cuts_cpu, cuts_gpu, training_profile)

    tree = SingleTree(DATASET_CONFIG.get("n_classes"))
    _train_tree(tree, quantized_train_batches, cuts_cpu, training_profile)
    if training_profile is not None:
        _finish_profile(training_profile)
        _print_profile(training_profile)
    tree.finalize_prediction_state()

    train_mse, mean_sum_prob = _evaluate_cached_training_stream(tree, quantized_train_batches)
    _profile_fresh_inference(tree, provider_kwargs)

    print()
    print(f"Built tree with {len(tree.nodes)} nodes and {tree.n_leaves} leaves.")
    print(f"Root {OBJECTIVE.name} score: {tree.root_score:.6f}")
    print(f"Root objective weight: {tree.root_weight:.6f}")
    if OBJECTIVE.use_weights:
        print(f"Final streamed train weighted MSE: {train_mse:.6f}")
    else:
        print(f"Final streamed train MSE: {train_mse:.6f}")
    print(f"Mean sum of class predictions: {mean_sum_prob:.6f}")
    print(f"Prediction method: {TRAINING_CONFIG.get('predict_method')}")

    _emit_plot_artifacts(tree, provider_kwargs)


if __name__ == "__main__":
    main()
