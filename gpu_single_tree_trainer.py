from __future__ import annotations

import os
import time

import cupy as cp
import numpy as np
from numba import cuda

from single_tree import Node, Objective, SingleTree
from synthetic_provider import GaussianClassStreamProvider

try:
    import psutil
except Exception:
    psutil = None

try:
    import resource
except Exception:
    resource = None


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


def print_profile(profile_state: dict):
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


class GpuSingleTreeTrainer:
    def __init__(self, tree_config: dict, dataset_config: dict, training_config: dict, objective: Objective):
        self.tree_config = tree_config
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.objective = objective

    @property
    def device_name(self) -> str:
        return str(cuda.get_current_device().name)

    def provider_kwargs(self) -> dict:
        return {
            "n_features": self.dataset_config.get("n_features"),
            "n_classes": self.dataset_config.get("n_classes"),
            "batch_size": self.dataset_config.get("batch_size"),
            "n_batches": self.dataset_config.get("n_batches"),
            "feature_offset_scale": self.dataset_config.get("feature_offset_scale"),
            "feature_noise": self.dataset_config.get("feature_noise"),
            "seed": self.dataset_config.get("seed"),
        }

    def _bin_cp_dtype(self):
        return cp.uint8 if self.tree_config.get("max_bin") <= 256 else cp.uint16

    def build_cuts(self, provider_kwargs: dict) -> tuple[np.ndarray, cp.ndarray]:
        cut_batches = []
        sampled_rows = 0
        for x_cpu, _ in GaussianClassStreamProvider(**provider_kwargs):
            take = min(x_cpu.shape[0], self.tree_config.get("cut_sample_rows") - sampled_rows)
            if take > 0:
                cut_batches.append(x_cpu[:take].copy())
                sampled_rows += take
            if sampled_rows >= self.tree_config.get("cut_sample_rows"):
                break

        cut_sample = np.concatenate(cut_batches, axis=0)
        quantile_levels = np.linspace(0.0, 1.0, self.tree_config.get("max_bin") + 1, dtype=np.float64)[1:-1]
        cuts_cpu = np.quantile(cut_sample, quantile_levels, axis=0).T.astype(np.float32)
        return cuts_cpu, cp.asarray(cuts_cpu)

    def build_quantized_cache(
        self,
        provider_kwargs: dict,
        cuts_gpu: cp.ndarray,
        profile: bool,
        training_profile: dict | None,
    ) -> list[tuple[np.ndarray, ...]]:
        cache_profile = _start_profile("cache_build") if profile else None
        cache_x_gpu = None
        cache_bins_gpu = None
        quantized_train_batches = []

        for x_cpu, y_cpu in GaussianClassStreamProvider(**provider_kwargs):
            if cache_x_gpu is None or cache_x_gpu.shape != x_cpu.shape:
                cache_x_gpu = cp.empty(x_cpu.shape, dtype=cp.float32)
                cache_bins_gpu = cp.empty(x_cpu.shape, dtype=self._bin_cp_dtype())

            cache_x_gpu.set(x_cpu)
            quant_blocks = ((cache_x_gpu.shape[0] + 15) // 16, (cache_x_gpu.shape[1] + 15) // 16)
            quantize_batch[quant_blocks, (16, 16)](cache_x_gpu, cuts_gpu, cache_bins_gpu)
            cuda.synchronize()
            quantized_train_batches.append(self.objective.cache_batch(cp.asnumpy(cache_bins_gpu), y_cpu))
            if cache_profile is not None:
                _update_profile(cache_profile)
            if training_profile is not None:
                _update_profile(training_profile)

        if cache_profile is not None:
            _finish_profile(cache_profile)
            print_profile(cache_profile)
        return quantized_train_batches

    def _finalize_gpu_prediction_state(self, tree: SingleTree):
        tree.split_feature_gpu = cp.asarray(tree.split_feature_cpu)
        tree.split_bin_gpu = cp.asarray(tree.split_bin_cpu)
        tree.split_threshold_gpu = cp.asarray(tree.split_threshold_cpu)
        tree.left_child_gpu = cp.asarray(tree.left_child_cpu)
        tree.right_child_gpu = cp.asarray(tree.right_child_cpu)
        tree.is_leaf_gpu = cp.asarray(tree.is_leaf_cpu)
        tree.leaf_value_gpu = cp.asarray(tree.leaf_value_cpu)

    def predict_batch(self, tree: SingleTree, x: np.ndarray | None = None, bins: np.ndarray | None = None) -> np.ndarray:
        if (x is None) == (bins is None):
            raise ValueError("Provide exactly one of x or bins.")

        pred_gpu = cp.empty(((x.shape[0] if x is not None else bins.shape[0]), tree.n_classes), dtype=cp.float32)
        threads_per_block = self.training_config.get("threads_per_block")
        blocks_1d = (pred_gpu.shape[0] + threads_per_block - 1) // threads_per_block
        if bins is not None:
            bins_gpu = cp.asarray(bins)
            predict_rows_gpu_bins_kernel[blocks_1d, threads_per_block](
                bins_gpu,
                tree.split_feature_gpu,
                tree.split_bin_gpu,
                tree.left_child_gpu,
                tree.right_child_gpu,
                tree.is_leaf_gpu,
                tree.leaf_value_gpu,
                pred_gpu,
            )
        else:
            x_gpu = cp.asarray(x, dtype=cp.float32)
            predict_rows_gpu_kernel[blocks_1d, threads_per_block](
                x_gpu,
                tree.split_feature_gpu,
                tree.split_threshold_gpu,
                tree.left_child_gpu,
                tree.right_child_gpu,
                tree.is_leaf_gpu,
                tree.leaf_value_gpu,
                pred_gpu,
            )
        cuda.synchronize()
        return cp.asnumpy(pred_gpu)

    def fit(
        self,
        tree: SingleTree,
        quantized_train_batches: list[tuple[np.ndarray, ...]],
        cuts_cpu: np.ndarray,
        profile: bool,
        training_profile: dict | None,
    ):
        tree_growth_profile = _start_profile("tree_growth") if profile else None

        while True:
            candidate_node_ids = tree.candidate_node_ids(self.tree_config)
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
                (len(candidate_node_ids), self.dataset_config.get("n_features"), self.tree_config.get("max_bin")),
                dtype=cp.int32,
            )
            hist_sum_gpu = cp.zeros(
                (
                    len(candidate_node_ids),
                    self.dataset_config.get("n_features"),
                    self.tree_config.get("max_bin"),
                    self.dataset_config.get("n_classes"),
                ),
                dtype=cp.float32,
            )

            threads_per_block = self.training_config.get("threads_per_block")
            for batch in quantized_train_batches:
                bins_gpu = cp.asarray(batch[0])
                cls_gpu = cp.asarray(batch[1])
                blocks_1d = (bins_gpu.shape[0] + threads_per_block - 1) // threads_per_block
                row_slot_gpu = cp.empty((bins_gpu.shape[0],), dtype=cp.int32)
                route_rows_to_candidate_slots[blocks_1d, threads_per_block](
                    bins_gpu,
                    split_feature_gpu,
                    split_bin_gpu,
                    left_child_gpu,
                    right_child_gpu,
                    is_leaf_gpu,
                    candidate_slot_of_node_gpu,
                    row_slot_gpu,
                )
                if self.objective.use_weights:
                    build_candidate_histograms_weighted[blocks_1d, threads_per_block](
                        bins_gpu,
                        cls_gpu,
                        cp.asarray(batch[2]),
                        row_slot_gpu,
                        hist_count_gpu,
                        hist_sum_gpu,
                    )
                else:
                    build_candidate_histograms_unweighted[blocks_1d, threads_per_block](
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
            slot_parent_sum_gpu = cp.zeros((n_slots, self.dataset_config.get("n_classes")), dtype=cp.float32)
            feature_best_gain_gpu = cp.full((n_slots, self.dataset_config.get("n_features")), -1.0e30, dtype=cp.float32)
            feature_best_bin_gpu = cp.full((n_slots, self.dataset_config.get("n_features")), -1, dtype=cp.int32)
            feature_best_left_count_gpu = cp.zeros((n_slots, self.dataset_config.get("n_features")), dtype=cp.int32)
            feature_best_right_count_gpu = cp.zeros((n_slots, self.dataset_config.get("n_features")), dtype=cp.int32)
            feature_best_left_sum_gpu = cp.zeros(
                (n_slots, self.dataset_config.get("n_features"), self.dataset_config.get("n_classes")),
                dtype=cp.float32,
            )
            feature_best_right_sum_gpu = cp.zeros(
                (n_slots, self.dataset_config.get("n_features"), self.dataset_config.get("n_classes")),
                dtype=cp.float32,
            )

            eval_blocks = ((n_slots + 7) // 8, (self.dataset_config.get("n_features") + 7) // 8)
            if self.objective.use_weights:
                evaluate_feature_splits_weighted[eval_blocks, (8, 8)](
                    hist_count_gpu,
                    hist_sum_gpu,
                    self.tree_config.get("min_samples_leaf"),
                    self.tree_config.get("reg_lambda"),
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
                    self.tree_config.get("min_samples_leaf"),
                    self.tree_config.get("reg_lambda"),
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
            slot_best_left_sum_gpu = cp.zeros((n_slots, self.dataset_config.get("n_classes")), dtype=cp.float32)
            slot_best_right_sum_gpu = cp.zeros((n_slots, self.dataset_config.get("n_classes")), dtype=cp.float32)

            reduce_blocks = (n_slots + threads_per_block - 1) // threads_per_block
            reduce_feature_bests[reduce_blocks, threads_per_block](
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
                node_denominator = self.objective.denominator_from_sum(parent_sum, node.count)
                node.value = self.objective.leaf_value(parent_sum.astype(np.float32), node_denominator, self.tree_config.get("reg_lambda"))
                parent_score = self.objective.leaf_score(parent_sum, node_denominator, self.tree_config.get("reg_lambda"))
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

                if node.count < 2 * self.tree_config.get("min_samples_leaf"):
                    node.expandable = False
                    continue

                best_feature = int(slot_best_feature[slot])
                best_bin = int(slot_best_bin[slot])
                best_gain = float(slot_best_gain[slot])
                best_left_count = int(slot_best_left_count[slot])
                best_right_count = int(slot_best_right_count[slot])
                if best_feature < 0 or best_bin < 0 or best_gain < self.tree_config.get("min_split_loss"):
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
                node.best_left_value = self.objective.leaf_value(
                    best_left_sum,
                    self.objective.denominator_from_sum(best_left_sum.astype(np.float64), best_left_count),
                    self.tree_config.get("reg_lambda"),
                )
                node.best_right_value = self.objective.leaf_value(
                    best_right_sum,
                    self.objective.denominator_from_sum(best_right_sum.astype(np.float64), best_right_count),
                    self.tree_config.get("reg_lambda"),
                )
                split_plans.append((node.gain, node_id))

            if not split_plans:
                break

            split_plans.sort(reverse=True)
            selected_node_ids = []
            if self.tree_config.get("grow_policy") == "lossguide":
                best_gain, best_node_id = split_plans[0]
                if best_gain >= self.tree_config.get("min_split_loss") and tree.n_leaves < self.tree_config.get("max_leaves"):
                    selected_node_ids.append(best_node_id)
            else:
                budget = self.tree_config.get("max_leaves") - tree.n_leaves
                for gain, node_id in split_plans:
                    if budget <= 0:
                        break
                    if gain < self.tree_config.get("min_split_loss"):
                        continue
                    selected_node_ids.append(node_id)
                    budget -= 1

            if not selected_node_ids:
                break

            for node_id in selected_node_ids:
                node = tree.nodes[node_id]
                if node.depth >= self.tree_config.get("max_depth") or tree.n_leaves >= self.tree_config.get("max_leaves"):
                    node.expandable = False
                    continue

                left_node = Node(
                    node_id=tree.next_node_id,
                    depth=node.depth + 1,
                    value=node.best_left_value,
                    count=node.best_left_count,
                    expandable=(node.depth + 1) < self.tree_config.get("max_depth"),
                )
                tree.next_node_id += 1
                right_node = Node(
                    node_id=tree.next_node_id,
                    depth=node.depth + 1,
                    value=node.best_right_value,
                    count=node.best_right_count,
                    expandable=(node.depth + 1) < self.tree_config.get("max_depth"),
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
            print_profile(tree_growth_profile)

    def evaluate_cached_training_stream(
        self,
        tree: SingleTree,
        quantized_train_batches: list[tuple[np.ndarray, ...]],
        provider_kwargs: dict,
        profile: bool,
    ) -> tuple[float, float]:
        evaluation_profile = _start_profile("evaluation") if profile else None
        total_count = 0
        total_denominator = 0.0
        total_error = 0.0
        sum_prob = 0.0

        if self.training_config.get("predict_method") == "gpu":
            for batch in quantized_train_batches:
                pred_cpu = self.predict_batch(tree, bins=batch[0])
                error_sum, denominator = self.objective.mse_from_predictions(
                    pred_cpu,
                    batch[1],
                    batch[2] if self.objective.use_weights else None,
                )
                total_count += batch[0].shape[0]
                total_error += error_sum
                total_denominator += denominator
                sum_prob += float(np.sum(pred_cpu))
                if evaluation_profile is not None:
                    _update_profile(evaluation_profile)
        else:
            for x_cpu, y_cpu in GaussianClassStreamProvider(**provider_kwargs):
                cls_cpu = np.argmax(y_cpu, axis=1)
                pred_cpu = tree.predict_batch(x_cpu)
                error_sum, denominator = self.objective.mse_from_predictions(
                    pred_cpu,
                    cls_cpu,
                    self.objective.class_weights[cls_cpu] if self.objective.use_weights else None,
                )
                total_count += x_cpu.shape[0]
                total_error += error_sum
                total_denominator += denominator
                sum_prob += float(np.sum(pred_cpu))
                if evaluation_profile is not None:
                    _update_profile(evaluation_profile)

        if evaluation_profile is not None:
            _finish_profile(evaluation_profile)
            print_profile(evaluation_profile)
            print()
            print("Inference done.")

        return total_error / max(total_denominator, 1.0), sum_prob / max(total_count, 1)

    def profile_fresh_inference(self, tree: SingleTree, provider_kwargs: dict, profile: bool):
        if not profile or self.training_config.get("predict_method") != "gpu":
            return

        fresh_profile = _start_profile("fresh_inference")
        fresh_sum_prob = 0.0
        fresh_count = 0
        for x_cpu, _ in GaussianClassStreamProvider(**provider_kwargs):
            pred_cpu = self.predict_batch(tree, x=x_cpu)
            fresh_sum_prob += float(np.sum(pred_cpu))
            fresh_count += x_cpu.shape[0]
            _update_profile(fresh_profile)
        _finish_profile(fresh_profile)
        print_profile(fresh_profile)
        print(f"Fresh inference mean sum of class predictions: {fresh_sum_prob / max(fresh_count, 1):.6f}")

    def run(self, profile: bool) -> tuple[SingleTree, dict, float, float]:
        provider_kwargs = self.provider_kwargs()
        cuts_cpu, cuts_gpu = self.build_cuts(provider_kwargs)
        training_profile = _start_profile("training") if profile else None
        quantized_train_batches = self.build_quantized_cache(provider_kwargs, cuts_gpu, profile, training_profile)
        tree = SingleTree(self.dataset_config.get("n_classes"))
        self.fit(tree, quantized_train_batches, cuts_cpu, profile, training_profile)
        if training_profile is not None:
            _finish_profile(training_profile)
            print_profile(training_profile)
        tree.finalize_prediction_state()
        self._finalize_gpu_prediction_state(tree)
        train_mse, mean_sum_prob = self.evaluate_cached_training_stream(tree, quantized_train_batches, provider_kwargs, profile)
        self.profile_fresh_inference(tree, provider_kwargs, profile)
        return tree, provider_kwargs, train_mse, mean_sum_prob
