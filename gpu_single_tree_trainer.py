from __future__ import annotations

import os
import time
from dataclasses import dataclass

import cupy as cp
import numpy as np
from numba import cuda

from single_tree import AdditiveEnsemble, Node, SingleTree

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
def build_candidate_histograms_unweighted(bins, target_stats, row_slot, hist_count, hist_weight, hist_sum):
    i = cuda.grid(1)
    if i < bins.shape[0]:
        slot = row_slot[i]
        if slot >= 0:
            for f in range(bins.shape[1]):
                b = bins[i, f]
                cuda.atomic.add(hist_count, (slot, f, b), 1)
                cuda.atomic.add(hist_weight, (slot, f, b), 1.0)
                for c in range(target_stats.shape[1]):
                    cuda.atomic.add(hist_sum, (slot, f, b, c), target_stats[i, c])


@cuda.jit
def build_candidate_histograms_weighted(bins, target_stats, sample_weight, row_slot, hist_count, hist_weight, hist_sum):
    i = cuda.grid(1)
    if i < bins.shape[0]:
        slot = row_slot[i]
        if slot >= 0:
            weight = sample_weight[i]
            for f in range(bins.shape[1]):
                b = bins[i, f]
                cuda.atomic.add(hist_count, (slot, f, b), 1)
                cuda.atomic.add(hist_weight, (slot, f, b), weight)
                for c in range(target_stats.shape[1]):
                    cuda.atomic.add(hist_sum, (slot, f, b, c), weight * target_stats[i, c])


@cuda.jit
def evaluate_feature_splits(
    hist_count,
    hist_weight,
    hist_sum,
    min_samples_leaf,
    reg_lambda,
    slot_parent_count,
    slot_parent_weight,
    slot_parent_sum,
    feature_best_gain,
    feature_best_bin,
    feature_best_left_count,
    feature_best_right_count,
    feature_best_left_weight,
    feature_best_right_weight,
    feature_best_left_sum,
    feature_best_right_sum,
):
    slot, feature = cuda.grid(2)
    if slot >= hist_count.shape[0] or feature >= hist_count.shape[1]:
        return

    parent_count = 0
    parent_weight = 0.0
    for b in range(hist_count.shape[2]):
        parent_count += hist_count[slot, feature, b]
        parent_weight += hist_weight[slot, feature, b]

    if feature == 0:
        slot_parent_count[slot] = parent_count
        slot_parent_weight[slot] = parent_weight
        for c in range(hist_sum.shape[3]):
            s = 0.0
            for b in range(hist_sum.shape[2]):
                s += hist_sum[slot, feature, b, c]
            slot_parent_sum[slot, c] = s

    if parent_count <= 0 or parent_weight <= 0.0:
        feature_best_gain[slot, feature] = -1.0e30
        feature_best_bin[slot, feature] = -1
        feature_best_left_count[slot, feature] = 0
        feature_best_right_count[slot, feature] = 0
        feature_best_left_weight[slot, feature] = 0.0
        feature_best_right_weight[slot, feature] = 0.0
        for c in range(hist_sum.shape[3]):
            feature_best_left_sum[slot, feature, c] = 0.0
            feature_best_right_sum[slot, feature, c] = 0.0
        return

    parent_sum = cuda.local.array(16, dtype=np.float32)
    left_sum = cuda.local.array(16, dtype=np.float32)
    best_left_sum = cuda.local.array(16, dtype=np.float32)
    best_right_sum = cuda.local.array(16, dtype=np.float32)
    parent_score_num = 0.0
    for c in range(hist_sum.shape[3]):
        s = 0.0
        for b in range(hist_sum.shape[2]):
            s += hist_sum[slot, feature, b, c]
        parent_sum[c] = s
        left_sum[c] = 0.0
        parent_score_num += s * s
    parent_score = parent_score_num / (parent_weight + reg_lambda)

    left_count = 0
    left_weight = 0.0
    best_gain = -1.0e30
    best_bin = -1
    best_left_count = 0
    best_right_count = 0
    best_left_weight = 0.0
    best_right_weight = 0.0

    for split_bin in range(hist_count.shape[2] - 1):
        left_count += hist_count[slot, feature, split_bin]
        left_weight += hist_weight[slot, feature, split_bin]
        right_count = parent_count - left_count
        right_weight = parent_weight - left_weight

        for c in range(hist_sum.shape[3]):
            left_sum[c] += hist_sum[slot, feature, split_bin, c]

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
            best_left_weight = left_weight
            best_right_weight = right_weight
            for c in range(hist_sum.shape[3]):
                best_left_sum[c] = left_sum[c]
                best_right_sum[c] = parent_sum[c] - left_sum[c]

    feature_best_gain[slot, feature] = best_gain
    feature_best_bin[slot, feature] = best_bin
    feature_best_left_count[slot, feature] = best_left_count
    feature_best_right_count[slot, feature] = best_right_count
    feature_best_left_weight[slot, feature] = best_left_weight
    feature_best_right_weight[slot, feature] = best_right_weight
    for c in range(hist_sum.shape[3]):
        feature_best_left_sum[slot, feature, c] = best_left_sum[c]
        feature_best_right_sum[slot, feature, c] = best_right_sum[c]


@cuda.jit
def reduce_feature_bests(
    feature_best_gain,
    feature_best_bin,
    feature_best_left_count,
    feature_best_right_count,
    feature_best_left_weight,
    feature_best_right_weight,
    feature_best_left_sum,
    feature_best_right_sum,
    slot_best_gain,
    slot_best_feature,
    slot_best_bin,
    slot_best_left_count,
    slot_best_right_count,
    slot_best_left_weight,
    slot_best_right_weight,
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
        best_left_weight = 0.0
        best_right_weight = 0.0
        for feature in range(feature_best_gain.shape[1]):
            gain = feature_best_gain[slot, feature]
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_bin = feature_best_bin[slot, feature]
                best_left_count = feature_best_left_count[slot, feature]
                best_right_count = feature_best_right_count[slot, feature]
                best_left_weight = feature_best_left_weight[slot, feature]
                best_right_weight = feature_best_right_weight[slot, feature]
                for c in range(feature_best_left_sum.shape[2]):
                    slot_best_left_sum[slot, c] = feature_best_left_sum[slot, feature, c]
                    slot_best_right_sum[slot, c] = feature_best_right_sum[slot, feature, c]
        slot_best_gain[slot] = best_gain
        slot_best_feature[slot] = best_feature
        slot_best_bin[slot] = best_bin
        slot_best_left_count[slot] = best_left_count
        slot_best_right_count[slot] = best_right_count
        slot_best_left_weight[slot] = best_left_weight
        slot_best_right_weight[slot] = best_right_weight


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
    rss = _rss_bytes()
    return {
        "stage": stage_name,
        "wall_start": time.perf_counter(),
        "cpu_start": time.process_time(),
        "rss_start": rss,
        "rss_max": rss,
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


@dataclass
class TrainingCacheBatch:
    bins: np.ndarray
    target_stats: np.ndarray
    sample_weight: np.ndarray | None
    prediction: np.ndarray


class TrainingCache:
    def __init__(self, batches: list[TrainingCacheBatch], prediction_dim: int):
        self.batches = batches
        self.prediction_dim = prediction_dim

    def __iter__(self):
        return iter(self.batches)

    def initialize_predictions(self, base_prediction: np.ndarray):
        base = np.asarray(base_prediction, dtype=np.float32)
        for batch in self.batches:
            batch.prediction[...] = base

    def target_stat_mean(self) -> np.ndarray:
        total = np.zeros((self.prediction_dim,), dtype=np.float64)
        denominator = 0.0
        for batch in self.batches:
            if batch.sample_weight is None:
                total += np.sum(batch.target_stats, axis=0, dtype=np.float64)
                denominator += batch.target_stats.shape[0]
            else:
                total += np.sum(batch.sample_weight[:, None] * batch.target_stats, axis=0, dtype=np.float64)
                denominator += float(np.sum(batch.sample_weight))
        return (total / max(denominator, 1.0)).astype(np.float32)

    def monitor_metric(self, family) -> tuple[float, float, float]:
        total_error = 0.0
        total_weight = 0.0
        total_sum = 0.0
        total_count = 0
        for batch in self.batches:
            error_sum, denominator = family.monitor_metric(batch.prediction, batch.target_stats, batch.sample_weight)
            total_error += error_sum
            total_weight += denominator
            total_sum += float(np.sum(batch.prediction))
            total_count += batch.prediction.shape[0]
        return total_error / max(total_weight, 1.0), total_sum / max(total_count, 1), total_weight


class GpuSingleTreeTrainer:
    def __init__(self, tree_config: dict, dataset_config: dict, training_config: dict, family):
        self.tree_config = tree_config
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.family = family

    @property
    def device_name(self) -> str:
        return str(cuda.get_current_device().name)

    def provider_kwargs(self) -> dict:
        return self.family.provider_kwargs(self.dataset_config)

    def fresh_inference_dataset_config(self) -> dict:
        fresh_config = dict(self.dataset_config)
        if self.training_config.get("fresh_inference_batch_size") is not None:
            fresh_config["batch_size"] = self.training_config.get("fresh_inference_batch_size")
        if self.training_config.get("fresh_inference_n_batches") is not None:
            fresh_config["n_batches"] = self.training_config.get("fresh_inference_n_batches")
        return fresh_config

    def _bin_cp_dtype(self):
        return cp.uint8 if self.tree_config.get("max_bin") <= 256 else cp.uint16

    def build_cuts(self, provider_kwargs: dict) -> tuple[np.ndarray, cp.ndarray]:
        cut_batches = []
        sampled_rows = 0
        for batch in self.family.stream_batches(self.dataset_config):
            take = min(batch.x.shape[0], self.tree_config.get("cut_sample_rows") - sampled_rows)
            if take > 0:
                cut_batches.append(batch.x[:take].copy())
                sampled_rows += take
            if sampled_rows >= self.tree_config.get("cut_sample_rows"):
                break
        cut_sample = np.concatenate(cut_batches, axis=0)
        quantile_levels = np.linspace(0.0, 1.0, self.tree_config.get("max_bin") + 1, dtype=np.float64)[1:-1]
        cuts_cpu = np.quantile(cut_sample, quantile_levels, axis=0).T.astype(np.float32)
        return cuts_cpu, cp.asarray(cuts_cpu)

    def build_training_cache(self, cuts_gpu: cp.ndarray, profile: bool, training_profile: dict | None) -> TrainingCache:
        cache_profile = _start_profile("cache_build") if profile else None
        cache_x_gpu = None
        cache_bins_gpu = None
        cache_batches = []
        prediction_dim = self.dataset_config.get("n_classes")
        for batch in self.family.stream_batches(self.dataset_config):
            if cache_x_gpu is None or cache_x_gpu.shape != batch.x.shape:
                cache_x_gpu = cp.empty(batch.x.shape, dtype=cp.float32)
                cache_bins_gpu = cp.empty(batch.x.shape, dtype=self._bin_cp_dtype())
            cache_x_gpu.set(batch.x)
            quant_blocks = ((cache_x_gpu.shape[0] + 15) // 16, (cache_x_gpu.shape[1] + 15) // 16)
            quantize_batch[quant_blocks, (16, 16)](cache_x_gpu, cuts_gpu, cache_bins_gpu)
            cuda.synchronize()
            cache_batches.append(
                TrainingCacheBatch(
                    bins=cp.asnumpy(cache_bins_gpu),
                    target_stats=batch.target_stats.astype(np.float32, copy=False),
                    sample_weight=None if batch.sample_weight is None else batch.sample_weight.astype(np.float32, copy=False),
                    prediction=np.empty((batch.target_stats.shape[0], prediction_dim), dtype=np.float32),
                )
            )
            if cache_profile is not None:
                _update_profile(cache_profile)
            if training_profile is not None:
                _update_profile(training_profile)
        if cache_profile is not None:
            _finish_profile(cache_profile)
            print_profile(cache_profile)
        return TrainingCache(cache_batches, prediction_dim)

    def _finalize_gpu_prediction_state(self, tree: SingleTree):
        tree.split_feature_gpu = cp.asarray(tree.split_feature_cpu)
        tree.split_bin_gpu = cp.asarray(tree.split_bin_cpu)
        tree.split_threshold_gpu = cp.asarray(tree.split_threshold_cpu)
        tree.left_child_gpu = cp.asarray(tree.left_child_cpu)
        tree.right_child_gpu = cp.asarray(tree.right_child_cpu)
        tree.is_leaf_gpu = cp.asarray(tree.is_leaf_cpu)
        tree.leaf_value_gpu = cp.asarray(tree.leaf_value_cpu)

    def predict_tree_batch(self, tree: SingleTree, x: np.ndarray | None = None, bins: np.ndarray | None = None) -> np.ndarray:
        if (x is None) == (bins is None):
            raise ValueError("Provide exactly one of x or bins.")
        n_rows = x.shape[0] if x is not None else bins.shape[0]
        pred_gpu = cp.empty((n_rows, tree.prediction_dim), dtype=cp.float32)
        threads_per_block = self.training_config.get("threads_per_block")
        blocks_1d = (n_rows + threads_per_block - 1) // threads_per_block
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

    def predict_model_batch(self, model: AdditiveEnsemble, x: np.ndarray) -> np.ndarray:
        pred = np.repeat(model.base_prediction[None, :], x.shape[0], axis=0)
        for tree in model.trees:
            pred += model.learning_rate * self.predict_tree_batch(tree, x=x)
        return self.family.project_prediction(pred)

    def _leaf_value(self, target_stat_sum: np.ndarray, total_weight: float) -> np.ndarray:
        return target_stat_sum / (total_weight + self.tree_config.get("reg_lambda"))

    def _leaf_score(self, target_stat_sum: np.ndarray, total_weight: float) -> float:
        if total_weight <= 0.0:
            return -np.inf
        return float(np.dot(target_stat_sum, target_stat_sum) / (total_weight + self.tree_config.get("reg_lambda")))

    def fit_tree(
        self,
        training_cache: TrainingCache,
        cuts_cpu: np.ndarray,
        profile: bool,
        training_profile: dict | None,
        round_idx: int,
    ) -> SingleTree:
        tree = SingleTree(self.dataset_config.get("n_classes"))
        tree_profile = _start_profile(f"tree_growth_round_{round_idx + 1}") if profile else None
        threads_per_block = self.training_config.get("threads_per_block")

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

            n_slots = len(candidate_node_ids)
            n_features = self.dataset_config.get("n_features")
            n_bins = self.tree_config.get("max_bin")
            pred_dim = self.dataset_config.get("n_classes")
            hist_count_gpu = cp.zeros((n_slots, n_features, n_bins), dtype=cp.int32)
            hist_weight_gpu = cp.zeros((n_slots, n_features, n_bins), dtype=cp.float32)
            hist_sum_gpu = cp.zeros((n_slots, n_features, n_bins, pred_dim), dtype=cp.float32)

            for batch in training_cache:
                bins_gpu = cp.asarray(batch.bins)
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
                residual_gpu = cp.asarray(batch.target_stats - batch.prediction, dtype=cp.float32)
                if batch.sample_weight is None:
                    build_candidate_histograms_unweighted[blocks_1d, threads_per_block](
                        bins_gpu,
                        residual_gpu,
                        row_slot_gpu,
                        hist_count_gpu,
                        hist_weight_gpu,
                        hist_sum_gpu,
                    )
                else:
                    build_candidate_histograms_weighted[blocks_1d, threads_per_block](
                        bins_gpu,
                        residual_gpu,
                        cp.asarray(batch.sample_weight),
                        row_slot_gpu,
                        hist_count_gpu,
                        hist_weight_gpu,
                        hist_sum_gpu,
                    )
                if tree_profile is not None:
                    _update_profile(tree_profile)
                if training_profile is not None:
                    _update_profile(training_profile)

            slot_parent_count_gpu = cp.zeros((n_slots,), dtype=cp.int32)
            slot_parent_weight_gpu = cp.zeros((n_slots,), dtype=cp.float32)
            slot_parent_sum_gpu = cp.zeros((n_slots, pred_dim), dtype=cp.float32)
            feature_best_gain_gpu = cp.full((n_slots, n_features), -1.0e30, dtype=cp.float32)
            feature_best_bin_gpu = cp.full((n_slots, n_features), -1, dtype=cp.int32)
            feature_best_left_count_gpu = cp.zeros((n_slots, n_features), dtype=cp.int32)
            feature_best_right_count_gpu = cp.zeros((n_slots, n_features), dtype=cp.int32)
            feature_best_left_weight_gpu = cp.zeros((n_slots, n_features), dtype=cp.float32)
            feature_best_right_weight_gpu = cp.zeros((n_slots, n_features), dtype=cp.float32)
            feature_best_left_sum_gpu = cp.zeros((n_slots, n_features, pred_dim), dtype=cp.float32)
            feature_best_right_sum_gpu = cp.zeros((n_slots, n_features, pred_dim), dtype=cp.float32)

            eval_blocks = ((n_slots + 7) // 8, (n_features + 7) // 8)
            evaluate_feature_splits[eval_blocks, (8, 8)](
                hist_count_gpu,
                hist_weight_gpu,
                hist_sum_gpu,
                self.tree_config.get("min_samples_leaf"),
                self.tree_config.get("reg_lambda"),
                slot_parent_count_gpu,
                slot_parent_weight_gpu,
                slot_parent_sum_gpu,
                feature_best_gain_gpu,
                feature_best_bin_gpu,
                feature_best_left_count_gpu,
                feature_best_right_count_gpu,
                feature_best_left_weight_gpu,
                feature_best_right_weight_gpu,
                feature_best_left_sum_gpu,
                feature_best_right_sum_gpu,
            )

            slot_best_gain_gpu = cp.full((n_slots,), -1.0e30, dtype=cp.float32)
            slot_best_feature_gpu = cp.full((n_slots,), -1, dtype=cp.int32)
            slot_best_bin_gpu = cp.full((n_slots,), -1, dtype=cp.int32)
            slot_best_left_count_gpu = cp.zeros((n_slots,), dtype=cp.int32)
            slot_best_right_count_gpu = cp.zeros((n_slots,), dtype=cp.int32)
            slot_best_left_weight_gpu = cp.zeros((n_slots,), dtype=cp.float32)
            slot_best_right_weight_gpu = cp.zeros((n_slots,), dtype=cp.float32)
            slot_best_left_sum_gpu = cp.zeros((n_slots, pred_dim), dtype=cp.float32)
            slot_best_right_sum_gpu = cp.zeros((n_slots, pred_dim), dtype=cp.float32)

            reduce_blocks = (n_slots + threads_per_block - 1) // threads_per_block
            reduce_feature_bests[reduce_blocks, threads_per_block](
                feature_best_gain_gpu,
                feature_best_bin_gpu,
                feature_best_left_count_gpu,
                feature_best_right_count_gpu,
                feature_best_left_weight_gpu,
                feature_best_right_weight_gpu,
                feature_best_left_sum_gpu,
                feature_best_right_sum_gpu,
                slot_best_gain_gpu,
                slot_best_feature_gpu,
                slot_best_bin_gpu,
                slot_best_left_count_gpu,
                slot_best_right_count_gpu,
                slot_best_left_weight_gpu,
                slot_best_right_weight_gpu,
                slot_best_left_sum_gpu,
                slot_best_right_sum_gpu,
            )
            cuda.synchronize()

            if tree_profile is not None:
                _update_profile(tree_profile)
            if training_profile is not None:
                _update_profile(training_profile)

            slot_parent_count = cp.asnumpy(slot_parent_count_gpu)
            slot_parent_weight = cp.asnumpy(slot_parent_weight_gpu)
            slot_parent_sum = cp.asnumpy(slot_parent_sum_gpu)
            slot_best_gain = cp.asnumpy(slot_best_gain_gpu)
            slot_best_feature = cp.asnumpy(slot_best_feature_gpu)
            slot_best_bin = cp.asnumpy(slot_best_bin_gpu)
            slot_best_left_count = cp.asnumpy(slot_best_left_count_gpu)
            slot_best_right_count = cp.asnumpy(slot_best_right_count_gpu)
            slot_best_left_weight = cp.asnumpy(slot_best_left_weight_gpu)
            slot_best_right_weight = cp.asnumpy(slot_best_right_weight_gpu)
            slot_best_left_sum = cp.asnumpy(slot_best_left_sum_gpu)
            slot_best_right_sum = cp.asnumpy(slot_best_right_sum_gpu)

            split_plans = []
            for slot, node_id in enumerate(candidate_node_ids):
                node = tree.nodes[node_id]
                node.count = int(slot_parent_count[slot])
                parent_sum = slot_parent_sum[slot].astype(np.float64)
                parent_weight = float(slot_parent_weight[slot])
                node.value = self._leaf_value(parent_sum.astype(np.float32), parent_weight)
                parent_score = self._leaf_score(parent_sum, parent_weight)
                if tree.root_score is None and node_id == 0:
                    tree.root_score = parent_score
                    tree.root_weight = parent_weight
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
                best_left_weight = float(slot_best_left_weight[slot])
                best_right_weight = float(slot_best_right_weight[slot])
                node.gain = best_gain
                node.split_feature = best_feature
                node.split_bin = best_bin
                node.split_threshold = float(cuts_cpu[best_feature, best_bin])
                node.best_left_count = best_left_count
                node.best_right_count = best_right_count
                node.best_left_value = self._leaf_value(best_left_sum, best_left_weight)
                node.best_right_value = self._leaf_value(best_right_sum, best_right_weight)
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

            if tree_profile is not None:
                _update_profile(tree_profile)
            if training_profile is not None:
                _update_profile(training_profile)

        if tree_profile is not None:
            _finish_profile(tree_profile)
            print_profile(tree_profile)
        tree.finalize_prediction_state()
        self._finalize_gpu_prediction_state(tree)
        return tree

    def update_training_cache(self, training_cache: TrainingCache, tree: SingleTree, learning_rate: float, profile: bool, training_profile: dict | None, round_idx: int):
        update_profile = _start_profile(f"cache_update_round_{round_idx + 1}") if profile else None
        for batch in training_cache:
            batch.prediction += learning_rate * self.predict_tree_batch(tree, bins=batch.bins)
            batch.prediction[...] = self.family.project_prediction(batch.prediction)
            if update_profile is not None:
                _update_profile(update_profile)
            if training_profile is not None:
                _update_profile(training_profile)
        if update_profile is not None:
            _finish_profile(update_profile)
            print_profile(update_profile)

    def evaluate_cached_training_stream(self, training_cache: TrainingCache, profile: bool) -> tuple[float, float]:
        evaluation_profile = _start_profile("evaluation") if profile else None
        train_metric, mean_sum_prob, _ = training_cache.monitor_metric(self.family)
        if evaluation_profile is not None:
            _update_profile(evaluation_profile)
            _finish_profile(evaluation_profile)
            print_profile(evaluation_profile)
            print()
            print("Inference done.")
        return train_metric, mean_sum_prob

    def profile_fresh_inference(self, model: AdditiveEnsemble, profile: bool):
        if not profile:
            return
        fresh_profile = _start_profile("fresh_inference")
        fresh_sum_prob = 0.0
        fresh_count = 0
        fresh_dataset_config = self.fresh_inference_dataset_config()
        for batch in self.family.stream_batches(fresh_dataset_config):
            if self.training_config.get("predict_method") == "gpu":
                pred_cpu = self.predict_model_batch(model, batch.x)
            else:
                pred_cpu = model.predict_batch(
                    batch.x,
                    predict_method="cpu",
                    project_prediction=self.family.project_prediction,
                    cpu_predictor=self.training_config.get("cpu_predictor"),
                )
            fresh_sum_prob += float(np.sum(pred_cpu))
            fresh_count += batch.x.shape[0]
            _update_profile(fresh_profile)
        _finish_profile(fresh_profile)
        print_profile(fresh_profile)
        print(f"Fresh inference mean sum of class predictions: {fresh_sum_prob / max(fresh_count, 1):.6f}")

    def run(self, profile: bool) -> tuple[AdditiveEnsemble, dict, float, float, list[float]]:
        provider_kwargs = self.provider_kwargs()
        cuts_cpu, cuts_gpu = self.build_cuts(provider_kwargs)
        training_profile = _start_profile("training") if profile else None
        training_cache = self.build_training_cache(cuts_gpu, profile, training_profile)
        base_prediction = self.family.base_prediction(training_cache.target_stat_mean())
        training_cache.initialize_predictions(base_prediction)
        model = AdditiveEnsemble(
            prediction_dim=self.dataset_config.get("n_classes"),
            base_prediction=base_prediction,
            learning_rate=self.training_config.get("learning_rate"),
        )

        loss_history = []
        initial_loss, initial_mean_sum_prob, _ = training_cache.monitor_metric(self.family)
        loss_history.append(initial_loss)
        print(f"Initial train {self.family.monitor_name}: {initial_loss:.6f}")

        for round_idx in range(self.training_config.get("n_boost_rounds")):
            tree = self.fit_tree(training_cache, cuts_cpu, profile, training_profile, round_idx)
            model.add_tree(tree)
            self.update_training_cache(training_cache, tree, model.learning_rate, profile, training_profile, round_idx)
            round_loss, _, _ = training_cache.monitor_metric(self.family)
            loss_history.append(round_loss)
            print(
                f"After round {round_idx + 1}: train {self.family.monitor_name}={round_loss:.6f} "
                f"nodes={len(tree.nodes)} leaves={tree.n_leaves}"
            )

        if training_profile is not None:
            _finish_profile(training_profile)
            print_profile(training_profile)

        train_metric, mean_sum_prob = self.evaluate_cached_training_stream(training_cache, profile)
        self.profile_fresh_inference(model, profile)
        return model, provider_kwargs, train_metric, mean_sum_prob, loss_history
