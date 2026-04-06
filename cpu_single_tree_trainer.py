from __future__ import annotations

import os
import time

import numpy as np
from numba import njit, prange, set_num_threads

from single_tree import AdditiveEnsemble, Node, SingleTree
from training_cache import TrainingCache, TrainingCacheBatch

try:
    import psutil
except Exception:
    psutil = None

try:
    import resource
except Exception:
    resource = None


@njit(cache=True)
def route_rows_to_candidate_slots_cpu(
    bins,
    split_feature,
    split_bin,
    left_child,
    right_child,
    is_leaf,
    candidate_slot_of_node,
    out_slot,
):
    for i in range(bins.shape[0]):
        node = 0
        while is_leaf[node] == 0:
            feature = split_feature[node]
            threshold_bin = split_bin[node]
            if bins[i, feature] <= threshold_bin:
                node = left_child[node]
            else:
                node = right_child[node]
        out_slot[i] = candidate_slot_of_node[node]


@njit(cache=True, parallel=True)
def route_rows_to_candidate_slots_cpu_parallel(
    bins,
    split_feature,
    split_bin,
    left_child,
    right_child,
    is_leaf,
    candidate_slot_of_node,
    out_slot,
):
    for i in prange(bins.shape[0]):
        node = 0
        while is_leaf[node] == 0:
            feature = split_feature[node]
            threshold_bin = split_bin[node]
            if bins[i, feature] <= threshold_bin:
                node = left_child[node]
            else:
                node = right_child[node]
        out_slot[i] = candidate_slot_of_node[node]


@njit(cache=True)
def build_candidate_histograms_unweighted_cpu(bins, target_stats, row_slot, hist_count, hist_weight, hist_sum):
    for i in range(bins.shape[0]):
        slot = row_slot[i]
        if slot >= 0:
            for f in range(bins.shape[1]):
                b = bins[i, f]
                hist_count[slot, f, b] += 1
                hist_weight[slot, f, b] += 1.0
                for c in range(target_stats.shape[1]):
                    hist_sum[slot, f, b, c] += target_stats[i, c]


@njit(cache=True, parallel=True)
def build_candidate_histograms_unweighted_cpu_parallel(bins, target_stats, row_slot, hist_count, hist_weight, hist_sum):
    for f in prange(bins.shape[1]):
        for i in range(bins.shape[0]):
            slot = row_slot[i]
            if slot >= 0:
                b = bins[i, f]
                hist_count[slot, f, b] += 1
                hist_weight[slot, f, b] += 1.0
                for c in range(target_stats.shape[1]):
                    hist_sum[slot, f, b, c] += target_stats[i, c]


@njit(cache=True)
def build_candidate_histograms_weighted_cpu(bins, target_stats, sample_weight, row_slot, hist_count, hist_weight, hist_sum):
    for i in range(bins.shape[0]):
        slot = row_slot[i]
        if slot >= 0:
            weight = sample_weight[i]
            for f in range(bins.shape[1]):
                b = bins[i, f]
                hist_count[slot, f, b] += 1
                hist_weight[slot, f, b] += weight
                for c in range(target_stats.shape[1]):
                    hist_sum[slot, f, b, c] += weight * target_stats[i, c]


@njit(cache=True, parallel=True)
def build_candidate_histograms_weighted_cpu_parallel(bins, target_stats, sample_weight, row_slot, hist_count, hist_weight, hist_sum):
    for f in prange(bins.shape[1]):
        for i in range(bins.shape[0]):
            slot = row_slot[i]
            if slot >= 0:
                weight = sample_weight[i]
                b = bins[i, f]
                hist_count[slot, f, b] += 1
                hist_weight[slot, f, b] += weight
                for c in range(target_stats.shape[1]):
                    hist_sum[slot, f, b, c] += weight * target_stats[i, c]


@njit(cache=True)
def evaluate_slot_bests_cpu(
    hist_count,
    hist_weight,
    hist_sum,
    min_samples_leaf,
    reg_lambda,
    slot_parent_count,
    slot_parent_weight,
    slot_parent_sum,
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
    n_slots = hist_count.shape[0]
    n_features = hist_count.shape[1]
    n_bins = hist_count.shape[2]
    pred_dim = hist_sum.shape[3]

    for slot in range(n_slots):
        parent_count = 0
        parent_weight = 0.0
        for b in range(n_bins):
            parent_count += hist_count[slot, 0, b]
            parent_weight += hist_weight[slot, 0, b]
        slot_parent_count[slot] = parent_count
        slot_parent_weight[slot] = parent_weight
        for c in range(pred_dim):
            s = 0.0
            for b in range(n_bins):
                s += hist_sum[slot, 0, b, c]
            slot_parent_sum[slot, c] = s

        best_gain = -1.0e30
        best_feature = -1
        best_bin = -1
        best_left_count = 0
        best_right_count = 0
        best_left_weight = 0.0
        best_right_weight = 0.0

        if parent_count <= 0 or parent_weight <= 0.0:
            slot_best_gain[slot] = best_gain
            slot_best_feature[slot] = best_feature
            slot_best_bin[slot] = best_bin
            slot_best_left_count[slot] = best_left_count
            slot_best_right_count[slot] = best_right_count
            slot_best_left_weight[slot] = best_left_weight
            slot_best_right_weight[slot] = best_right_weight
            for c in range(pred_dim):
                slot_best_left_sum[slot, c] = 0.0
                slot_best_right_sum[slot, c] = 0.0
            continue

        for feature in range(n_features):
            parent_score_num = 0.0
            parent_sum = np.zeros(pred_dim, dtype=np.float32)
            for c in range(pred_dim):
                s = 0.0
                for b in range(n_bins):
                    s += hist_sum[slot, feature, b, c]
                parent_sum[c] = s
                parent_score_num += s * s
            parent_score = parent_score_num / (parent_weight + reg_lambda)

            left_count = 0
            left_weight = 0.0
            left_sum = np.zeros(pred_dim, dtype=np.float32)

            feature_best_gain = -1.0e30
            feature_best_bin = -1
            feature_best_left_count = 0
            feature_best_right_count = 0
            feature_best_left_weight = 0.0
            feature_best_right_weight = 0.0
            feature_best_left_sum = np.zeros(pred_dim, dtype=np.float32)
            feature_best_right_sum = np.zeros(pred_dim, dtype=np.float32)

            for split_b in range(n_bins - 1):
                left_count += hist_count[slot, feature, split_b]
                left_weight += hist_weight[slot, feature, split_b]
                right_count = parent_count - left_count
                right_weight = parent_weight - left_weight

                for c in range(pred_dim):
                    left_sum[c] += hist_sum[slot, feature, split_b, c]

                if left_count < min_samples_leaf or right_count < min_samples_leaf or left_weight <= 0.0 or right_weight <= 0.0:
                    continue

                left_score_num = 0.0
                right_score_num = 0.0
                for c in range(pred_dim):
                    right_sum_c = parent_sum[c] - left_sum[c]
                    left_score_num += left_sum[c] * left_sum[c]
                    right_score_num += right_sum_c * right_sum_c
                gain = left_score_num / (left_weight + reg_lambda) + right_score_num / (right_weight + reg_lambda) - parent_score
                if gain > feature_best_gain:
                    feature_best_gain = gain
                    feature_best_bin = split_b
                    feature_best_left_count = left_count
                    feature_best_right_count = right_count
                    feature_best_left_weight = left_weight
                    feature_best_right_weight = right_weight
                    for c in range(pred_dim):
                        feature_best_left_sum[c] = left_sum[c]
                        feature_best_right_sum[c] = parent_sum[c] - left_sum[c]

            if feature_best_gain > best_gain:
                best_gain = feature_best_gain
                best_feature = feature
                best_bin = feature_best_bin
                best_left_count = feature_best_left_count
                best_right_count = feature_best_right_count
                best_left_weight = feature_best_left_weight
                best_right_weight = feature_best_right_weight
                for c in range(pred_dim):
                    slot_best_left_sum[slot, c] = feature_best_left_sum[c]
                    slot_best_right_sum[slot, c] = feature_best_right_sum[c]

        slot_best_gain[slot] = best_gain
        slot_best_feature[slot] = best_feature
        slot_best_bin[slot] = best_bin
        slot_best_left_count[slot] = best_left_count
        slot_best_right_count[slot] = best_right_count
        slot_best_left_weight[slot] = best_left_weight
        slot_best_right_weight[slot] = best_right_weight


@njit(cache=True, parallel=True)
def evaluate_slot_bests_cpu_parallel(
    hist_count,
    hist_weight,
    hist_sum,
    min_samples_leaf,
    reg_lambda,
    slot_parent_count,
    slot_parent_weight,
    slot_parent_sum,
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
    n_slots = hist_count.shape[0]
    n_features = hist_count.shape[1]
    n_bins = hist_count.shape[2]
    pred_dim = hist_sum.shape[3]

    for slot in prange(n_slots):
        parent_count = 0
        parent_weight = 0.0
        for b in range(n_bins):
            parent_count += hist_count[slot, 0, b]
            parent_weight += hist_weight[slot, 0, b]
        slot_parent_count[slot] = parent_count
        slot_parent_weight[slot] = parent_weight
        for c in range(pred_dim):
            s = 0.0
            for b in range(n_bins):
                s += hist_sum[slot, 0, b, c]
            slot_parent_sum[slot, c] = s

        best_gain = -1.0e30
        best_feature = -1
        best_bin = -1
        best_left_count = 0
        best_right_count = 0
        best_left_weight = 0.0
        best_right_weight = 0.0

        if parent_count <= 0 or parent_weight <= 0.0:
            slot_best_gain[slot] = best_gain
            slot_best_feature[slot] = best_feature
            slot_best_bin[slot] = best_bin
            slot_best_left_count[slot] = best_left_count
            slot_best_right_count[slot] = best_right_count
            slot_best_left_weight[slot] = best_left_weight
            slot_best_right_weight[slot] = best_right_weight
            for c in range(pred_dim):
                slot_best_left_sum[slot, c] = 0.0
                slot_best_right_sum[slot, c] = 0.0
            continue

        for feature in range(n_features):
            parent_score_num = 0.0
            parent_sum = np.zeros(pred_dim, dtype=np.float32)
            for c in range(pred_dim):
                s = 0.0
                for b in range(n_bins):
                    s += hist_sum[slot, feature, b, c]
                parent_sum[c] = s
                parent_score_num += s * s
            parent_score = parent_score_num / (parent_weight + reg_lambda)

            left_count = 0
            left_weight = 0.0
            left_sum = np.zeros(pred_dim, dtype=np.float32)

            feature_best_gain = -1.0e30
            feature_best_bin = -1
            feature_best_left_count = 0
            feature_best_right_count = 0
            feature_best_left_weight = 0.0
            feature_best_right_weight = 0.0
            feature_best_left_sum = np.zeros(pred_dim, dtype=np.float32)
            feature_best_right_sum = np.zeros(pred_dim, dtype=np.float32)

            for split_b in range(n_bins - 1):
                left_count += hist_count[slot, feature, split_b]
                left_weight += hist_weight[slot, feature, split_b]
                right_count = parent_count - left_count
                right_weight = parent_weight - left_weight

                for c in range(pred_dim):
                    left_sum[c] += hist_sum[slot, feature, split_b, c]

                if left_count < min_samples_leaf or right_count < min_samples_leaf or left_weight <= 0.0 or right_weight <= 0.0:
                    continue

                left_score_num = 0.0
                right_score_num = 0.0
                for c in range(pred_dim):
                    right_sum_c = parent_sum[c] - left_sum[c]
                    left_score_num += left_sum[c] * left_sum[c]
                    right_score_num += right_sum_c * right_sum_c
                gain = left_score_num / (left_weight + reg_lambda) + right_score_num / (right_weight + reg_lambda) - parent_score
                if gain > feature_best_gain:
                    feature_best_gain = gain
                    feature_best_bin = split_b
                    feature_best_left_count = left_count
                    feature_best_right_count = right_count
                    feature_best_left_weight = left_weight
                    feature_best_right_weight = right_weight
                    for c in range(pred_dim):
                        feature_best_left_sum[c] = left_sum[c]
                        feature_best_right_sum[c] = parent_sum[c] - left_sum[c]

            if feature_best_gain > best_gain:
                best_gain = feature_best_gain
                best_feature = feature
                best_bin = feature_best_bin
                best_left_count = feature_best_left_count
                best_right_count = feature_best_right_count
                best_left_weight = feature_best_left_weight
                best_right_weight = feature_best_right_weight
                for c in range(pred_dim):
                    slot_best_left_sum[slot, c] = feature_best_left_sum[c]
                    slot_best_right_sum[slot, c] = feature_best_right_sum[c]

        slot_best_gain[slot] = best_gain
        slot_best_feature[slot] = best_feature
        slot_best_bin[slot] = best_bin
        slot_best_left_count[slot] = best_left_count
        slot_best_right_count[slot] = best_right_count
        slot_best_left_weight[slot] = best_left_weight
        slot_best_right_weight[slot] = best_right_weight


def _rss_bytes():
    if psutil is not None:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    if resource is not None:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if os.uname().sysname.lower() == "darwin":
            return rss
        return rss * 1024
    return 0


def _start_profile(stage_name: str):
    rss = _rss_bytes()
    return {
        "stage": stage_name,
        "wall_start": time.perf_counter(),
        "cpu_start": time.process_time(),
        "rss_start": rss,
        "rss_max": rss,
        "gpu_used_start": 0,
        "gpu_used_max": 0,
        "gpu_pool_used_start": 0,
        "gpu_pool_used_max": 0,
        "gpu_pool_total_start": 0,
        "gpu_pool_total_max": 0,
    }


def _update_profile(profile_state: dict):
    rss = _rss_bytes()
    profile_state["rss_max"] = max(profile_state["rss_max"], rss)


def _finish_profile(profile_state: dict):
    profile_state["wall_end"] = time.perf_counter()
    profile_state["cpu_end"] = time.process_time()
    profile_state["rss_end"] = _rss_bytes()
    profile_state["gpu_used_end"] = 0
    profile_state["gpu_pool_used_end"] = 0
    profile_state["gpu_pool_total_end"] = 0
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
    print("  gpu_used_mb=start=0.0 end=0.0 max=0.0")
    print("  gpu_pool_used_mb=start=0.0 end=0.0 max=0.0")
    print("  gpu_pool_reserved_mb=start=0.0 end=0.0 max=0.0")


class CpuSingleTreeTrainer:
    def __init__(self, tree_config: dict, dataset_config: dict, training_config: dict, family):
        self.tree_config = tree_config
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.family = family
        self.cpu_threads = int(training_config.get("cpu_threads"))
        set_num_threads(self.cpu_threads)

    @property
    def device_name(self) -> str:
        return "CPU"

    def provider_kwargs(self) -> dict:
        return self.family.provider_kwargs(self.dataset_config)

    def fresh_inference_dataset_config(self) -> dict:
        fresh_config = dict(self.dataset_config)
        if self.training_config.get("fresh_inference_batch_size") is not None:
            fresh_config["batch_size"] = self.training_config.get("fresh_inference_batch_size")
        if self.training_config.get("fresh_inference_n_batches") is not None:
            fresh_config["n_batches"] = self.training_config.get("fresh_inference_n_batches")
        return fresh_config

    def _bin_dtype(self):
        return np.uint8 if self.tree_config.get("max_bin") <= 256 else np.uint16

    def build_cuts(self, provider_kwargs: dict) -> np.ndarray:
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
        return np.quantile(cut_sample, quantile_levels, axis=0).T.astype(np.float32)

    def quantize_batch(self, x: np.ndarray, cuts_cpu: np.ndarray) -> np.ndarray:
        bins = np.empty(x.shape, dtype=self._bin_dtype())
        for feature in range(x.shape[1]):
            bins[:, feature] = np.searchsorted(cuts_cpu[feature], x[:, feature], side="left")
        return bins

    def build_training_cache(self, cuts_cpu: np.ndarray, profile: bool, training_profile: dict | None) -> TrainingCache:
        cache_profile = _start_profile("cache_build") if profile else None
        cache_batches = []
        prediction_dim = self.dataset_config.get("n_classes")
        for batch in self.family.stream_batches(self.dataset_config):
            cache_batches.append(
                TrainingCacheBatch(
                    bins=self.quantize_batch(batch.x, cuts_cpu),
                    target_stats=batch.target_stats.astype(np.float32, copy=False),
                    sample_weight=None if batch.sample_weight is None else batch.sample_weight.astype(np.float32, copy=False),
                    state=np.empty((batch.target_stats.shape[0], prediction_dim), dtype=np.float32),
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

    def predict_tree_batch(self, tree: SingleTree, x: np.ndarray | None = None, bins: np.ndarray | None = None) -> np.ndarray:
        if (x is None) == (bins is None):
            raise ValueError("Provide exactly one of x or bins.")
        if x is not None:
            return tree.predict_batch(x, predict_method="cpu")
        pred = np.empty((bins.shape[0], tree.prediction_dim), dtype=np.float32)
        for i in range(bins.shape[0]):
            node = 0
            while tree.is_leaf_cpu[node] == 0:
                feature = tree.split_feature_cpu[node]
                threshold_bin = tree.split_bin_cpu[node]
                if bins[i, feature] <= threshold_bin:
                    node = tree.left_child_cpu[node]
                else:
                    node = tree.right_child_cpu[node]
            pred[i] = tree.leaf_value_cpu[node]
        return pred

    def predict_model_batch(self, model: AdditiveEnsemble, x: np.ndarray) -> np.ndarray:
        pred = np.repeat(model.base_state[None, :], x.shape[0], axis=0)
        for tree in model.trees:
            self.family.apply_update(pred, self.predict_tree_batch(tree, x=x), model.learning_rate)
        return self.family.predict_from_state(pred)

    def _leaf_value(self, target_stat_sum: np.ndarray, total_weight: float) -> np.ndarray:
        return target_stat_sum / (total_weight + self.tree_config.get("reg_lambda"))

    def _leaf_score(self, target_stat_sum: np.ndarray, total_weight: float) -> float:
        if total_weight <= 0.0:
            return -np.inf
        return float(np.dot(target_stat_sum, target_stat_sum) / (total_weight + self.tree_config.get("reg_lambda")))

    def fit_tree(self, training_cache: TrainingCache, cuts_cpu: np.ndarray, profile: bool, training_profile: dict | None, round_idx: int) -> SingleTree:
        tree = SingleTree(self.dataset_config.get("n_classes"))
        tree_profile = _start_profile(f"tree_growth_round_{round_idx + 1}") if profile else None

        while True:
            candidate_node_ids = tree.candidate_node_ids(self.tree_config)
            if not candidate_node_ids:
                break

            candidate_slot_of_node_cpu = np.full(len(tree.nodes), -1, dtype=np.int32)
            for slot, node_id in enumerate(candidate_node_ids):
                candidate_slot_of_node_cpu[node_id] = slot

            split_feature_cpu = np.array([node.split_feature for node in tree.nodes], dtype=np.int32)
            split_bin_cpu = np.array([node.split_bin for node in tree.nodes], dtype=np.int32)
            left_child_cpu = np.array([node.left_child for node in tree.nodes], dtype=np.int32)
            right_child_cpu = np.array([node.right_child for node in tree.nodes], dtype=np.int32)
            is_leaf_cpu = np.array([1 if node.is_leaf else 0 for node in tree.nodes], dtype=np.int8)

            n_slots = len(candidate_node_ids)
            n_features = self.dataset_config.get("n_features")
            n_bins = self.tree_config.get("max_bin")
            pred_dim = self.dataset_config.get("n_classes")
            hist_count = np.zeros((n_slots, n_features, n_bins), dtype=np.int32)
            hist_weight = np.zeros((n_slots, n_features, n_bins), dtype=np.float32)
            hist_sum = np.zeros((n_slots, n_features, n_bins, pred_dim), dtype=np.float32)

            for batch in training_cache:
                row_slot = np.empty((batch.bins.shape[0],), dtype=np.int32)
                if self.cpu_threads > 1:
                    route_rows_to_candidate_slots_cpu_parallel(
                        batch.bins,
                        split_feature_cpu,
                        split_bin_cpu,
                        left_child_cpu,
                        right_child_cpu,
                        is_leaf_cpu,
                        candidate_slot_of_node_cpu,
                        row_slot,
                    )
                else:
                    route_rows_to_candidate_slots_cpu(
                        batch.bins,
                        split_feature_cpu,
                        split_bin_cpu,
                        left_child_cpu,
                        right_child_cpu,
                        is_leaf_cpu,
                        candidate_slot_of_node_cpu,
                        row_slot,
                    )
                residual = self.family.preconditioned_target(batch.state, batch.target_stats).astype(np.float32, copy=False)
                if batch.sample_weight is None and self.cpu_threads > 1:
                    build_candidate_histograms_unweighted_cpu_parallel(
                        batch.bins, residual, row_slot, hist_count, hist_weight, hist_sum
                    )
                elif batch.sample_weight is None:
                    build_candidate_histograms_unweighted_cpu(batch.bins, residual, row_slot, hist_count, hist_weight, hist_sum)
                elif self.cpu_threads > 1:
                    build_candidate_histograms_weighted_cpu_parallel(
                        batch.bins, residual, batch.sample_weight, row_slot, hist_count, hist_weight, hist_sum
                    )
                else:
                    build_candidate_histograms_weighted_cpu(
                        batch.bins, residual, batch.sample_weight, row_slot, hist_count, hist_weight, hist_sum
                    )
                if tree_profile is not None:
                    _update_profile(tree_profile)
                if training_profile is not None:
                    _update_profile(training_profile)

            slot_parent_count = np.zeros((n_slots,), dtype=np.int32)
            slot_parent_weight = np.zeros((n_slots,), dtype=np.float32)
            slot_parent_sum = np.zeros((n_slots, pred_dim), dtype=np.float32)
            slot_best_gain = np.full((n_slots,), -1.0e30, dtype=np.float32)
            slot_best_feature = np.full((n_slots,), -1, dtype=np.int32)
            slot_best_bin = np.full((n_slots,), -1, dtype=np.int32)
            slot_best_left_count = np.zeros((n_slots,), dtype=np.int32)
            slot_best_right_count = np.zeros((n_slots,), dtype=np.int32)
            slot_best_left_weight = np.zeros((n_slots,), dtype=np.float32)
            slot_best_right_weight = np.zeros((n_slots,), dtype=np.float32)
            slot_best_left_sum = np.zeros((n_slots, pred_dim), dtype=np.float32)
            slot_best_right_sum = np.zeros((n_slots, pred_dim), dtype=np.float32)

            if self.cpu_threads > 1:
                evaluate_slot_bests_cpu_parallel(
                    hist_count,
                    hist_weight,
                    hist_sum,
                    self.tree_config.get("min_samples_leaf"),
                    self.tree_config.get("reg_lambda"),
                    slot_parent_count,
                    slot_parent_weight,
                    slot_parent_sum,
                    slot_best_gain,
                    slot_best_feature,
                    slot_best_bin,
                    slot_best_left_count,
                    slot_best_right_count,
                    slot_best_left_weight,
                    slot_best_right_weight,
                    slot_best_left_sum,
                    slot_best_right_sum,
                )
            else:
                evaluate_slot_bests_cpu(
                    hist_count,
                    hist_weight,
                    hist_sum,
                    self.tree_config.get("min_samples_leaf"),
                    self.tree_config.get("reg_lambda"),
                    slot_parent_count,
                    slot_parent_weight,
                    slot_parent_sum,
                    slot_best_gain,
                    slot_best_feature,
                    slot_best_bin,
                    slot_best_left_count,
                    slot_best_right_count,
                    slot_best_left_weight,
                    slot_best_right_weight,
                    slot_best_left_sum,
                    slot_best_right_sum,
                )

            if tree_profile is not None:
                _update_profile(tree_profile)
            if training_profile is not None:
                _update_profile(training_profile)

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
        return tree

    def update_training_cache(self, training_cache: TrainingCache, tree: SingleTree, learning_rate: float, profile: bool, training_profile: dict | None, round_idx: int):
        update_profile = _start_profile(f"cache_update_round_{round_idx + 1}") if profile else None
        for batch in training_cache:
            self.family.apply_update(batch.state, self.predict_tree_batch(tree, bins=batch.bins), learning_rate)
            if update_profile is not None:
                _update_profile(update_profile)
            if training_profile is not None:
                _update_profile(training_profile)
        if update_profile is not None:
            _finish_profile(update_profile)
            print_profile(update_profile)

    def evaluate_cached_training_stream(self, training_cache: TrainingCache, profile: bool) -> tuple[float, float]:
        evaluation_profile = _start_profile("evaluation") if profile else None
        train_metric, mean_sum_prob, _ = training_cache.monitoring_loss(self.family)
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
            pred_cpu = model.predict_batch(
                batch.x,
                predict_method="cpu",
                predict_from_state=self.family.predict_from_state,
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
        cuts_cpu = self.build_cuts(provider_kwargs)
        training_profile = _start_profile("training") if profile else None
        training_cache = self.build_training_cache(cuts_cpu, profile, training_profile)
        base_state = self.family.base_state(training_cache.target_stat_mean())
        training_cache.initialize_states(base_state)
        model = AdditiveEnsemble(
            prediction_dim=self.dataset_config.get("n_classes"),
            base_state=base_state,
            learning_rate=self.training_config.get("learning_rate"),
        )

        loss_history = []
        initial_loss, _, _ = training_cache.monitoring_loss(self.family)
        loss_history.append(initial_loss)
        print(f"Initial train {self.family.monitor_name}: {initial_loss:.6f}")

        for round_idx in range(self.training_config.get("n_boost_rounds")):
            tree = self.fit_tree(training_cache, cuts_cpu, profile, training_profile, round_idx)
            model.add_tree(tree)
            self.update_training_cache(training_cache, tree, model.learning_rate, profile, training_profile, round_idx)
            round_loss, _, _ = training_cache.monitoring_loss(self.family)
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
