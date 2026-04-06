from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _predict_forest_numba(
    x: np.ndarray,
    base_state: np.ndarray,
    tree_offsets: np.ndarray,
    split_feature: np.ndarray,
    split_threshold: np.ndarray,
    left_child: np.ndarray,
    right_child: np.ndarray,
    is_leaf: np.ndarray,
    leaf_value: np.ndarray,
    pred_out: np.ndarray,
):
    n_rows = x.shape[0]
    pred_dim = base_state.shape[0]
    n_trees = tree_offsets.shape[0] - 1
    for i in range(n_rows):
        for c in range(pred_dim):
            pred_out[i, c] = base_state[c]
        for tree_idx in range(n_trees):
            node = tree_offsets[tree_idx]
            while is_leaf[node] == 0:
                feature = split_feature[node]
                threshold = split_threshold[node]
                if x[i, feature] <= threshold:
                    node = left_child[node]
                else:
                    node = right_child[node]
            for c in range(pred_dim):
                pred_out[i, c] += leaf_value[node, c]


@njit(cache=True, parallel=True)
def _predict_forest_numba_parallel(
    x: np.ndarray,
    base_state: np.ndarray,
    tree_offsets: np.ndarray,
    split_feature: np.ndarray,
    split_threshold: np.ndarray,
    left_child: np.ndarray,
    right_child: np.ndarray,
    is_leaf: np.ndarray,
    leaf_value: np.ndarray,
    pred_out: np.ndarray,
):
    n_rows = x.shape[0]
    pred_dim = base_state.shape[0]
    n_trees = tree_offsets.shape[0] - 1
    for i in prange(n_rows):
        for c in range(pred_dim):
            pred_out[i, c] = base_state[c]
        for tree_idx in range(n_trees):
            node = tree_offsets[tree_idx]
            while is_leaf[node] == 0:
                feature = split_feature[node]
                threshold = split_threshold[node]
                if x[i, feature] <= threshold:
                    node = left_child[node]
                else:
                    node = right_child[node]
            for c in range(pred_dim):
                pred_out[i, c] += leaf_value[node, c]


@njit(cache=True)
def _predict_forest_numba_dim4(
    x: np.ndarray,
    base_state: np.ndarray,
    tree_offsets: np.ndarray,
    split_feature: np.ndarray,
    split_threshold: np.ndarray,
    left_child: np.ndarray,
    right_child: np.ndarray,
    is_leaf: np.ndarray,
    leaf_value: np.ndarray,
    pred_out: np.ndarray,
):
    n_rows = x.shape[0]
    n_trees = tree_offsets.shape[0] - 1
    base0 = base_state[0]
    base1 = base_state[1]
    base2 = base_state[2]
    base3 = base_state[3]
    for i in range(n_rows):
        pred0 = base0
        pred1 = base1
        pred2 = base2
        pred3 = base3
        for tree_idx in range(n_trees):
            node = tree_offsets[tree_idx]
            while is_leaf[node] == 0:
                feature = split_feature[node]
                if x[i, feature] <= split_threshold[node]:
                    node = left_child[node]
                else:
                    node = right_child[node]
            pred0 += leaf_value[node, 0]
            pred1 += leaf_value[node, 1]
            pred2 += leaf_value[node, 2]
            pred3 += leaf_value[node, 3]
        pred_out[i, 0] = pred0
        pred_out[i, 1] = pred1
        pred_out[i, 2] = pred2
        pred_out[i, 3] = pred3


@njit(cache=True, parallel=True)
def _predict_forest_numba_dim4_parallel(
    x: np.ndarray,
    base_state: np.ndarray,
    tree_offsets: np.ndarray,
    split_feature: np.ndarray,
    split_threshold: np.ndarray,
    left_child: np.ndarray,
    right_child: np.ndarray,
    is_leaf: np.ndarray,
    leaf_value: np.ndarray,
    pred_out: np.ndarray,
):
    n_rows = x.shape[0]
    n_trees = tree_offsets.shape[0] - 1
    base0 = base_state[0]
    base1 = base_state[1]
    base2 = base_state[2]
    base3 = base_state[3]
    for i in prange(n_rows):
        pred0 = base0
        pred1 = base1
        pred2 = base2
        pred3 = base3
        for tree_idx in range(n_trees):
            node = tree_offsets[tree_idx]
            while is_leaf[node] == 0:
                feature = split_feature[node]
                if x[i, feature] <= split_threshold[node]:
                    node = left_child[node]
                else:
                    node = right_child[node]
            pred0 += leaf_value[node, 0]
            pred1 += leaf_value[node, 1]
            pred2 += leaf_value[node, 2]
            pred3 += leaf_value[node, 3]
        pred_out[i, 0] = pred0
        pred_out[i, 1] = pred1
        pred_out[i, 2] = pred2
        pred_out[i, 3] = pred3


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
    def __init__(self, prediction_dim: int):
        self.prediction_dim = prediction_dim
        self.nodes: list[Node] = [Node(node_id=0, depth=0)]
        self.n_leaves = 1
        self.next_node_id = 1
        self.root_score: float | None = None
        self.leaf_value_cpu = None
        self.split_feature_cpu = None
        self.split_bin_cpu = None
        self.split_threshold_cpu = None
        self.left_child_cpu = None
        self.right_child_cpu = None
        self.is_leaf_cpu = None
        self.leaf_paths = None

    def candidate_node_ids(self, tree_config: dict) -> list[int]:
        node_ids = [
            node.node_id
            for node in self.nodes
            if node.is_leaf and node.expandable and node.depth < tree_config.get("max_depth")
        ]
        if tree_config.get("grow_policy") == "depthwise" and node_ids:
            frontier_depth = min(self.nodes[node_id].depth for node_id in node_ids)
            node_ids = [node_id for node_id in node_ids if self.nodes[node_id].depth == frontier_depth]
        return node_ids

    def finalize_prediction_state(self):
        self.leaf_value_cpu = np.zeros((len(self.nodes), self.prediction_dim), dtype=np.float32)
        self.split_feature_cpu = np.array([node.split_feature for node in self.nodes], dtype=np.int32)
        self.split_bin_cpu = np.array([node.split_bin for node in self.nodes], dtype=np.int32)
        self.split_threshold_cpu = np.array([node.split_threshold for node in self.nodes], dtype=np.float32)
        self.left_child_cpu = np.array([node.left_child for node in self.nodes], dtype=np.int32)
        self.right_child_cpu = np.array([node.right_child for node in self.nodes], dtype=np.int32)
        self.is_leaf_cpu = np.array([1 if node.is_leaf else 0 for node in self.nodes], dtype=np.int8)
        for node in self.nodes:
            if node.value is not None:
                self.leaf_value_cpu[node.node_id] = node.value
        self.leaf_paths = []
        self._collect_leaf_paths(0, [])

    def _collect_leaf_paths(self, node_id: int, path: list[tuple[int, float, bool]]):
        if self.is_leaf_cpu[node_id]:
            self.leaf_paths.append(
                (
                    node_id,
                    np.array([entry[0] for entry in path], dtype=np.int32),
                    np.array([entry[1] for entry in path], dtype=np.float32),
                    np.array([entry[2] for entry in path], dtype=np.bool_),
                )
            )
            return
        feature = int(self.split_feature_cpu[node_id])
        threshold = float(self.split_threshold_cpu[node_id])
        self._collect_leaf_paths(self.left_child_cpu[node_id], path + [(feature, threshold, True)])
        self._collect_leaf_paths(self.right_child_cpu[node_id], path + [(feature, threshold, False)])

    def predict_batch_cpu_index(self, x: np.ndarray) -> np.ndarray:
        pred = np.empty((x.shape[0], self.prediction_dim), dtype=np.float32)
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

    def predict_batch_cpu_leaf_mask(self, x: np.ndarray) -> np.ndarray:
        pred = np.empty((x.shape[0], self.prediction_dim), dtype=np.float32)
        for leaf_id, features, thresholds, go_left in self.leaf_paths:
            mask = np.ones(x.shape[0], dtype=np.bool_)
            for feature, threshold, left_flag in zip(features, thresholds, go_left):
                if left_flag:
                    mask &= x[:, feature] <= threshold
                else:
                    mask &= x[:, feature] > threshold
            pred[mask] = self.leaf_value_cpu[leaf_id]
        return pred

    def predict_batch_cpu(self, x: np.ndarray, cpu_predictor: str = "index") -> np.ndarray:
        if cpu_predictor == "leaf_mask":
            return self.predict_batch_cpu_leaf_mask(x)
        return self.predict_batch_cpu_index(x)

    def predict_batch(self, x: np.ndarray, predict_method: str = "cpu", gpu_predictor=None, cpu_predictor: str = "index") -> np.ndarray:
        if predict_method == "gpu":
            if gpu_predictor is None:
                raise ValueError("gpu_predictor is required for GPU prediction.")
            return gpu_predictor(x)
        return self.predict_batch_cpu(x, cpu_predictor=cpu_predictor)

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


class AdditiveEnsemble:
    def __init__(self, prediction_dim: int, base_state: np.ndarray, learning_rate: float):
        self.prediction_dim = prediction_dim
        self.base_state = np.asarray(base_state, dtype=np.float32)
        self.learning_rate = float(learning_rate)
        self.trees: list[SingleTree] = []
        self._numba_tree_offsets = None
        self._numba_split_feature = None
        self._numba_split_threshold = None
        self._numba_left_child = None
        self._numba_right_child = None
        self._numba_is_leaf = None
        self._numba_leaf_value = None

    def add_tree(self, tree: SingleTree):
        self.trees.append(tree)
        self._numba_tree_offsets = None

    def _ensure_numba_state(self):
        if self._numba_tree_offsets is not None:
            return
        tree_offsets = [0]
        split_feature = []
        split_threshold = []
        left_child = []
        right_child = []
        is_leaf = []
        leaf_value = []
        offset = 0
        for tree in self.trees:
            n_nodes = len(tree.nodes)
            tree_offsets.append(offset + n_nodes)
            split_feature.extend(tree.split_feature_cpu.tolist())
            split_threshold.extend(tree.split_threshold_cpu.tolist())
            left_child.extend([(child + offset) if child >= 0 else -1 for child in tree.left_child_cpu.tolist()])
            right_child.extend([(child + offset) if child >= 0 else -1 for child in tree.right_child_cpu.tolist()])
            is_leaf.extend(tree.is_leaf_cpu.tolist())
            leaf_value.append(tree.leaf_value_cpu)
            offset += n_nodes
        self._numba_tree_offsets = np.asarray(tree_offsets, dtype=np.int32)
        self._numba_split_feature = np.asarray(split_feature, dtype=np.int32)
        self._numba_split_threshold = np.asarray(split_threshold, dtype=np.float32)
        self._numba_left_child = np.asarray(left_child, dtype=np.int32)
        self._numba_right_child = np.asarray(right_child, dtype=np.int32)
        self._numba_is_leaf = np.asarray(is_leaf, dtype=np.int8)
        self._numba_leaf_value = (self.learning_rate * np.concatenate(leaf_value, axis=0)).astype(np.float32, copy=False)

    def _predict_batch_cpu_numba(self, x: np.ndarray, cpu_predictor: str) -> np.ndarray:
        self._ensure_numba_state()
        pred = np.empty((x.shape[0], self.prediction_dim), dtype=np.float32)
        if self.prediction_dim == 4:
            kernel = _predict_forest_numba_dim4_parallel if cpu_predictor == "numba_parallel" else _predict_forest_numba_dim4
        else:
            kernel = _predict_forest_numba_parallel if cpu_predictor == "numba_parallel" else _predict_forest_numba
        kernel(
            x,
            self.base_state,
            self._numba_tree_offsets,
            self._numba_split_feature,
            self._numba_split_threshold,
            self._numba_left_child,
            self._numba_right_child,
            self._numba_is_leaf,
            self._numba_leaf_value,
            pred,
        )
        return pred

    def predict_batch_cpu(self, x: np.ndarray, predict_from_state, cpu_predictor: str = "index") -> np.ndarray:
        if cpu_predictor in {"numba", "numba_parallel"}:
            pred = self._predict_batch_cpu_numba(x, cpu_predictor)
            return predict_from_state(pred)
        pred = np.repeat(self.base_state[None, :], x.shape[0], axis=0)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict_batch_cpu(x, cpu_predictor=cpu_predictor)
        return predict_from_state(pred)

    def predict_batch(self, x: np.ndarray, predict_method: str = "cpu", gpu_predictor=None, predict_from_state=None, cpu_predictor: str = "index") -> np.ndarray:
        if predict_method == "gpu":
            if gpu_predictor is None:
                raise ValueError("gpu_predictor is required for GPU prediction.")
            pred = gpu_predictor(x)
        else:
            pred = self.predict_batch_cpu(x, predict_from_state, cpu_predictor=cpu_predictor)
        if predict_from_state is None or predict_method != "gpu":
            return pred
        return predict_from_state(pred)
