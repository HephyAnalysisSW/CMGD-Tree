from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
        self.root_weight: float | None = None
        self.leaf_value_cpu = None
        self.split_feature_cpu = None
        self.split_bin_cpu = None
        self.split_threshold_cpu = None
        self.left_child_cpu = None
        self.right_child_cpu = None
        self.is_leaf_cpu = None

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

    def predict_batch_cpu(self, x: np.ndarray) -> np.ndarray:
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

    def predict_batch(self, x: np.ndarray, predict_method: str = "cpu", gpu_predictor=None) -> np.ndarray:
        if predict_method == "gpu":
            if gpu_predictor is None:
                raise ValueError("gpu_predictor is required for GPU prediction.")
            return gpu_predictor(x)
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


class AdditiveEnsemble:
    def __init__(self, prediction_dim: int, base_prediction: np.ndarray, learning_rate: float):
        self.prediction_dim = prediction_dim
        self.base_prediction = np.asarray(base_prediction, dtype=np.float32)
        self.learning_rate = float(learning_rate)
        self.trees: list[SingleTree] = []

    def add_tree(self, tree: SingleTree):
        self.trees.append(tree)

    def predict_batch_cpu(self, x: np.ndarray, project_prediction) -> np.ndarray:
        pred = np.repeat(self.base_prediction[None, :], x.shape[0], axis=0)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict_batch_cpu(x)
        return project_prediction(pred)

    def predict_batch(self, x: np.ndarray, predict_method: str = "cpu", gpu_predictor=None, project_prediction=None) -> np.ndarray:
        if predict_method == "gpu":
            if gpu_predictor is None:
                raise ValueError("gpu_predictor is required for GPU prediction.")
            pred = gpu_predictor(x)
        else:
            pred = self.predict_batch_cpu(x, project_prediction)
        if project_prediction is None or predict_method != "gpu":
            return pred
        return project_prediction(pred)
