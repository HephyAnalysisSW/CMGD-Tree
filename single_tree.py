from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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

    def mse_from_predictions(
        self,
        pred_cpu: np.ndarray,
        cls_cpu: np.ndarray,
        sample_weight_cpu: np.ndarray | None = None,
    ) -> tuple[float, float]:
        pred_sq = np.sum(pred_cpu * pred_cpu, axis=1)
        target_prob = pred_cpu[np.arange(pred_cpu.shape[0]), cls_cpu]
        per_row = 1.0 - 2.0 * target_prob + pred_sq
        if sample_weight_cpu is None:
            return float(np.sum(per_row)), float(pred_cpu.shape[0])
        return float(np.sum(sample_weight_cpu * per_row)), float(np.sum(sample_weight_cpu))


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

    def predict_batch(
        self,
        x: np.ndarray,
        predict_method: str = "cpu",
        gpu_predictor=None,
    ) -> np.ndarray:
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
