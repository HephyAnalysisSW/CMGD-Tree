from __future__ import annotations

import argparse
import ast
import os
import time

import numpy as np
import xgboost as xgb

try:
    import psutil
except Exception:
    psutil = None

try:
    import resource
except Exception:
    resource = None


MODEL_CONFIG = {
    "n_estimators": 100,
    "max_depth": 2,
    "learning_rate": 1.0,
    "tree_method": "hist",
    "device": "cpu",
    "multi_strategy": "multi_output_tree",
    "n_jobs": -1,
}

TRAIN_DATA_CONFIG = {
    "n_features": 4,
    "n_classes": 4,
    "batch_size": 65536,
    "n_batches": 12,
    "seed": 0,
    "feature_offset_scale": 2.5,
    "feature_noise": 1.0,
}

FRESH_INFERENCE_CONFIG = {
    "fresh_batch_size": 262144,
    "fresh_n_batches": 64,
}

CONFIG_GROUPS = {
    "model": MODEL_CONFIG,
    "train_data": TRAIN_DATA_CONFIG,
    "fresh_inference": FRESH_INFERENCE_CONFIG,
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
    parser = argparse.ArgumentParser(description="Standalone XGBoost comparison for the normal toy setup.")
    parser.add_argument("--modify", nargs="*", default=[])
    args, _unknown = parser.parse_known_args()

    all_config_keys = {}
    for group_name, group in CONFIG_GROUPS.items():
        for key in group:
            if key in all_config_keys:
                raise ValueError(f"Duplicate config key '{key}' in '{group_name}' and '{all_config_keys[key]}'.")
            all_config_keys[key] = group_name

    if len(args.modify) % 2 != 0:
        raise ValueError("--modify expects key value pairs.")

    for key, value_text in zip(args.modify[0::2], args.modify[1::2]):
        if key not in all_config_keys:
            raise KeyError(f"Unknown config key '{key}'.")
        group = CONFIG_GROUPS[all_config_keys[key]]
        group[key] = _cast_override(value_text, group.get(key))
    return args


def _rss_bytes():
    if psutil is not None:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    if resource is not None:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if os.uname().sysname.lower() == "darwin":
            return rss
        return rss * 1024
    return 0


class GaussianClassToyStream:
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        batch_size: int,
        n_batches: int,
        feature_offset_scale: float = 2.5,
        feature_noise: float = 1.0,
        seed: int = 0,
        dtype=np.float32,
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.feature_offset_scale = feature_offset_scale
        self.feature_noise = feature_noise
        self.seed = seed
        self.dtype = dtype
        self._rng = np.random.default_rng(self.seed)
        self._class_means = np.zeros((self.n_classes, self.n_features), dtype=self.dtype)
        self._class_targets = np.eye(self.n_classes, dtype=self.dtype)
        if self.n_features == 1 and self.n_classes == 2:
            self._class_means[0, 0] = 0.0
            self._class_means[1, 0] = self.feature_offset_scale
        else:
            n_anchor = min(self.n_classes, self.n_features)
            for c in range(self.n_classes):
                self._class_means[c, c % n_anchor] = self.feature_offset_scale
                if self.n_features > 1:
                    self._class_means[c, (c + 1) % self.n_features] = -0.5 * self.feature_offset_scale

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self.next_batch()

    def next_batch(self):
        cls = self._rng.integers(0, self.n_classes, size=self.batch_size, endpoint=False)
        x = self._rng.normal(
            loc=0.0,
            scale=self.feature_noise,
            size=(self.batch_size, self.n_features),
        ).astype(self.dtype)
        x += self._class_means[cls]
        y = self._class_targets[cls]
        return x, y


def make_train_arrays():
    stream = GaussianClassToyStream(
        n_features=TRAIN_DATA_CONFIG["n_features"],
        n_classes=TRAIN_DATA_CONFIG["n_classes"],
        batch_size=TRAIN_DATA_CONFIG["batch_size"],
        n_batches=TRAIN_DATA_CONFIG["n_batches"],
        feature_offset_scale=TRAIN_DATA_CONFIG["feature_offset_scale"],
        feature_noise=TRAIN_DATA_CONFIG["feature_noise"],
        seed=TRAIN_DATA_CONFIG["seed"],
    )
    x_batches = []
    y_batches = []
    for x, y in stream:
        x_batches.append(x)
        y_batches.append(y)
    return np.concatenate(x_batches, axis=0), np.concatenate(y_batches, axis=0)


def mse_from_predictions(pred: np.ndarray, y: np.ndarray) -> float:
    per_row = np.sum((pred - y) ** 2, axis=1)
    return float(np.mean(per_row))


def main():
    _parse_args()
    print("Model config:", MODEL_CONFIG)
    print("Train data config:", TRAIN_DATA_CONFIG)
    print("Fresh inference config:", FRESH_INFERENCE_CONFIG)

    train_x, train_y = make_train_arrays()
    print("Train shape:", train_x.shape, train_y.shape)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=MODEL_CONFIG["n_estimators"],
        max_depth=MODEL_CONFIG["max_depth"],
        learning_rate=MODEL_CONFIG["learning_rate"],
        tree_method=MODEL_CONFIG["tree_method"],
        device=MODEL_CONFIG["device"],
        multi_strategy=MODEL_CONFIG["multi_strategy"],
        n_jobs=MODEL_CONFIG["n_jobs"],
        verbosity=1,
    )

    rss_before_train = _rss_bytes()
    train_wall_start = time.perf_counter()
    train_cpu_start = time.process_time()
    model.fit(train_x, train_y)
    train_wall = time.perf_counter() - train_wall_start
    train_cpu = time.process_time() - train_cpu_start
    rss_after_train = _rss_bytes()
    booster = model.get_booster()

    pred_train = model.predict(train_x)
    train_mse = mse_from_predictions(pred_train, train_y)

    fresh_stream = GaussianClassToyStream(
        n_features=TRAIN_DATA_CONFIG["n_features"],
        n_classes=TRAIN_DATA_CONFIG["n_classes"],
        batch_size=FRESH_INFERENCE_CONFIG["fresh_batch_size"],
        n_batches=FRESH_INFERENCE_CONFIG["fresh_n_batches"],
        feature_offset_scale=TRAIN_DATA_CONFIG["feature_offset_scale"],
        feature_noise=TRAIN_DATA_CONFIG["feature_noise"],
        seed=TRAIN_DATA_CONFIG["seed"],
    )
    fresh_wall_start = time.perf_counter()
    fresh_cpu_start = time.process_time()
    fresh_sum = 0.0
    fresh_rows = 0
    for x_batch, _y_batch in fresh_stream:
        pred = booster.inplace_predict(x_batch)
        fresh_sum += float(np.sum(pred))
        fresh_rows += pred.shape[0]
    fresh_wall = time.perf_counter() - fresh_wall_start
    fresh_cpu = time.process_time() - fresh_cpu_start

    print()
    print("[xgboost:training]")
    print(f"  wall_s={train_wall:.3f} cpu_s={train_cpu:.3f} cpu_pct_est={100.0 * train_cpu / max(train_wall, 1e-9):.1f}")
    print(f"  rss_mb=start={rss_before_train / 2**20:.1f} end={rss_after_train / 2**20:.1f}")
    print(f"  train_mse={train_mse:.6f}")
    print()
    print("[xgboost:fresh_inference]")
    print(f"  wall_s={fresh_wall:.3f} cpu_s={fresh_cpu:.3f} cpu_pct_est={100.0 * fresh_cpu / max(fresh_wall, 1e-9):.1f}")
    print(f"  mean_sum_prediction={fresh_sum / max(fresh_rows, 1):.6f}")


if __name__ == "__main__":
    main()
