# CMGD-Tree

CMGD-Tree is a small sandbox for streamed histogram-tree boosting with:

- GPU training
- CPU or GPU prediction
- family-specific losses and target statistics
- MGD and NGD examples
- toy data providers for quick experiments

The main command is:

```bash
python fit_single_tree_hist_demo.py
```

Latest writeup PDF:

- [GitHub Pages PDF](https://hephyanalysissw.github.io/CMGD-Tree/writeup.pdf)
- [Actions workflow page](https://github.com/HephyAnalysisSW/CMGD-Tree/actions/workflows/build-writeup.yml)

## What You Can Run

If you just want to see something work:

Run the default normal example:

```bash
python fit_single_tree_hist_demo.py
```

Run the same example and print the fitted trees:

```bash
python fit_single_tree_hist_demo.py --print-trees
```

Run the same example and generate plots:

```bash
python fit_single_tree_hist_demo.py --plot
```

Run the same example with both plots and printed trees:

```bash
python fit_single_tree_hist_demo.py --plot --print-trees
```

Write the output to a log file:

```bash
python fit_single_tree_hist_demo.py --plot > ~/logs/cmgtree-demo.log 2>&1
```

## Families and Example Commands

The current implemented families are:

- `normal_identity`
- `poisson`
- `poisson_ngd`
- `gamma`
- `negative_binomial`
- `heteroskedastic_normal`
- `heteroskedastic_normal_ngd`

Useful example commands:

Normal identity example.
This is the default multiclass-like mean fit.

```bash
python fit_single_tree_hist_demo.py \
  --plot \
  --print-trees \
  --modify family normal_identity
```

Poisson MGD example.
This learns positive mean count predictions.

```bash
python fit_single_tree_hist_demo.py \
  --plot \
  --print-trees \
  --modify family poisson
```

Poisson NGD example.
This uses the same Poisson toy, but with family-side Fisher preconditioning.

```bash
python fit_single_tree_hist_demo.py \
  --plot \
  --print-trees \
  --modify family poisson_ngd
```

Gamma MGD example.
This is a positive continuous target with fixed shape.

```bash
python fit_single_tree_hist_demo.py \
  --plot \
  --print-trees \
  --modify family gamma
```

Negative binomial MGD example.
This is an overdispersed count example with fixed dispersion.

```bash
python fit_single_tree_hist_demo.py \
  --plot \
  --print-trees \
  --modify family negative_binomial
```

Heteroskedastic normal MGD example.
This predicts first and second moments and derives variance from them.

```bash
python fit_single_tree_hist_demo.py \
  --plot \
  --print-trees \
  --modify family heteroskedastic_normal
```

Heteroskedastic normal NGD example.
This uses the same toy, but with an NGD-style family update.

```bash
python fit_single_tree_hist_demo.py \
  --plot \
  --print-trees \
  --modify family heteroskedastic_normal_ngd
```

## Command-Line Options

Top-level flags:

- `--modify key value ...`
  Override any config entry from the tree, dataset, or training groups.
- `--profile`
  Print timing and memory summaries for training and evaluation.
- `--plot`
  Write plots to `./plots/<plot_training_id>/`.
- `--print-trees`
  Print every fitted tree after the run.
- `--full-output`
  Compatibility alias for `--plot --print-trees`.

Example:

```bash
python fit_single_tree_hist_demo.py \
  --plot \
  --print-trees \
  --modify family gamma max_depth 4 max_leaves 16 n_features 8
```

## Config Overrides

The script has three config groups:

```python
TREE_CONFIG = {
    "max_bin": 64,
    "cut_sample_rows": 200000,
    "grow_policy": "depthwise",
    "max_depth": 2,
    "max_leaves": 4,
    "min_samples_leaf": 512,
    "min_split_loss": 1e-3,
    "reg_lambda": 0.0,
    "family": "normal_identity",
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
    "plot_mode": "all",
    "threads_per_block": 128,
    "training_backend": "auto",
    "cpu_threads": 0,
    "predict_method": "cpu",
    "cpu_predictor": "numba_parallel",
    "n_boost_rounds": 2,
    "learning_rate": 1.0,
    "fresh_inference_batch_size": None,
    "fresh_inference_n_batches": None,
}
```

### Tree Hyperparameters

- `max_bin`
  Number of histogram bins per feature.
  Larger values make split search finer, but cost more memory and compute.

- `cut_sample_rows`
  Maximum number of streamed rows used to estimate the feature cuts.
  This does not cap training rows.
  It only affects how the bin boundaries are chosen.

- `grow_policy`
  Tree growth strategy.
  `depthwise` expands level by level.
  `lossguide` expands the currently best leaves first.

- `max_depth`
  Maximum tree depth.

- `max_leaves`
  Maximum number of leaves.
  This can be more restrictive than `max_depth`.

- `min_samples_leaf`
  Minimum effective sample count required in a leaf.
  Prevents very small leaves.

- `min_split_loss`
  Minimum gain required to keep a split.
  Larger values make the tree more conservative.

- `reg_lambda`
  L2-style regularization term used in split scoring / leaf scoring.

- `family`
  Which probabilistic example family to use.

- `class_weights`
  Optional per-target weighting vector.
  Mainly useful for multi-output normal-style fits.

### Dataset Hyperparameters

- `n_features`
  Input feature dimension.

- `n_classes`
  Output target-stat dimension.
  For some examples this is literally the number of outputs.
  For heteroskedastic normal it is fixed to `2`, representing `[y, y^2]`.

- `batch_size`
  Number of streamed events per batch.

- `n_batches`
  Number of streamed batches.
  Total training events are `batch_size * n_batches`.

- `seed`
  Random seed for the toy provider.

- `feature_offset_scale`
  Provider-side offset used to make the toy data less degenerate.

- `feature_noise`
  Provider-side feature scale.

### Training Hyperparameters

- `plot_training_id`
  Output directory name under `./plots/`.

- `plot_bins`
  Number of bins used in diagnostic plots.

- `plot_mode`
  Provider-specific plotting mode selector.
  In most cases, leave this at `all`.

- `threads_per_block`
  CUDA kernel launch block size for the GPU trainer.

- `training_backend`
  `auto`, `gpu`, or `cpu`.
  `auto` uses the GPU when available and otherwise falls back to CPU.

- `cpu_threads`
  Number of CPU threads to use for CPU prediction and CPU-side update work.
  `0` resolves to the default thread policy.

- `predict_method`
  `cpu` or `gpu`.
  This controls prediction and cache-update prediction, not the tree-fitting backend.

- `cpu_predictor`
  CPU prediction implementation.
  Choices:
  - `index`
  - `leaf_mask`
  - `numba`
  - `numba_parallel`

- `n_boost_rounds`
  Number of boosting iterations.

- `learning_rate`
  Shrinkage factor applied to each fitted tree.

- `fresh_inference_batch_size`
  Optional batch size for the separate fresh-inference benchmark path in profiling mode.

- `fresh_inference_n_batches`
  Optional number of batches for the separate fresh-inference benchmark path in profiling mode.

## Example Defaults

Some families come with example-owned defaults.
These are applied only when you do not override them explicitly.

Current example defaults:

- `heteroskedastic_normal`
  - `n_features=2`
  - `n_classes=2`
  - `n_batches=24`
  - `max_depth=2`
  - `max_leaves=4`
  - `n_boost_rounds=50`
  - `learning_rate=0.2`

- `gamma`
  - `n_features=4`
  - `n_classes=4`
  - `max_depth=3`
  - `max_leaves=8`
  - `n_boost_rounds=50`
  - `learning_rate=0.2`

- `negative_binomial`
  - `n_features=4`
  - `n_classes=4`
  - `max_depth=3`
  - `max_leaves=8`
  - `n_boost_rounds=50`
  - `learning_rate=0.2`

## Prediction Modes

Prediction is an important runtime choice.

`predict_method=gpu`
- keeps prediction on the GPU
- usually best when you are already on the GPU and the batches are large
- useful when cache updates should stay on-device during GPU training

`predict_method=cpu`
- uses the CPU predictors in [single_tree.py](/home/rschoefbeck/CMGD-Tree/single_tree.py)
- often convenient for small runs and inspection
- useful when you want to compare CPU inference implementations

CPU predictor choices:

- `index`
  Simple index-set traversal.
- `leaf_mask`
  Leaf-mask style NumPy predictor.
- `numba`
  Compiled single-core CPU predictor.
- `numba_parallel`
  Compiled multi-core CPU predictor.

Example:

Run GPU training with GPU prediction:

```bash
python fit_single_tree_hist_demo.py \
  --modify training_backend gpu predict_method gpu
```

Run GPU training with fast multi-core CPU prediction:

```bash
python fit_single_tree_hist_demo.py \
  --modify training_backend gpu predict_method cpu cpu_predictor numba_parallel cpu_threads 8
```

Run everything on CPU:

```bash
python fit_single_tree_hist_demo.py \
  --modify training_backend cpu predict_method cpu cpu_predictor numba_parallel cpu_threads 8
```

## Plotting

Plotting is off by default.

When `--plot` is enabled, plots are written under:

```bash
./plots/<plot_training_id>/
```

The actual plots depend on the provider:

- `normal_identity`
  feature-density plots and feature-target mean overlays
- `poisson`
  feature-density plots and feature-target mean overlays
- `gamma`
  feature-density plots and feature-target mean overlays
- `negative_binomial`
  feature-density plots and feature-target mean overlays
- `heteroskedastic_normal`
  observed/predicted mean and observed/predicted variance overlays

For smaller exploratory runs:

```bash
python fit_single_tree_hist_demo.py \
  --plot \
  --modify n_features 4 n_classes 4 plot_training_id small_demo
```

## MGD and NGD

The code is organized around additive boosting of tree outputs.

### MGD

In the MGD examples, the family supplies target statistics `T(y)`, a model state, and a dual prediction.
At round `t`, the tree is fit to the residual-style pseudo-response

```text
R_t(x, y) = T(y) - eta_t^*(x)
```

and the model is updated additively with a learning rate:

```text
state_{t+1}(x) = state_t(x) + alpha * f_t(x)
```

The histogram tree learner itself is generic: it just fits the supplied vector pseudo-response.

### NGD

In NGD, the trainer still fits a tree to a supplied pseudo-response, but the family changes what that target is.

Instead of the plain residual, the family can supply a Fisher-preconditioned target

```text
U_t(x, y) = G(state_t(x))^{-1} (T(y) - eta_t^*(x))
```

where `G` is the Fisher information in the chosen coordinate system.

So algorithmically:

- MGD is the identity / unpreconditioned case
- NGD is the family-preconditioned case

This is why both styles can share the same tree trainer.

## Profiling

If you want a timing and memory summary:

```bash
python fit_single_tree_hist_demo.py --profile
```

This reports the built-in training and evaluation profiling information.

## Benchmarks

Benchmark utilities live under [benchmarks/](/home/rschoefbeck/CMGD-Tree/benchmarks).
See [benchmarks/README.md](/home/rschoefbeck/CMGD-Tree/benchmarks/README.md) for how to run them.
