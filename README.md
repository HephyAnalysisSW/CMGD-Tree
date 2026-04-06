# CMGD-Tree

CMGD-Tree is a lean streamed histogram-tree sandbox for vector-valued boosting with:

- GPU training
- CPU or GPU inference
- family-specific target statistics and monitoring losses
- toy providers for quick experiments

The main entrypoint is [fit_single_tree_hist_demo.py](/home/rschoefbeck/CMGD-Tree/fit_single_tree_hist_demo.py).

## Repository Layout

- [fit_single_tree_hist_demo.py](/home/rschoefbeck/CMGD-Tree/fit_single_tree_hist_demo.py): user-facing demo CLI
- [gpu_single_tree_trainer.py](/home/rschoefbeck/CMGD-Tree/gpu_single_tree_trainer.py): GPU trainer backend and `TrainingCache`
- [single_tree.py](/home/rschoefbeck/CMGD-Tree/single_tree.py): `SingleTree`, `AdditiveEnsemble`, CPU predictors
- [families/](/home/rschoefbeck/CMGD-Tree/families): family interfaces and implementations
- [providers/](/home/rschoefbeck/CMGD-Tree/providers): streamed toy data providers
- [plot_feature_ratios.py](/home/rschoefbeck/CMGD-Tree/plot_feature_ratios.py): diagnostic plots
- [benchmarks/](/home/rschoefbeck/CMGD-Tree/benchmarks): XGBoost comparisons and scaling scripts
- [normal_identity_family.py](/home/rschoefbeck/CMGD-Tree/normal_identity_family.py): compatibility shim for older imports

## Basic Usage

Small default run:

```bash
python fit_single_tree_hist_demo.py
```

Print the fitted trees:

```bash
python fit_single_tree_hist_demo.py --print-trees
```

Generate plots:

```bash
python fit_single_tree_hist_demo.py --plot
```

Print trees and generate plots:

```bash
python fit_single_tree_hist_demo.py --print-trees --plot
```

Profile training and inference only:

```bash
python fit_single_tree_hist_demo.py --profile
```

Logs should go under `~/logs`, for example:

```bash
python fit_single_tree_hist_demo.py --plot > ~/logs/demo_run.log 2>&1
```

## CLI Flags

The demo currently supports these top-level flags:

- `--modify key value ...`
  - override any config entry from the tree, dataset, or training config groups
- `--profile`
  - print built-in timing and memory summaries for cache build, tree growth, evaluation, and fresh inference
- `--plot`
  - write diagnostic plots under `./plots/<plot_training_id>/`
- `--print-trees`
  - print every fitted tree after the run
- `--full-output`
  - compatibility alias for `--plot --print-trees`

## Config Overrides

The defaults live near the top of [fit_single_tree_hist_demo.py](/home/rschoefbeck/CMGD-Tree/fit_single_tree_hist_demo.py) in three groups.

### `TREE_CONFIG`

- `max_bin`
- `cut_sample_rows`
- `grow_policy`
- `max_depth`
- `max_leaves`
- `min_samples_leaf`
- `min_split_loss`
- `reg_lambda`
- `family`
- `class_weights`
- `fit_target_indices`

### `DATASET_CONFIG`

- `n_features`
- `n_classes`
- `batch_size`
- `n_batches`
- `seed`
- `feature_offset_scale`
- `feature_noise`

### `TRAINING_CONFIG`

- `plot_training_id`
- `plot_bins`
- `plot_mode`
- `threads_per_block`
- `training_backend`
- `cpu_threads`
- `predict_method`
- `cpu_predictor`
- `n_boost_rounds`
- `learning_rate`
- `fit_schedule_groups`
- `fit_schedule_probs`
- `fresh_inference_batch_size`
- `fresh_inference_n_batches`

Override any of them with `--modify`.

Examples:

Normal toy run with a slightly deeper model, plots, and tree printing:

```bash
python fit_single_tree_hist_demo.py \
  --print-trees \
  --plot \
  --modify n_features 4 max_depth 3 max_leaves 8
```

Poisson MGD toy run with plots:

```bash
python fit_single_tree_hist_demo.py \
  --plot \
  --modify family poisson n_features 4 n_classes 4 max_depth 3 max_leaves 8
```

Poisson NGD toy run with plots and printed trees:

```bash
python fit_single_tree_hist_demo.py \
  --print-trees \
  --plot \
  --modify family poisson_ngd n_features 4 n_classes 4 max_depth 3 max_leaves 8
```

Scalar heteroskedastic normal toy run with plots:

```bash
python fit_single_tree_hist_demo.py \
  --print-trees \
  --plot \
  --modify family heteroskedastic_normal
```

CPU inference with the compiled multicore predictor:

```bash
python fit_single_tree_hist_demo.py \
  --modify predict_method cpu n_features 4
```

Explicit CPU training with a bounded thread override:

```bash
python fit_single_tree_hist_demo.py \
  --modify training_backend cpu cpu_threads 8 predict_method cpu n_features 4
```

Experimental coordinate-selection examples:

```bash
python fit_single_tree_hist_demo.py \
  --modify family heteroskedastic_normal fit_target_indices "[1]"

python fit_single_tree_hist_demo.py \
  --modify family heteroskedastic_normal fit_schedule_groups "[[0],[1]]"
```

The global default learning rate is `1.0`.

Concrete toy examples may provide their own example defaults for tree, dataset, and
training settings. These are part of the example specification rather than the generic
CLI. They are applied only for keys you did not override explicitly.

The current `heteroskedastic_normal` example uses:

- `n_features=4`
- `n_classes=2`
- `max_depth=3`
- `max_leaves=8`
- `learning_rate=0.2`

Examples:

```bash
python fit_single_tree_hist_demo.py --modify family heteroskedastic_normal
python fit_single_tree_hist_demo.py --modify family heteroskedastic_normal learning_rate 1.0
```

## Families

Implemented family names:

- `normal_identity`
- `heteroskedastic_normal`
- `poisson`
- `poisson_mgd`
- `poisson_ngd`

Select one with:

```bash
python fit_single_tree_hist_demo.py --modify family normal_identity
python fit_single_tree_hist_demo.py --modify family heteroskedastic_normal
python fit_single_tree_hist_demo.py --modify family poisson
python fit_single_tree_hist_demo.py --modify family poisson_ngd
```

Current division of responsibility:

- [families/](/home/rschoefbeck/CMGD-Tree/families) defines:
  - state representation
  - target statistics
  - update rule hooks
  - monitoring loss
- [providers/](/home/rschoefbeck/CMGD-Tree/providers) defines:
  - streamed toy data generation
  - use-case-specific plotting recipes
  - the batch payload consumed by the trainer

## Prediction Modes

Prediction backend:

- `predict_method cpu`
- `predict_method gpu`

CPU predictor choices:

- `index`
- `leaf_mask`
- `numba`
- `numba_parallel`

The default CPU predictor is `numba_parallel`.

Example:

```bash
python fit_single_tree_hist_demo.py \
  --modify predict_method cpu n_features 4
```

## Plotting

Plotting is off by default.

When `--plot` is enabled, the demo writes diagnostic plots to:

```bash
./plots/<plot_training_id>/
```

The current default `plot_mode` is `all`, which means:

- the provider-specific default plot set for the selected toy example

Examples:

- `normal_identity`
  - feature-density plots
  - feature-target 2D mean-overlay plots
- `poisson`
  - feature-density plots
  - feature-target 2D mean-overlay plots
- `heteroskedastic_normal`
  - x-vs-y density with observed/predicted mean overlay
  - observed/predicted conditional variance overlay

For small exploratory runs, it is often useful to reduce the toy dimensions:

```bash
python fit_single_tree_hist_demo.py \
  --plot \
  --modify n_features 4 n_classes 4 plot_training_id small_demo
```

## Benchmark Utilities

Benchmark helpers live under [benchmarks/](/home/rschoefbeck/CMGD-Tree/benchmarks).

Matched comparison against XGBoost:

```bash
python -m benchmarks.compare_normal_single_core_inference \
  --modify training_backend gpu cpu_threads 1
```

Standalone XGBoost baseline:

```bash
python -m benchmarks.xgboost_normal_compare --modify n_jobs 1
```

Depth-scaling sweep:

```bash
bash benchmarks/run_depth_scaling_benchmarks.sh
```

CPU thread-scaling sweep:

```bash
bash benchmarks/run_cpu_thread_scaling_benchmarks.sh
```

That sweep benchmarks CPU training and fresh CPU inference as a function of:

- tree depth
- total number of streamed events
- `cpu_threads`

and writes result files under:

```bash
~/logs/cpu_thread_scaling/results/
```

followed by plots under:

```bash
./plots/cpu_thread_scaling/
```

Fixed-`N` CPU thread sweep with thread count on the x-axis:

```bash
bash benchmarks/run_cpu_threads_fixed_n.sh
```

That sweep writes result files under:

```bash
~/logs/cpu_threads_fixed_n/results/
```

and plots under:

```bash
./plots/cpu_threads_fixed_n/
```

## Notes

- Training defaults to GPU when available and falls back to CPU otherwise.
- CPU training exists as a separate histogram backend.
- `cpu_threads` defaults to a conservative value of `1`.
- CPU tree fitting is intentionally single-threaded now because the threaded fit path was slower in benchmarks.
- `cpu_threads` mainly affects CPU cache updates and CPU fresh inference, where threading does help.
- `cpu_predictor` defaults to `numba_parallel`, which is the current best CPU inference path.
- The single-core XGBoost comparison benchmark defaults to `cpu_predictor=numba`.
- Plotting is intentionally outside the main profiling target.
- `TrainingCache` is in-memory now, but its shape is intended to make later storage changes possible.
