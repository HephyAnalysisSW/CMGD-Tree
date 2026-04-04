# CMGD-Tree

Lean GPU-first histogram tree fitting with streamed data, additive boosting, and CPU/GPU inference benchmarks.

The current code fits vector-valued regression trees to streamed target statistics `T(y)`. The default example is a shallow boosted normal-identity model, and the code also includes a first Poisson family example.

## Main Files

- [fit_single_tree_hist_demo.py](/home/rschoefbeck/CMGD-Tree/fit_single_tree_hist_demo.py): main demo and benchmark entrypoint
- [gpu_single_tree_trainer.py](/home/rschoefbeck/CMGD-Tree/gpu_single_tree_trainer.py): GPU training backend and in-memory `TrainingCache`
- [single_tree.py](/home/rschoefbeck/CMGD-Tree/single_tree.py): `SingleTree`, `AdditiveEnsemble`, CPU predictors
- [normal_identity_family.py](/home/rschoefbeck/CMGD-Tree/normal_identity_family.py): user-facing family definitions and toy data streams
- [plot_feature_ratios.py](/home/rschoefbeck/CMGD-Tree/plot_feature_ratios.py): diagnostic plotting
- [compare_normal_single_core_inference.py](/home/rschoefbeck/CMGD-Tree/compare_normal_single_core_inference.py): matched comparison against XGBoost
- [xgboost_normal_compare.py](/home/rschoefbeck/CMGD-Tree/xgboost_normal_compare.py): standalone XGBoost baseline

## Quick Start

Standard run:

```bash
python fit_single_tree_hist_demo.py
```

Built-in profiling:

```bash
python fit_single_tree_hist_demo.py --profile
```

Full output with tree printing and plots:

```bash
python fit_single_tree_hist_demo.py --profile --full-output
```

Logs should go under `~/logs`, for example:

```bash
python fit_single_tree_hist_demo.py --profile | tee ~/logs/standard_test.log
```

## Config Overrides

All defaults live near the top of [fit_single_tree_hist_demo.py](/home/rschoefbeck/CMGD-Tree/fit_single_tree_hist_demo.py) in:

- `TREE_CONFIG`
- `DATASET_CONFIG`
- `TRAINING_CONFIG`

Override any config entry with `--modify key value ...`:

```bash
python fit_single_tree_hist_demo.py --profile --modify max_depth 4 max_leaves 16
```

Example heavier CPU inference benchmark:

```bash
python fit_single_tree_hist_demo.py \
  --profile \
  --modify \
    n_features 4 \
    n_boost_rounds 100 \
    predict_method cpu \
    cpu_predictor numba_parallel \
    batch_size 65536 \
    n_batches 12 \
    fresh_inference_batch_size 262144 \
    fresh_inference_n_batches 64 \
  | tee ~/logs/cpu_predict_numba_parallel_manual.log
```

## Current Families

Select the family with:

```bash
python fit_single_tree_hist_demo.py --modify family normal_identity
python fit_single_tree_hist_demo.py --modify family poisson
```

Currently implemented:

- `normal_identity`
  - target statistics are one-hot class targets in the toy example
  - monitor metric is MSE
- `poisson`
  - target statistics are vector Poisson counts
  - monitor metric is Poisson NLL

Family-specific code lives in [normal_identity_family.py](/home/rschoefbeck/CMGD-Tree/normal_identity_family.py). That file is the intended user-facing place for:

- toy stream generation
- target statistics `T(y)`
- base prediction
- prediction-domain projection
- monitoring metric
- plot configuration

## Prediction Modes

Set prediction backend with:

```bash
python fit_single_tree_hist_demo.py --modify predict_method cpu
python fit_single_tree_hist_demo.py --modify predict_method gpu
```

CPU predictors:

- `index`: simple row-index traversal
- `leaf_mask`: NumPy leaf-mask traversal
- `numba`: compiled single-core forest traversal
- `numba_parallel`: compiled multi-core forest traversal

Example:

```bash
python fit_single_tree_hist_demo.py --profile --modify predict_method cpu cpu_predictor numba_parallel
```

## Profiling Stages

`--profile` reports wall time, process CPU time, RSS, GPU memory, and CuPy pool usage for the main stages.

Typical stages include:

- `cache_build`
- `tree_growth_round_k`
- `cache_update_round_k`
- `training`
- `evaluation`
- `fresh_inference`

`evaluation` uses the cached training stream. `fresh_inference` streams new raw batches and is the better proxy for real inference throughput.

## XGBoost Comparison

Standalone XGBoost benchmark:

```bash
python xgboost_normal_compare.py --modify n_jobs 1
```

Matched comparison against our implementation:

```bash
python compare_normal_single_core_inference.py --modify cpu_predictor numba
```

Deeper multicore comparison:

```bash
python compare_normal_single_core_inference.py \
  --modify max_depth 7 max_leaves 128 cpu_predictor numba_parallel xgb_n_jobs 64 \
  | tee ~/logs/compare_normal_multicore_inference_depth7_numba_parallel.log
```

## Plots

By default the demo writes:

- feature-density plots
- feature-target 2D mean-overlay plots

Output directory:

```bash
./plots/<plot_training_id>/
```

The default training id is `single_tree_demo`.

## Notes

- Training cache is in memory now, but the code is organized around `TrainingCache` so storage can move later.
- The current training backend is GPU-only.
- CPU training is not implemented yet.
- Plotting is intentionally outside the main profiling target.
