# CMGD-Tree

CMGD-Tree is a small playground for probabilistic tree boosting with:

- streamed toy data
- histogram-based trees
- CPU or GPU training
- CPU or GPU prediction
- family-side MGD and NGD updates

If you want one mental model for the codebase, use this:

- choose a `family`
- choose the input and output dimensions
- choose the tree size and number of boosting rounds
- run the demo

The main entry point is:

```bash
python fit_single_tree_hist_demo.py
```

Writeup:

- [PDF](https://hephyanalysissw.github.io/CMGD-Tree/writeup.pdf)
- [build workflow](https://github.com/HephyAnalysisSW/CMGD-Tree/actions/workflows/build-writeup.yml)

## Quick Start

Start with the default multi-dimensional Gaussian example:

```bash
python fit_single_tree_hist_demo.py
```

That runs the `normal_identity` family with:

- `n_features = 32`
- `n_classes = 4`
- shallow trees
- 2 boosting rounds

If you want to see the fitted trees:

```bash
python fit_single_tree_hist_demo.py --print-trees
```

If you want timing output:

```bash
python fit_single_tree_hist_demo.py --profile
```

If you want plots:

```bash
python fit_single_tree_hist_demo.py --plot
```

## A Good First Run

A more realistic first run is a slightly larger multi-dimensional Gaussian:

```bash
python fit_single_tree_hist_demo.py \
  --print-trees \
  --modify \
    family normal_identity \
    n_features 8 \
    n_classes 4 \
    max_depth 3 \
    max_leaves 8 \
    n_boost_rounds 20 \
    learning_rate 0.2
```

Read that command like this:

- `family normal_identity`
  chooses the probabilistic model
- `n_features 8`
  sets the input dimension
- `n_classes 4`
  sets the output dimension
- `max_depth 3` and `max_leaves 8`
  make the trees more expressive
- `n_boost_rounds 20`
  fits a larger ensemble
- `learning_rate 0.2`
  makes boosting more conservative

That is usually the easiest place to start changing things.

## How To Think About The Settings

There are three groups of settings.

### 1. Family

This answers: what distribution or statistical problem am I fitting?

Current families:

- `normal_identity`
- `poisson`
- `poisson_ngd`
- `gamma`
- `negative_binomial`
- `heteroskedastic_normal`
- `heteroskedastic_normal_ngd`

Typical first changes:

```bash
python fit_single_tree_hist_demo.py --modify family poisson
python fit_single_tree_hist_demo.py --modify family gamma
python fit_single_tree_hist_demo.py --modify family negative_binomial
```

### 2. Dimensions and Data Size

This answers: how many inputs, how many outputs, and how much data?

Use:

```python
--modify \
  n_features 8 \
  n_classes 4 \
  batch_size 65536 \
  n_batches 12
```

Meaning:

- `n_features`
  input dimension
- `n_classes`
  output dimension of the fitted target statistics
- `batch_size`
  events per streamed batch
- `n_batches`
  number of streamed batches

Total training events are:

```text
batch_size * n_batches
```

Example:

```bash
python fit_single_tree_hist_demo.py \
  --modify n_features 16 n_classes 4 batch_size 32768 n_batches 24
```

### 3. Tree and Training

This answers: how large should the trees be, and how aggressively should boosting update?

Use:

```python
--modify \
  max_depth 4 \
  max_leaves 16 \
  max_bin 64 \
  n_boost_rounds 50 \
  learning_rate 0.1
```

Meaning:

- `max_depth`
  maximum tree depth
- `max_leaves`
  maximum number of leaves
- `max_bin`
  histogram resolution for split search
- `n_boost_rounds`
  number of boosting iterations
- `learning_rate`
  shrinkage per tree

Example:

```bash
python fit_single_tree_hist_demo.py \
  --modify max_depth 4 max_leaves 16 n_boost_rounds 50 learning_rate 0.1
```

## Common Workflows

Run a Gaussian example with more dimensions:

```bash
python fit_single_tree_hist_demo.py \
  --modify family normal_identity n_features 16 n_classes 8
```

Switch to Poisson:

```bash
python fit_single_tree_hist_demo.py \
  --modify family poisson n_features 8 n_classes 4
```

Run the NGD Poisson example:

```bash
python fit_single_tree_hist_demo.py \
  --modify family poisson_ngd n_features 8 n_classes 4
```

Run everything on CPU:

```bash
python fit_single_tree_hist_demo.py \
  --modify training_backend cpu predict_method cpu cpu_predictor numba_parallel
```

Run GPU training with GPU prediction:

```bash
python fit_single_tree_hist_demo.py \
  --modify training_backend gpu predict_method gpu
```

Run GPU training but keep prediction on CPU:

```bash
python fit_single_tree_hist_demo.py \
  --modify training_backend gpu predict_method cpu cpu_predictor numba_parallel
```

## Command Line

Top-level options:

- `--config path-or-name`
  load a complete example YAML, e.g. `--config poisson`
- `--modify key value ...`
  override config values
- `--profile`
  print timing and memory summaries
- `--plot`
  write plots under `./plots/<training_id>/`
- `--print-trees`
  print the fitted trees
- `--full-output`
  compatibility alias for `--plot --print-trees`

Example:

```bash
python fit_single_tree_hist_demo.py \
  --plot \
  --print-trees \
  --modify family normal_identity n_features 8 n_classes 4 max_depth 3
```

## Config Keys

The runnable examples now live in YAML:

- `configs/default.yaml`
  internal fallback defaults
- `configs/examples/*.yaml`
  complete user-facing example configs

Each example YAML has four top-level groups:

- `tree`
- `dataset`
- `training`
- `plot`

Most important keys:

- `family`: statistical model
- `n_features`: input dimension
- `n_classes`: output dimension
- `batch_size`, `n_batches`: training size
- `max_depth`, `max_leaves`: tree expressivity
- `n_boost_rounds`, `learning_rate`: boosting strength
- `training_backend`: `auto`, `gpu`, or `cpu`
- `predict_method`: `gpu` or `cpu`
- `cpu_predictor`: `index`, `leaf_mask`, `numba`, or `numba_parallel`

Two keys that are useful but easy to misunderstand:

- `cut_sample_rows`
  only controls how many rows are used to estimate feature cuts
- `max_bin`
  controls how fine the histogram split search is

## Example Configs

Each example YAML is explicit and self-contained.

For example:

- `configs/examples/heteroskedastic_normal.yaml`
  uses the 2D heteroskedastic toy stream and scalar diagnostic plot mode
- `configs/examples/gamma.yaml`
  and `configs/examples/negative_binomial.yaml`
  use their matching toy streams and longer boosted runs

## Prediction Modes

Prediction is an important runtime choice.

Use `predict_method=gpu` when:

- you are already training on GPU
- your batches are large
- you want cache updates to stay on device

Use `predict_method=cpu` when:

- you want easier inspection
- you want to compare CPU predictors
- you are running a CPU-only setup

The default CPU predictor is:

- `numba_parallel`

## MGD and NGD

The trainer always fits trees to a family-supplied pseudo-response.

For MGD, that pseudo-response is the plain residual-style target:

```text
T(y) - eta*(x)
```

For NGD, the family can precondition that target with the Fisher information:

```text
G(x)^{-1} (T(y) - eta*(x))
```

So the tree code stays the same, while the family changes the geometry.

## Where To Add Things

If you extend the project:

- add a new statistical model in `families/`
- add a new toy generator or real loader in `data_providers/`
- add a new runnable setup in `configs/examples/`

That is the intended workflow for new users as well:

1. start from an existing example
2. change the family
3. change the dimensions
4. change the tree and training settings
5. run again
