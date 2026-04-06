# Benchmarks

This directory contains the benchmark and comparison utilities for CMGD-Tree.

All benchmark commands should be run from the repository root.

## What Is Here

- [compare_normal_single_core_inference.py](/home/rschoefbeck/CMGD-Tree/benchmarks/compare_normal_single_core_inference.py)
  Matched comparison against XGBoost on the normal example.

- [xgboost_normal_compare.py](/home/rschoefbeck/CMGD-Tree/benchmarks/xgboost_normal_compare.py)
  Standalone XGBoost baseline script.

- [compare_cpu_thread_scaling.py](/home/rschoefbeck/CMGD-Tree/benchmarks/compare_cpu_thread_scaling.py)
  Single benchmark point for CPU training and fresh inference at a chosen thread count.

- [plot_depth_scaling.py](/home/rschoefbeck/CMGD-Tree/benchmarks/plot_depth_scaling.py)
  Plot depth-scaling comparison results.

- [plot_cpu_thread_scaling.py](/home/rschoefbeck/CMGD-Tree/benchmarks/plot_cpu_thread_scaling.py)
  Plot CPU thread-scaling sweep results.

- [plot_cpu_threads_fixed_n.py](/home/rschoefbeck/CMGD-Tree/benchmarks/plot_cpu_threads_fixed_n.py)
  Plot fixed-`N` CPU thread scaling results.

- [run_depth_scaling_benchmarks.sh](/home/rschoefbeck/CMGD-Tree/benchmarks/run_depth_scaling_benchmarks.sh)
  Serial depth-vs-`N` comparison sweep.

- [run_cpu_thread_scaling_benchmarks.sh](/home/rschoefbeck/CMGD-Tree/benchmarks/run_cpu_thread_scaling_benchmarks.sh)
  Larger CPU thread-scaling sweep.

- [run_cpu_threads_fixed_n.sh](/home/rschoefbeck/CMGD-Tree/benchmarks/run_cpu_threads_fixed_n.sh)
  Focused CPU thread-scaling sweep for fixed event counts.

## Typical Commands

Matched XGBoost comparison:

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

Focused fixed-`N` CPU thread comparison:

```bash
bash benchmarks/run_cpu_threads_fixed_n.sh
```

## Output Locations

The shell scripts write logs and result files under `~/logs/`.
Plots are written under `./plots/`.

Examples:

- depth scaling:
  - logs: `~/logs/depth_scaling/`
  - plots: `./plots/depth_scaling/`

- CPU thread scaling:
  - logs: `~/logs/cpu_thread_scaling/`
  - plots: `./plots/cpu_thread_scaling/`

- fixed-`N` CPU thread scaling:
  - logs: `~/logs/cpu_threads_fixed_n/`
  - plots: `./plots/cpu_threads_fixed_n/`

## Notes

- The shell scripts are resumable: existing nonempty result files are skipped.
- They run benchmark points one after the other to reduce interference.
- If you want to tweak workload size, edit the corresponding shell script or pass environment variables where supported.
