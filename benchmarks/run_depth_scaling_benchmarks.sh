#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${HOME}/logs/depth_scaling"
RESULT_DIR="${LOG_DIR}/results"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

BATCH_SIZE=65536
N_BATCHES_LIST=(1 4 16 64 128)
DEPTHS=(3 4 5 6)
XGB_N_JOBS="${XGB_N_JOBS:--1}"
CPU_PREDICTOR="${CPU_PREDICTOR:-numba_parallel}"
TRAINING_BACKEND="${TRAINING_BACKEND:-gpu}"
CPU_THREADS="${CPU_THREADS:-1}"

for depth in "${DEPTHS[@]}"; do
  max_leaves=$((1 << depth))
  for n_batches in "${N_BATCHES_LIST[@]}"; do
    result_file="${RESULT_DIR}/depth_${depth}_n_batches_${n_batches}.txt"
    log_file="${LOG_DIR}/depth_${depth}_n_batches_${n_batches}.log"
    if [[ -s "${result_file}" ]]; then
      echo "Skipping depth=${depth} n_batches=${n_batches}; found ${result_file}"
      continue
    fi
    cmd=(
      python -m benchmarks.compare_normal_single_core_inference
      --modify
      max_depth "${depth}"
      max_leaves "${max_leaves}"
      train_batch_size "${BATCH_SIZE}"
      train_n_batches "${n_batches}"
      fresh_batch_size "${BATCH_SIZE}"
      fresh_n_batches "${n_batches}"
      training_backend "${TRAINING_BACKEND}"
      cpu_threads "${CPU_THREADS}"
      cpu_predictor "${CPU_PREDICTOR}"
      xgb_n_jobs "${XGB_N_JOBS}"
      --result-path "${result_file}"
    )
    printf 'Running depth=%s n_batches=%s\n' "${depth}" "${n_batches}"
    printf 'Command: %q ' "${cmd[@]}" | tee "${log_file}"
    printf '\n' | tee -a "${log_file}"
    "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  done
done

python -m benchmarks.plot_depth_scaling \
  --results-dir "${RESULT_DIR}" \
  --output-dir "${ROOT_DIR}/plots/depth_scaling"
