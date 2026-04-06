#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${HOME}/logs/cpu_threads_fixed_n"
RESULT_DIR="${LOG_DIR}/results"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

BATCH_SIZE=65536
DEPTH=4
MAX_LEAVES=16
N_BATCHES_LIST=(16 32)
THREADS_LIST=(1 2 4 8)
CPU_PREDICTOR="${CPU_PREDICTOR:-numba_parallel}"

for n_batches in "${N_BATCHES_LIST[@]}"; do
  for cpu_threads in "${THREADS_LIST[@]}"; do
    result_file="${RESULT_DIR}/depth_${DEPTH}_threads_${cpu_threads}_n_batches_${n_batches}.txt"
    log_file="${LOG_DIR}/depth_${DEPTH}_threads_${cpu_threads}_n_batches_${n_batches}.log"
    if [[ -s "${result_file}" ]]; then
      echo "Skipping depth=${DEPTH} cpu_threads=${cpu_threads} n_batches=${n_batches}; found ${result_file}"
      continue
    fi
    cmd=(
      python -m benchmarks.compare_cpu_thread_scaling
      --modify
      max_depth "${DEPTH}"
      max_leaves "${MAX_LEAVES}"
      train_batch_size "${BATCH_SIZE}"
      train_n_batches "${n_batches}"
      fresh_batch_size "${BATCH_SIZE}"
      fresh_n_batches "${n_batches}"
      cpu_threads "${cpu_threads}"
      cpu_predictor "${CPU_PREDICTOR}"
      --result-path "${result_file}"
    )
    printf 'Running depth=%s cpu_threads=%s n_batches=%s\n' "${DEPTH}" "${cpu_threads}" "${n_batches}" | tee "${log_file}"
    printf 'Command: %q ' "${cmd[@]}" | tee -a "${log_file}"
    printf '\n' | tee -a "${log_file}"
    "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  done
done

python -m benchmarks.plot_cpu_threads_fixed_n \
  --results-dir "${RESULT_DIR}" \
  --output-dir "${ROOT_DIR}/plots/cpu_threads_fixed_n"
