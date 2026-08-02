#!/bin/bash

#SBATCH --job-name=mem_sweep
#SBATCH --partition=gpu-l40s
#SBATCH --account=golub
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=out/mem_sweep.txt
#SBATCH --error=err/mem_sweep.txt

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

REPO=/mmfs1/home/wong2/repos/dgn
RUNS_ROOT=/mmfs1/gscratch/golub/wong2/runs
mkdir -p "$RUNS_ROOT" out err

STAMP=$(date +%y%m%d%H%M)
MAX_CONCURRENT=10

HIDDEN_SIZES=(1024)
MEMORYS=(1 2 3 4 5 10)

for H in "${HIDDEN_SIZES[@]}"; do
  for WS in "${MEMORYS[@]}"; do
    RUN_DIR="${RUNS_ROOT}/memory_network_h${H}_m${WS}_id${STAMP}"
    LOG="out/mn_h${H}_ws${WS}.log"

    echo "Launching h=${H} ws=${WS} -> ${RUN_DIR}"
    python "${REPO}/generate_data.py" memory_network \
      --hidden_size "$H" \
      --memory "$WS" \
      --noise 2 \
      --input_weight_init_var_scale 1.0 \
      --run_name "$RUN_DIR" \
      > "$LOG" 2>&1 &

    while [ "$(jobs -r | wc -l)" -ge "$MAX_CONCURRENT" ]; do
      sleep 10
    done
  done
done

wait
echo "All sweep runs done."
