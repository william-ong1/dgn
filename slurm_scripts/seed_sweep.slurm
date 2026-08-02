#!/bin/bash

#SBATCH --job-name=seed_sweep
#SBATCH --partition=gpu-l40s
#SBATCH --account=cse
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=out/seed_sweep.txt
#SBATCH --error=err/seed_sweep.txt

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

REPO=/mmfs1/home/wong2/repos/dgn
RUNS_ROOT=/mmfs1/gscratch/golub/wong2/runs
mkdir -p "$RUNS_ROOT" out err

STAMP=$(date +%y%m%d%H%M)
MAX_CONCURRENT=10

HIDDEN_SIZES=(512)
SEEDS=(0 42 123 2026 1337)

for H in "${HIDDEN_SIZES[@]}"; do
  for S in "${SEEDS[@]}"; do
    RUN_DIR="${RUNS_ROOT}/memory_network_h${H}_s${S}_id${STAMP}"
    LOG="out/mn_h${H}_s${S}.log"

    echo "Launching h=${H} seed=${S} -> ${RUN_DIR}"
    python "${REPO}/scripts/generate_data.py" memory_network \
      --hidden_size "$H" \
      --noise 0.5 \
      --input_weight_init_var_scale 1.0 \
      --seed "$S" \
      --run_name "$RUN_DIR" \
      > "$LOG" 2>&1 &

    while [ "$(jobs -r | wc -l)" -ge "$MAX_CONCURRENT" ]; do
      sleep 10
    done
  done
done

wait
echo "All sweep runs done."
