#!/bin/bash

#SBATCH --job-name=pd_noise_sweep_seed0_small
#SBATCH --partition=gpu-l40s
#SBATCH --account=golub
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=out/pd_noise_sweep_seed0_small.txt
#SBATCH --error=err/pd_noise_sweep_seed0_small.txt

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

REPO=/mmfs1/home/wong2/repos/dgn
RUNS_ROOT=/mmfs1/gscratch/golub/wong2/runs/pass_decision
mkdir -p "$RUNS_ROOT" out err

STAMP=$(date +%y%m%d%H%M)
MAX_CONCURRENT=40

HIDDEN_SIZES=(32 64 128)
NOISES=(0.001 0.005 0.01 0.05 0.1 0.5 1 2 5)

for H in "${HIDDEN_SIZES[@]}"; do
  for N in "${NOISES[@]}"; do
    RUN_DIR="${RUNS_ROOT}/pass_decision_h${H}_n${N}_seed0_small_id${STAMP}"
    LOG="out/mn_h${H}_n${N}.log"

    echo "Launching h=${H} noise=${N} -> ${RUN_DIR}"
    python "${REPO}/scripts/generate_data.py" pass_decision \
      --hidden_size "$H" \
      --noise_p "$N" \
      --noise_d "$N" \
      --input_weight_init_var_scale 1.0 \
      --seed 0 \
      --run_name "$RUN_DIR" \
      > "$LOG" 2>&1 &

    while [ "$(jobs -r | wc -l)" -ge "$MAX_CONCURRENT" ]; do
      sleep 10
    done
  done
done

wait
echo "All sweep runs done."
