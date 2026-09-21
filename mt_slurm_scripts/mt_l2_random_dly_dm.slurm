#!/bin/bash
#SBATCH --job-name=mt_l2r_dly_dm
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_l2r_dly_dm_%j.out
#SBATCH --error=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_l2r_dly_dm_%j.err

# 10 generate_data runs on 1 GPU: 2 tasks x 5 x h=64.
# Each run draws a unique (l2_scale, l2_comm_scale) pair and its own random seed.
# Both L2 terms ramp 0.3 -> 1 over 200 epochs; early stopping waits for that.

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

REPO=/mmfs1/home/wong2/repos/dgn
RUNS_ROOT=/mmfs1/gscratch/golub/wong2/runs/multi_task/l2_random_runs
TASKS="dly_dm_1 dly_dm_2"
RAMP=200
STAMP="$(date +%y%m%d%H%M)"

mkdir -p "${RUNS_ROOT}" /mmfs1/gscratch/golub/wong2/slurm_logs
cd "${REPO}"

# Lustre/gscratch: HDF5 locks fail when many runs write data.h5 at once.
export HDF5_USE_FILE_LOCKING=FALSE

echo "Job ${SLURM_JOB_ID}: 10 random L2/L2-comm runs on 1 GPU  tasks=${TASKS}"

i=0
while read -r TASK EXP HS L2 L2C SEED; do
  RUN_NAME="multi_task_${TASK}_h${HS}_l2${L2}_l2c${L2C}_l2inc${RAMP}_gseed${SEED}_seed${SEED}_id${STAMP}"
  RUN_DIR="${RUNS_ROOT}/${RUN_NAME}"
  echo "Launching ${TASK} ${EXP} h=${HS} l2=${L2} l2c=${L2C} seed=${SEED} -> ${RUN_DIR}"

  python "${REPO}/scripts/generate_data.py" "${EXP}" \
    --seed "${SEED}" \
    --run_name "${RUN_DIR}" \
    --override "datamodule.task_names=[${TASK}]" \
    --override "model.l2_scale=${L2}" \
    --override "model.l2_comm_scale=${L2C}" \
    --override "model.graph_kwargs.seed=${SEED}" \
    --override "callbacks.early_stopping.delay_epochs=${RAMP}" &

  i=$((i + 1))
done < <(python - "${TASKS}" <<'PY'
import random
import sys

tasks = sys.argv[1].split()
vals = ["0", "1e-5", "1e-4", "1e-3", "1e-2"]
pairs = [(a, b) for a in vals for b in vals]

for task in tasks:
    for exp, hs in [("multi_task", 64)]:
        for l2, l2c in random.sample(pairs, 5):
            print(task, exp, hs, l2, l2c, random.randrange(2**31 - 1))
PY
)

wait
echo "Finished ${i} runs under ${RUNS_ROOT}"
