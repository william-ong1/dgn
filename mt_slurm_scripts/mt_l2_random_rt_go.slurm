#!/bin/bash
#SBATCH --job-name=mt_l2_rand_rt_go
#SBATCH --account=golub
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_l2_rand_rt_go_%j.out
#SBATCH --error=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_l2_rand_rt_go_%j.err

# Compute probe: 10 generate_data runs on 1 GPU.
# 5 x multi_task (h=64, lr=5e-4) and 5 x multi_task_h256 (h=256, lr=5e-5).
# Each run draws random l2_scale and l2_comm_scale. Both L2 terms ramp
# 0.3 -> 1 over 200 epochs (same as noise). Early stopping is delayed
# until that ramp finishes.

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

REPO=/mmfs1/home/wong2/repos/dgn
RUNS_ROOT=/mmfs1/gscratch/golub/wong2/runs/multi_task/l2_random_probe
TASK=rt_go
SEED="$(python -c "import hashlib,sys; print(int(hashlib.sha256(sys.argv[1].encode()).hexdigest(), 16) % (2**31 - 1))" "${TASK}")"
RAMP=200
STAMP="$(date +%y%m%d%H%M)"

mkdir -p "${RUNS_ROOT}" /mmfs1/gscratch/golub/wong2/slurm_logs
cd "${REPO}"

echo "Job ${SLURM_JOB_ID}: 10 random L2/L2-comm runs on 1 GPU  TASK=${TASK} SEED=${SEED}"

i=0
while read -r EXP HS L2 L2C; do
  RUN_NAME="multi_task_${TASK}_h${HS}_l2${L2}_l2c${L2C}_l2inc${RAMP}_gseed${SEED}_id${STAMP}"
  RUN_DIR="${RUNS_ROOT}/${RUN_NAME}"
  echo "Launching ${EXP} h=${HS} l2=${L2} l2c=${L2C} -> ${RUN_DIR}"

  python "${REPO}/scripts/generate_data.py" "${EXP}" \
    --seed "${SEED}" \
    --run_name "${RUN_DIR}" \
    --override "datamodule.task_names=[${TASK}]" \
    --override "model.l2_scale=${L2}" \
    --override "model.l2_comm_scale=${L2C}" \
    --override "model.graph_kwargs.seed=${SEED}" \
    --override "callbacks.early_stopping.delay_epochs=${RAMP}" &

  i=$((i + 1))
done < <(python -c "
import random
random.seed(0)
vals = ['0', '1e-5', '1e-4', '1e-3', '1e-2']
for exp, hs in [('multi_task', 64), ('multi_task_h256', 256)]:
    for _ in range(5):
        print(exp, hs, random.choice(vals), random.choice(vals))
")

wait
echo "Finished ${i} runs under ${RUNS_ROOT}"
