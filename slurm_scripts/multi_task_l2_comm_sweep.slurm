#!/bin/bash
#SBATCH --job-name=mt_l2c
#SBATCH --account=golub
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_l2c_%j.out
#SBATCH --error=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_l2c_%j.err

# Phase 2: freeze model.l2_scale=1e-5 (best from phase 1) and sweep
# model.l2_comm_scale. L2 and L2-comm ramp like noise: init -> 1 over
# increase epochs. All runs in parallel on one GPU. Shared connectome
# via graph_kwargs.seed=0.

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

REPO=/mmfs1/home/wong2/repos/dgn
RUNS_ROOT=/mmfs1/gscratch/golub/wong2/runs/multi_task/l2_comm_runs
SEED=0
GSEED=0
TASK=rt_go
L2=1e-4
L2CS=(0 1e-5 1e-4 1e-3 1e-2)
L2_START=0
L2_INCREASE=200
L2_INIT=0.3
L2C_START=0
L2C_INCREASE=200
L2C_INIT=0.3
STAMP="$(date +%y%m%d%H%M)"

mkdir -p "${RUNS_ROOT}"
mkdir -p /mmfs1/gscratch/golub/wong2/slurm_logs

cd "${REPO}"

echo "Job ${SLURM_JOB_ID}: launching ${#L2CS[@]} l2_comm_scale runs (l2_scale=${L2}, graph_seed=${GSEED}) in parallel on 1 GPU"
echo "L2 ramp: start=${L2_START} increase=${L2_INCREASE} init=${L2_INIT}"
echo "L2-comm ramp: start=${L2C_START} increase=${L2C_INCREASE} init=${L2C_INIT}"

i=0
for L2C in "${L2CS[@]}"; do
  RUN_NAME="multi_task_${TASK}_l2${L2}_l2c${L2C}_l2inc${L2_INCREASE}_l2cinc${L2C_INCREASE}_gseed${GSEED}_seed${SEED}_id${STAMP}"
  RUN_DIR="${RUNS_ROOT}/${RUN_NAME}"

  echo "Launching ${TASK} l2_scale=${L2} l2_comm_scale=${L2C} -> ${RUN_DIR}"
  python "${REPO}/scripts/generate_data.py" multi_task \
    --seed "${SEED}" \
    --run_name "${RUN_DIR}" \
    --override "datamodule.task_names=[${TASK}]" \
    --override "model.l2_scale=${L2}" \
    --override "model.l2_start=${L2_START}" \
    --override "model.l2_increase=${L2_INCREASE}" \
    --override "model.l2_init=${L2_INIT}" \
    --override "model.l2_comm_scale=${L2C}" \
    --override "model.l2_comm_start=${L2C_START}" \
    --override "model.l2_comm_increase=${L2C_INCREASE}" \
    --override "model.l2_comm_init=${L2C_INIT}" \
    --override "model.graph_kwargs.seed=${GSEED}" &

  i=$((i + 1))
done

wait
echo "Finished all ${i} runs under ${RUNS_ROOT}"
