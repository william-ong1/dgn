#!/bin/bash
#SBATCH --job-name=mt_l2cg128
#SBATCH --account=golub
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_l2cg128_%j.out
#SBATCH --error=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_l2cg128_%j.err

# Same 128-neuron l2_comm_scale sweep as multi_task_l2_comm_sweep_h128.slurm,
# l2_scale=1e-5, but graph seed 1.

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

REPO=/mmfs1/home/wong2/repos/dgn
RUNS_ROOT=/mmfs1/gscratch/golub/wong2/runs/multi_task/l2_comm_runs
SEED=0
GSEED=1
TASK=rt_go
HS=128
L2=1e-5
L2CS=(0 1e-5 1e-4 1e-3 1e-2)
STAMP="$(date +%y%m%d%H%M)"

mkdir -p "${RUNS_ROOT}"
mkdir -p /mmfs1/gscratch/golub/wong2/slurm_logs

cd "${REPO}"

echo "Job ${SLURM_JOB_ID}: launching ${#L2CS[@]} l2_comm_scale runs (h=${HS}, l2_scale=${L2}, graph_seed=${GSEED}) in parallel on 1 GPU"

i=0
for L2C in "${L2CS[@]}"; do
  RUN_NAME="multi_task_${TASK}_h${HS}_l2${L2}_l2c${L2C}_gseed${GSEED}_seed${SEED}_id${STAMP}"
  RUN_DIR="${RUNS_ROOT}/${RUN_NAME}"

  echo "Launching ${TASK} h=${HS} l2_scale=${L2} l2_comm_scale=${L2C} graph_seed=${GSEED} -> ${RUN_DIR}"
  python "${REPO}/scripts/generate_data.py" multi_task \
    --seed "${SEED}" \
    --hidden_size "${HS}" \
    --run_name "${RUN_DIR}" \
    --override "datamodule.task_names=[${TASK}]" \
    --override "model.l2_scale=${L2}" \
    --override "model.l2_comm_scale=${L2C}" \
    --override "model.graph_kwargs.seed=${GSEED}" &

  i=$((i + 1))
done

wait
echo "Finished all ${i} runs under ${RUNS_ROOT}"
