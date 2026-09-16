#!/bin/bash
#SBATCH --job-name=mt_l2_w_dly_go
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_l2_w_dly_go_%j.out
#SBATCH --error=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_l2_w_dly_go_%j.err

# Sweep l2_scale at h=64 (configs/multi_task) and h=128 (configs/multi_task_h128)
# in parallel on 1 GPU. h=128 uses lr_init=1e-4 and max_epochs=2000.
# l2_comm_scale=0. Same --seed / graph seed so the random diagram is shared.

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

REPO=/mmfs1/home/wong2/repos/dgn
RUNS_ROOT=/mmfs1/gscratch/golub/wong2/runs/multi_task/l2_weight_runs
SEED=0
TASK=dly_go
L2S=(0 1e-5 1e-4 1e-3 1e-2)
L2_START=0
L2_INCREASE=200
L2_INIT=0.3
STAMP="$(date +%y%m%d%H%M)"

mkdir -p "${RUNS_ROOT}"
mkdir -p /mmfs1/gscratch/golub/wong2/slurm_logs

cd "${REPO}"

echo "Job ${SLURM_JOB_ID}: launching h=64 and h=128 l2_scale sweeps in parallel on 1 GPU"
echo "L2 ramp: start=${L2_START} increase=${L2_INCREASE} init=${L2_INIT}"

i=0
for HS in 64 128; do
  if [[ "${HS}" == 64 ]]; then
    EXP=multi_task
  else
    EXP=multi_task_h128
  fi

  for L2 in "${L2S[@]}"; do
    RUN_NAME="multi_task_${TASK}_h${HS}_l2${L2}_l2c0_l2inc${L2_INCREASE}_gseed0_seed${SEED}_id${STAMP}"
    RUN_DIR="${RUNS_ROOT}/${RUN_NAME}"

    echo "Launching ${EXP} ${TASK} h=${HS} l2_scale=${L2} l2_comm_scale=0 -> ${RUN_DIR}"
    python "${REPO}/scripts/generate_data.py" "${EXP}" \
      --seed "${SEED}" \
      --run_name "${RUN_DIR}" \
      --override "datamodule.task_names=[${TASK}]" \
      --override "model.l2_scale=${L2}" \
      --override "model.l2_start=${L2_START}" \
      --override "model.l2_increase=${L2_INCREASE}" \
      --override "model.l2_init=${L2_INIT}" \
      --override "model.l2_comm_scale=0" \
      --override "model.graph_kwargs.seed=0" &

    i=$((i + 1))
  done
done

wait
echo "Finished all ${i} runs under ${RUNS_ROOT}"
