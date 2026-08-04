#!/bin/bash
#SBATCH --job-name=mt_hs_nz
#SBATCH --account=golub
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_hs_nz_%j.out
#SBATCH --error=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_hs_nz_%j.err

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

REPO=/mmfs1/home/wong2/repos/dgn
RUNS_ROOT=/mmfs1/gscratch/golub/wong2/runs/multi_task/hs_noise_runs
SEED=0

# Single-task sweep: rt_go only
TASK=rt_go
HS=(64 128 256 512)
NZ=(0.0 0.001 0.01 0.05)

STAMP="$(date +%y%m%d%H%M)"

mkdir -p "${RUNS_ROOT}"
mkdir -p /mmfs1/gscratch/golub/wong2/slurm_logs

cd "${REPO}"

echo "Job ${SLURM_JOB_ID}: launching ${#HS[@]} x ${#NZ[@]} runs on 1 GPU"

i=0
for H in "${HS[@]}"; do
  for N in "${NZ[@]}"; do
    RUN_NAME="multi_task_${TASK}_h${H}_n${N}_seed${SEED}_id${STAMP}"
    RUN_DIR="${RUNS_ROOT}/${RUN_NAME}"

    echo "Launching ${TASK} h=${H} n=${N} -> ${RUN_DIR}"
    python "${REPO}/scripts/generate_data.py" multi_task \
      --seed "${SEED}" \
      --hidden_size "${H}" \
      --noise "${N}" \
      --run_name "${RUN_DIR}" \
      --override "datamodule.task_names=[${TASK}]" &

    i=$((i + 1))
  done
done

wait
echo "Finished all ${i} runs under ${RUNS_ROOT}"
