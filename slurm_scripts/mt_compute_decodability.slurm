#!/bin/bash
#SBATCH --job-name=mt_decode
#SBATCH --account=golub
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_decode_%A_%a.out
#SBATCH --error=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_decode_%A_%a.err

# One array task per cognitive task type (4 total):
#   0=rt_go  1=rt_go_anti  2=dly_go  3=dly_go_anti
# CPU-only. Change --partition if cpu-g2 is unavailable.

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

REPO=/mmfs1/home/wong2/repos/dgn
RUNS_ROOT=/mmfs1/gscratch/golub/wong2/runs/multi_task

TASKS=(rt_go rt_go_anti dly_go dly_go_anti)
# Patterns relative to RUNS_ROOT; rt_go_h* avoids matching rt_go_anti (same for dly).
PATTERNS=(
  "rt_go/multi_task_rt_go_h*"
  "rt_go_anti/multi_task_rt_go_anti_*"
  "dly_go/multi_task_dly_go_h*"
  "dly_go_anti/multi_task_dly_go_anti_*"
)

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
PATTERN="${PATTERNS[$SLURM_ARRAY_TASK_ID]}"
OUT_CSV="${RUNS_ROOT}/${TASK}/mt_decodability_results.csv"

mkdir -p /mmfs1/gscratch/golub/wong2/slurm_logs
cd "${REPO}"

echo "Job ${SLURM_JOB_ID} array ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Task:      ${TASK}"
echo "Repo:      ${REPO}"
echo "Runs root: ${RUNS_ROOT}"
echo "Pattern:   ${PATTERN}"
echo "Output:    ${OUT_CSV}"
echo "Matching runs:"
n_match=0
for d in "${RUNS_ROOT}"/${PATTERN}; do
  if [[ -d "${d}" && -f "${d}/data.h5" ]]; then
    echo "  ${d}"
    n_match=$((n_match + 1))
  fi
done
echo "Total runs with data.h5: ${n_match}"

python "${REPO}/scripts/mt_compute_decodability.py" "${RUNS_ROOT}" \
  --pattern "${PATTERN}" \
  --output "${OUT_CSV}" \
  --seed 0

echo "Done. Wrote ${OUT_CSV}"
