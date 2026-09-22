#!/bin/bash
#SBATCH --job-name=mt_rank_pois_unrel
#SBATCH --account=golub
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_rank_pois_unrel_%j.out
#SBATCH --error=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_rank_pois_unrel_%j.err

# Rank the two IRCB-26-unreleased Yang tasks (ctxt_dm_max, dly_dm_max)
# among h64 MultiTaskNet runs under l2_random_runs.
# CPU-only (reads data.h5 + TensorBoard; no GPU needed).

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

export PYTHONPATH="/mmfs1/home/wong2/repos/mrlfads2:${PYTHONPATH:-}"

REPO=/mmfs1/home/wong2/repos/dgn
RUNS_ROOT=/gscratch/golub/wong2/runs/multi_task/l2_random_runs
OUT_CSV="${RUNS_ROOT}/mt_poisson_dataset_pairs_unreleased.csv"

mkdir -p /mmfs1/gscratch/golub/wong2/slurm_logs
cd "${REPO}"

echo "Job ${SLURM_JOB_ID}"
echo "Repo:      ${REPO}"
echo "Runs root: ${RUNS_ROOT}"
echo "Tasks:     unreleased (ctxt_dm_max dly_dm_max)"
echo "Hidden:    64"
echo "Output:    ${OUT_CSV}"

python scripts/mt_rank_poisson_dataset.py "${RUNS_ROOT}" \
  --tasks unreleased \
  --hidden-size 64 \
  --output "${OUT_CSV}" \
  --seed 0

echo "Done. Wrote ${OUT_CSV}"
echo "Ranking CSV: ${RUNS_ROOT}/mt_poisson_dataset_unreleased_ranking.csv"
