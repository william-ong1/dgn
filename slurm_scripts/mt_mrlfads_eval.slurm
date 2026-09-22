#!/bin/bash
#SBATCH --job-name=mt_mrlfads_eval
#SBATCH --account=golub
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_mrlfads_eval_%j.out
#SBATCH --error=/mmfs1/gscratch/golub/wong2/slurm_logs/mt_mrlfads_eval_%j.err

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

export PYTHONPATH="/mmfs1/home/wong2/repos/mrlfads2:${PYTHONPATH:-}"

REPO=/mmfs1/home/wong2/repos/dgn
RUNS=/gscratch/golub/wong2/mrlfads_runs
OUT=/gscratch/golub/wong2/mrlfads_runs/mt_eval_outputs

mkdir -p /mmfs1/gscratch/golub/wong2/slurm_logs "${OUT}"
cd "${REPO}"

python scripts/batch_mrlfads_eval.py \
  --runs-dir "${RUNS}" \
  --experiment-type multi_task \
  --output-dist poisson \
  --resolve-data-from-run \
  --run-glob '*_kl*' \
  --accelerator gpu \
  --output-dir "${OUT}" \
  --summary-csv "${OUT}/evaluation_summary_all_runs.csv" \
  --skip-existing
