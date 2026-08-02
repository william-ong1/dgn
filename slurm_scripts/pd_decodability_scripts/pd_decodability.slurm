#!/bin/bash
#SBATCH --job-name=pd_noise_decodability
#SBATCH --partition=gpu-l40s
#SBATCH --account=cse
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=out/pd_noise_decode.txt
#SBATCH --error=err/pd_noise_decode.txt

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

REPO=/mmfs1/home/wong2/repos/dgn
RUNS_ROOT=/mmfs1/gscratch/golub/wong2/runs/pass_decision/noise_runs

# Saturate the allocated CPUs through BLAS (sklearn Ridge solve is BLAS-heavy).
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p out err

python "${REPO}/scripts/pd_compute_decodability.py" "$RUNS_ROOT" \
  --pattern "pass_decision_h256_n0.001_seed0_id2607210153" \
  --output "$RUNS_ROOT/decodability_results.csv"

echo "Done. CSV is at $RUNS_ROOT/decodability_results.csv"
