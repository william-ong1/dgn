#!/bin/bash
#SBATCH --job-name=mn_decode
#SBATCH --partition=gpu-l40s
#SBATCH --account=golub
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=out/mn_decode.txt
#SBATCH --error=err/mn_decode.txt

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

REPO=/mmfs1/home/wong2/repos/dgn
RUNS_ROOT=/mmfs1/gscratch/golub/wong2/runs

# Saturate the allocated CPUs through BLAS (sklearn Ridge solve is BLAS-heavy).
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p out err

python "${REPO}/pd_compute_decodability.py" "$RUNS_ROOT" \
  --output "$RUNS_ROOT/decodability_results.csv"

echo "Done. CSV is at $RUNS_ROOT/decodability_results.csv"
