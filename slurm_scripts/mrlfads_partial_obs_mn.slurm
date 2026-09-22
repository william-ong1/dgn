#!/bin/bash
#SBATCH --job-name=mrlfads_p50_mn
#SBATCH --account=golub
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --output=/mmfs1/gscratch/golub/wong2/slurm_logs/mrlfads_p50_mn_%j.out
#SBATCH --error=/mmfs1/gscratch/golub/wong2/slurm_logs/mrlfads_p50_mn_%j.err

# Delayed-memory 50% observation. KL (co, com): (1, 0.1), (0.1, 0.01), (0.01, 0.001).
# Source HDF5 files are read-only.

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

export PYTHONPATH="/mmfs1/home/wong2/repos/mrlfads2:${PYTHONPATH:-}"

REPO=/mmfs1/home/wong2/repos/dgn
CFG="${REPO}/partial_obs_configs/mn_pois"
mkdir -p /mmfs1/gscratch/golub/wong2/slurm_logs
cd "${REPO}"

echo "Job ${SLURM_JOB_ID}"
echo "Launching MN partial-obs trains"

python "${CFG}/kl1/exec.py" &
python "${CFG}/kl01/exec.py" &
python "${CFG}/kl001/exec.py" &

wait
echo "Done."
