#!/bin/bash
#SBATCH --job-name=mrlfads_noa0_ctxt_dm_1
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/gscratch/golub/wong2/slurm_logs/mrlfads_noa0_ctxt_dm_1_%j.out
#SBATCH --error=/mmfs1/gscratch/golub/wong2/slurm_logs/mrlfads_noa0_ctxt_dm_1_%j.err

# ctxt_dm_1 region holdout: A0 (fixation) unobserved. A1-A3 fully observed.
# KL (co, com): (3.0, 0.1), (1.0, 0.1), (0.1, 0.01), (0.01, 0.001).
# Source HDF5 files are read-only.

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

export PYTHONPATH="/mmfs1/home/wong2/repos/mrlfads2:${PYTHONPATH:-}"

REPO=/mmfs1/home/wong2/repos/dgn
CFG="${REPO}/partial_obs_configs/ctxt_dm_1"
mkdir -p /mmfs1/gscratch/golub/wong2/slurm_logs
cd "${REPO}"

echo "Job ${SLURM_JOB_ID}"
echo "Launching ctxt_dm_1 A0-unobserved trains"

python "${CFG}/kl3/exec.py" &
python "${CFG}/kl1/exec.py" &
python "${CFG}/kl01/exec.py" &
python "${CFG}/kl001/exec.py" &

wait
echo "Done."
