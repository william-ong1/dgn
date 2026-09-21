#!/bin/bash
  
#SBATCH --job-name=mt_fit_dms_nogo
#SBATCH --partition=gpu-l40s
#SBATCH --account=golub
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=mt_fit_dms_nogo_stdout.txt
#SBATCH --error=mt_fit_dms_nogo_stderr.txt
#SBATCH --cpus-per-task=8

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

export PYTHONPATH="/mmfs1/home/wong2/repos/mrlfads2:${PYTHONPATH}"

python /mmfs1/home/wong2/repos/mrlfads2/code/mt_pois_configs/dms_nogo/kl3/exec.py &
python /mmfs1/home/wong2/repos/mrlfads2/code/mt_pois_configs/dms_nogo/kl1/exec.py &
python /mmfs1/home/wong2/repos/mrlfads2/code/mt_pois_configs/dms_nogo/kl01/exec.py &
python /mmfs1/home/wong2/repos/mrlfads2/code/mt_pois_configs/dms_nogo/kl001/exec.py &
python /mmfs1/home/wong2/repos/mrlfads2/code/mt_pois_configs/dms_nogo/kl0001/exec.py &

wait