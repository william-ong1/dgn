#!/bin/bash

#SBATCH --job-name=pass_decision
#SBATCH --partition=gpu-l40s
#SBATCH --account=cse
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --output=pass_decision_stdout.txt
#SBATCH --error=pass_decision_stderr.txt


source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

python /mmfs1/home/wong2/repos/dgn/generate_data.py pass_decision