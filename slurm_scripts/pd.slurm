#!/bin/bash

#SBATCH --job-name=pd_data_poisson
#SBATCH --partition=gpu-l40s
#SBATCH --account=golub
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=out/pd_data.txt
#SBATCH --error=err/pd_data.txt


source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

python /mmfs1/home/wong2/repos/dgn/scripts/generate_data.py pass_decision
