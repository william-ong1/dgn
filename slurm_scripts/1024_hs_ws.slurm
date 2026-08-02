#!/bin/bash

#SBATCH --job-name=1024_hs_ws_sweep
#SBATCH --partition=gpu-l40s
#SBATCH --account=golub
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --output=out/1024_hs_ws_sweep.txt
#SBATCH --error=err/1024_hs_ws_sweep.txt

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

REPO=/mmfs1/home/wong2/repos/dgn

python "${REPO}/scripts/generate_data.py" memory_network \
	--hidden_size 1024 \
      --input_weight_init_var_scale 1.0 \
    --run_name /gscratch/golub/wong2/runs/memory_network_h1024_ws1.0_id2606221443 \
    --ckpt_path /gscratch/golub/wong2/runs/memory_network_h1024_ws1.0_id2606221443/lightning_checkpoints/last.ckpt &

python "${REPO}/scripts/generate_data.py" memory_network \
	--hidden_size 1024 \
      --input_weight_init_var_scale 1.5 \
    --run_name /gscratch/golub/wong2/runs/memory_network_h1024_ws1.5_id2606221443 \
    --ckpt_path /gscratch/golub/wong2/runs/memory_network_h1024_ws1.5_id2606221443/lightning_checkpoints/last.ckpt &

python "${REPO}/scripts/generate_data.py" memory_network \
	--hidden_size 1024 \
      --input_weight_init_var_scale 2.25 \
    --run_name /gscratch/golub/wong2/runs/memory_network_h1024_ws2.25_id2606221443 \
    --ckpt_path /gscratch/golub/wong2/runs/memory_network_h1024_ws2.25_id2606221443/lightning_checkpoints/last.ckpt &

wait
echo "All sweep runs done."
