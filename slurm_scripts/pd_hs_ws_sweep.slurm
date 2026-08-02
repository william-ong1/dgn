#!/bin/bash

#SBATCH --job-name=pd_hs_ws_sweep
#SBATCH --partition=gpu-l40s
#SBATCH --account=golub
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=out/pd_hs_ws_sweep.txt
#SBATCH --error=err/pd_hs_ws_sweep.txt

set -euo pipefail

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

REPO=/mmfs1/home/wong2/repos/dgn
RUNS_ROOT=/mmfs1/gscratch/golub/wong2/runs/pass_decision
mkdir -p "$RUNS_ROOT" out err

STAMP=$(date +%y%m%d%H%M)
MAX_CONCURRENT=50

HIDDEN_SIZES=(32 64 128 256 512)
WEIGHT_SCALES=(1.0 1.5 2.0)
SEEDS=(0 42)

for H in "${HIDDEN_SIZES[@]}"; do
	for WS in "${WEIGHT_SCALES[@]}"; do
		for SEED in "${SEEDS[@]}"; do
	      		RUN_DIR="${RUNS_ROOT}/pass_decision_h${H}_ws${WS}_seed${SEED}_id${STAMP}"
	    		LOG="out/mn_h${H}_ws${WS}_seed${SEED}.log"

			echo "Launching h=${H} ws=${WS} seed=${SEED} -> ${RUN_DIR}"
			python "${REPO}/generate_data.py" pass_decision \
		      		--hidden_size "$H" \
				--input_weight_init_var_scale "$WS" \
				--run_name "$RUN_DIR" \
				--seed "$SEED" \
				> "$LOG" 2>&1 &

    			while [ "$(jobs -r | wc -l)" -ge "$MAX_CONCURRENT" ]; do
      				sleep 10
    			done
  		done
	done	
done

wait
echo "All sweep runs done."
