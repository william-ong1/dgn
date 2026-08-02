#!/bin/bash
#SBATCH --job-name=mrlfads_metrics_mn_1
#SBATCH --partition=gpu-l40s
#SBATCH --account=cse
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=out/mrlfads_metrics_mn_1.txt
#SBATCH --error=err/mrlfads_metrics_mn_1.txt

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

export PYTHONPATH="/mmfs1/home/wong2/repos/mrlfads2:${PYTHONPATH}"

python scripts/batch_mrlfads_eval.py \
	  --runs-dir /gscratch/golub/wong2/mrlfads_runs \
	    --run-glob 'mn_pois_1*' \
	      --input-h5 /mmfs1/gscratch/golub/wong2/runs/noise_runs/memory_network_h512_n0.5_id2606221443/data.h5 \
	        --truth-h5 /mmfs1/gscratch/golub/wong2/runs/noise_runs/memory_network_h512_n0.5_id2606221443/data.h5 \
		  --config-dir configs/pass_decision \
		    --output-dir /gscratch/golub/wong2/mrlfads_runs/eval_outputs/mn_pois_1 \
		      --summary-csv /gscratch/golub/wong2/mrlfads_runs/eval_outputs/mn_pois_1/evaluation_summary_all_runs.csv \
		        --experiment-type pass_decision \
			  --rates-truth-h5 /gscratch/golub/wong2/runs/noise_runs/memory_network_h512_n0.5_id2606221443/gaussian.h5 \
			    --output-dist poisson \
			      --truth-time-start 10 \
			        --accelerator gpu \
				  --poisson-rate-max 40.0 \
				    --poisson-dt 0.01
