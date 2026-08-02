#!/bin/bash
#SBATCH --job-name=mrlfads_metrics_full_seed1337
#SBATCH --partition=gpu-l40s
#SBATCH --account=cse
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=out/mrlfads_metrics_full_seed1337.txt
#SBATCH --error=err/mrlfads_metrics_full_seed1337.txt

source /gscratch/golub/wong2/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/golub/wong2/mr-lfads

export PYTHONPATH="/mmfs1/home/wong2/repos/mrlfads2:${PYTHONPATH}"

python batch_mrlfads_eval.py \
	  --runs-dir /gscratch/golub/wong2/mrlfads_runs \
	    --run-glob 'full_model_seed1337*' \
	      --input-h5 /gscratch/golub/wong2/runs/pass_decision/noise_runs/pass_decision_h256_n0.001_seed0_id2607210153/data.h5 \
	        --truth-h5 /gscratch/golub/wong2/runs/pass_decision/noise_runs/pass_decision_h256_n0.001_seed0_id2607210153/data.h5 \
		  --config-dir configs/pass_decision \
		    --output-dir /gscratch/golub/wong2/mrlfads_runs/eval_outputs/full_model_seed1337 \
		      --summary-csv /gscratch/golub/wong2/mrlfads_runs/eval_outputs/full_model_seed1337/evaluation_summary_all_runs.csv \
		        --experiment-type pass_decision \
			  --rates-truth-h5 /gscratch/golub/wong2/runs/pass_decision/noise_runs/pass_decision_h256_n0.001_seed0_id2607210153/gaussian.h5 \
			    --output-dist poisson \
			      --truth-time-start 10 \
			        --accelerator gpu \
				  --poisson-rate-max 40.0 \
				    --poisson-dt 0.01
