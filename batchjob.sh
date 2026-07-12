#!/usr/bin/env bash
#
# ===== Slurm resource request (edit these for your cluster) =====
#SBATCH --job-name=mod_exp              # a short name for the job
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                    # one GPU
#SBATCH --account="bigdata"            # <-- your Slurm account
#SBATCH --partition=h200_normal_q      # <-- your GPU partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12             # keep >= num_workers in main.py
#SBATCH --time=7:00:00
#SBATCH --output=mod_exp_%j.out        # stdout+stderr goes here
# ===== end of request =====

# 1. set up your environment
module load Miniconda3                 # <-- your cluster's conda module
source activate ~/env/pmod             # <-- path to your conda env

# 2. point at the DeepSense 6G data (the Multi_Modal folder)
export DEEPSENSE_ROOT=/path/to/deepsense/Multi_Modal   # <-- EDIT ME

# 3. go to the repo (the directory this job was submitted from)
cd "$SLURM_SUBMIT_DIR"

# 4. run training (SEED defaults to 0 if not exported at submit time)
python main.py --dataset d6g --seed "${SEED:-0}" --n_layers 8
