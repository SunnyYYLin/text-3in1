#!/bin/bash
#SBATCH --job-name=text3in1
#SBATCH --gpus=2
#SBATCH --output=slurm_logs/%j.log

module purge
module load anaconda3/2024.02-1
module load cuda/12.4
module load cudnn/8.9.7_cuda12.x
source /data/apps/conda/2024.02-1/etc/profile.d/conda.sh
conda activate text3in1

cd ~/run/text-3in1
python test.py
