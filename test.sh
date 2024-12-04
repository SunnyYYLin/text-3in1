#!/bin/bash
#SBATCH --job-name=text3in1-test
#SBATCH --gpus=1
#SBATCH --output=slurm_logs/%j.log

module purge
module load anaconda3/2024.02-1
module load cuda/12.4
module load cudnn/8.9.7_cuda12.x
source /data/apps/conda/2024.02-1/etc/profile.d/conda.sh
conda activate text3in1

cd ~/run/text-3in1
python main.py --mode="example" --task="translation" \
    --model_dir="checkpoints/translation/Transformer_layers2_ffnsize1024_heads4_emb256_dropout0.2"
