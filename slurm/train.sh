#!/bin/bash
#SBATCH --job-name=train-bus-routing
#SBATCH -t 36:00:00  # time requested in hour:minute:second
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --output=slurm-logs/train_%j.out
#SBATCH --error=slurm-logs/train_%j.err

hostname && nvidia-smi

echo Currently processing training loop.

cd /home/users/jfr29/bus-routing
pip install --upgrade typing_extensions torch gymnasium

python3 -u -m src.train