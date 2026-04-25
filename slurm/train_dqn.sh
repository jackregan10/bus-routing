#!/bin/bash
#SBATCH --job-name=train-bus-routing
#SBATCH -t 36:00:00
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --output=slurm-logs/dqn%j.out
#SBATCH --error=slurm-logs/dqn%j.err

hostname && nvidia-smi

echo Currently processing training loop.

cd /home/users/jfr29/bus-routing/src

python3 -u train_dqn.py