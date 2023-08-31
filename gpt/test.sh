#!/bin/bash
#SBATCH -o out.txt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p A100
#SBATCH --cpus-per-task=24
#SBATCH --gpus=1 

ssh GPU08
cd ~/lenet/gpt
conda activate mytorch
accelerate launch train_acce.py