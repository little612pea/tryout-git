#!/bin/bash
#SBATCH -o out.txt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p A100
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1 

CUDA_VISIBLE_DEVICES=1 python3 train.py