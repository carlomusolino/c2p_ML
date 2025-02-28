#!/bin/bash

#SBATCH --job-name=training
#SBATCH --output=training.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=32G

source /mnt/rafast/musolino/pyenv/numrel/bin/activate 


python train.py --d_ff $1 --model_name $2 