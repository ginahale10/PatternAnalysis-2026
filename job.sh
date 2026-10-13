#!/bin/bash
#SBATCH --job-name=demo-check
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=results.out
#SBATCH --error=results.err
#SBATCH --cpus-per-task=4
# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate torch
conda install matplotlib
# Run the script
python3 basic-cnn.py
