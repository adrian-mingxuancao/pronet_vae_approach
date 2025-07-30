#!/bin/bash
#SBATCH --job-name=pronet_vae_fixed
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pronet_vae_fixed_%j.out
#SBATCH --error=logs/pronet_vae_fixed_%j.err

# Load conda environment
source ~/.bashrc
conda activate /net/scratch/caom/dplm_env

# Set PYTHONPATH for OpenFold CUDA extensions
export PYTHONPATH=/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:$PYTHONPATH

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the simplified training script (epochs controlled by config file)
python train_simple.py --config ../configs/pronet_vae.yaml 