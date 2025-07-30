#!/bin/bash
#SBATCH --job-name=pronet_vae_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/pronet_vae_eval_%j.out
#SBATCH --error=logs/pronet_vae_eval_%j.err

# Load conda environment
source ~/.bashrc
conda activate /net/scratch/caom/dplm_env

# Set PYTHONPATH for OpenFold CUDA extensions
export PYTHONPATH=/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:$PYTHONPATH

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the evaluation script
python evaluate_simple.py \
    --config ../configs/pronet_vae.yaml \
    --checkpoint checkpoints/pronet_vae-epoch=02-val_loss=579.6155.ckpt \
    --data_path /home/caom/AID3/protein-dit/data/protein_train/processed/protein_test_split.pt \
    --output evaluation_results.txt \
    --batch_size 4 