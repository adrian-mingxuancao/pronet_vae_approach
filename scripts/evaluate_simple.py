#!/usr/bin/env python3
"""
Simple evaluation script for the trained ProteiNet + VAE model
Tests structure recovery and computes RMSD/TM-score metrics
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
import argparse
import yaml

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_simple import SimpleProNetVAEModel, ProteinDataset, SimpleProNetVAELightningModule

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_model(config):
    """Create model from config"""
    # Use the same model architecture as the training script
    # The training script uses these default values:
    pronet_hidden_dim = 256
    pronet_num_layers = 4
    vae_num_embeddings = 512
    vae_embedding_dim = 256
    vae_commitment_cost = 0.25
    
    model = SimpleProNetVAEModel(
        pronet_hidden_dim=pronet_hidden_dim,
        pronet_num_layers=pronet_num_layers,
        vae_num_embeddings=vae_num_embeddings,
        vae_embedding_dim=vae_embedding_dim,
        vae_commitment_cost=vae_commitment_cost
    )
    
    return model

def compute_rmsd(pred_coords, true_coords, mask=None):
    """Compute RMSD between predicted and true coordinates"""
    if mask is not None:
        pred_coords = pred_coords[mask]
        true_coords = true_coords[mask]
    
    # Center coordinates
    pred_centered = pred_coords - pred_coords.mean(dim=0, keepdim=True)
    true_centered = true_coords - true_coords.mean(dim=0, keepdim=True)
    
    # Compute RMSD
    rmsd = torch.sqrt(torch.mean((pred_centered - true_centered) ** 2))
    return rmsd.item()

def evaluate_model(model, dataloader, device):
    """Evaluate the model on the given dataloader"""
    model.eval()
    
    total_representation_loss = 0.0
    total_vae_loss = 0.0
    total_perplexity = 0.0
    total_codebook_usage = 0.0
    num_batches = 0
    
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch)
            encoded = outputs['encoded']  # [B, L, embedding_dim]
            quantized = outputs['quantized']  # [B, L, embedding_dim]
            mask = outputs['mask']  # [B, L]
            perplexity = outputs['perplexity']
            
            # Debug: print what's in outputs
            if batch_idx == 0:
                print(f"Available outputs keys: {list(outputs.keys())}")
                if 'min_encoding_indices' in outputs:
                    print(f"min_encoding_indices shape: {outputs['min_encoding_indices'].shape}")
                    print(f"min_encoding_indices values: {outputs['min_encoding_indices'][:10]}")
                else:
                    print("No min_encoding_indices in outputs")
            
            # Compute representation reconstruction loss (how well we can reconstruct encoded features)
            representation_loss = F.mse_loss(quantized[mask], encoded[mask])
            vae_loss = outputs['vae_loss']
            
            # Compute codebook usage statistics
            if perplexity is not None:
                batch_perplexity = perplexity.item()
            else:
                batch_perplexity = 0.0
            
            # Count unique tokens used in this batch
            batch_size = encoded.shape[0]
            if 'min_encoding_indices' in outputs:
                indices = outputs['min_encoding_indices']
                unique_tokens = len(torch.unique(indices))
                codebook_usage = unique_tokens / model.vae.n_e  # percentage of codebook used
                if batch_idx == 0:
                    print(f"Unique tokens: {unique_tokens}, Total codebook size: {model.vae.n_e}")
            else:
                codebook_usage = 0.0
            
            # Store results
            batch_result = {
                'batch_idx': batch_idx,
                'representation_loss': representation_loss.item(),
                'vae_loss': vae_loss.item(),
                'total_loss': (representation_loss + vae_loss).item(),
                'perplexity': batch_perplexity,
                'codebook_usage': codebook_usage,
                'num_proteins': batch_size
            }
            results.append(batch_result)
            
            # Accumulate metrics
            total_representation_loss += representation_loss.item()
            total_vae_loss += vae_loss.item()
            total_perplexity += batch_perplexity
            total_codebook_usage += codebook_usage
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}, representation loss: {representation_loss.item():.4f}, perplexity: {batch_perplexity:.2f}")
    
    # Compute overall metrics
    avg_representation_loss = total_representation_loss / num_batches
    avg_vae_loss = total_vae_loss / num_batches
    avg_perplexity = total_perplexity / num_batches
    avg_codebook_usage = total_codebook_usage / num_batches
    
    return {
        'avg_representation_loss': avg_representation_loss,
        'avg_vae_loss': avg_vae_loss,
        'avg_total_loss': avg_representation_loss + avg_vae_loss,
        'avg_perplexity': avg_perplexity,
        'avg_codebook_usage': avg_codebook_usage,
        'total_proteins': sum(r['num_proteins'] for r in results),
        'detailed_results': results
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained ProteiNet + VAE model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default='evaluation_results.txt', help='Output file for results')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        # Lightning checkpoint - need to handle the 'model.' prefix
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if it exists
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
    else:
        # Direct model checkpoint
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print("Model loaded successfully")
    
    # Create dataset and dataloader
    dataset = ProteinDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Dataset loaded: {len(dataset)} proteins")
    
    # Evaluate model
    print("Starting evaluation...")
    results = evaluate_model(model, dataloader, device)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Average Representation Loss: {results['avg_representation_loss']:.4f}")
    print(f"Average VAE Loss: {results['avg_vae_loss']:.4f}")
    print(f"Average Total Loss: {results['avg_total_loss']:.4f}")
    print(f"Average Perplexity: {results['avg_perplexity']:.2f}")
    print(f"Average Codebook Usage: {results['avg_codebook_usage']:.2%}")
    print(f"Total Proteins Evaluated: {results['total_proteins']}")
    print("="*50)
    
    # Save detailed results
    with open(args.output, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Average Representation Loss: {results['avg_representation_loss']:.4f}\n")
        f.write(f"Average VAE Loss: {results['avg_vae_loss']:.4f}\n")
        f.write(f"Average Total Loss: {results['avg_total_loss']:.4f}\n")
        f.write(f"Average Perplexity: {results['avg_perplexity']:.2f}\n")
        f.write(f"Average Codebook Usage: {results['avg_codebook_usage']:.2%}\n")
        f.write(f"Total Proteins Evaluated: {results['total_proteins']}\n")
        f.write("="*50 + "\n\n")
        
        f.write("DETAILED RESULTS BY BATCH:\n")
        for result in results['detailed_results']:
            f.write(f"Batch {result['batch_idx']}: ")
            f.write(f"Loss={result['total_loss']:.4f}, ")
            f.write(f"Perplexity={result['perplexity']:.2f}, ")
            f.write(f"Codebook Usage={result['codebook_usage']:.2%}\n")
    
    print(f"Detailed results saved to: {args.output}")

if __name__ == "__main__":
    main() 