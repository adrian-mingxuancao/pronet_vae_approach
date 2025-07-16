#!/usr/bin/env python3
"""
Demo script for ProteiNet + VAE + ESM-Fold Structure Tokenization Model

This script demonstrates the new structure tokenization pipeline:
1. ProteiNet: Structure -> Continuous embeddings (replaces GVP)
2. VAE: Continuous embeddings -> Discrete tokens (replaces LFQ)
3. ESM-Fold: Discrete tokens -> Structure (same as DPLM-2)

Usage:
    python demo.py --config configs/pronet_vae.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.structok_pronet_vae import ProNetVAEModel


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_sample_batch(batch_size=2, seq_len=100, device='cpu'):
    """Create a sample batch for testing"""
    # Create random atom positions (atom37 format)
    atom_positions = torch.randn(batch_size, seq_len, 37, 3, device=device)
    
    # Create atom mask (all atoms exist)
    atom_mask = torch.ones(batch_size, seq_len, 37, device=device)
    
    # Create amino acid types (random)
    aatype = torch.randint(0, 20, (batch_size, seq_len), device=device)
    
    # Create residue mask (all residues exist)
    res_mask = torch.ones(batch_size, seq_len, device=device)
    
    # Create sequence length
    seq_length = torch.full((batch_size,), seq_len, device=device)
    
    batch = {
        "all_atom_positions": atom_positions,
        "all_atom_mask": atom_mask,
        "aatype": aatype,
        "res_mask": res_mask,
        "seq_length": seq_length,
    }
    
    return batch


def demo_tokenization(model, batch):
    """Demonstrate the tokenization process"""
    print("=== Structure Tokenization Demo ===")
    
    # Step 1: Encode with ProteiNet
    print("\n1. ProteiNet Encoding...")
    with torch.no_grad():
        pre_quant, encoder_feats = model.encode(
            atom_positions=batch["all_atom_positions"],
            mask=batch["res_mask"],
            seq_length=batch["seq_length"],
        )
    
    print(f"   Input shape: {batch['all_atom_positions'].shape}")
    print(f"   Encoder output shape: {encoder_feats.shape}")
    print(f"   Pre-quantization shape: {pre_quant.shape}")
    
    # Step 2: Quantize with VAE
    print("\n2. VAE Quantization...")
    with torch.no_grad():
        quant, loss, (perplexity, min_encodings, struct_tokens) = model.quantize(
            pre_quant, mask=batch["res_mask"].bool()
        )
    
    print(f"   Quantized shape: {quant.shape}")
    print(f"   Structure tokens shape: {struct_tokens.shape}")
    print(f"   VAE loss: {loss.item():.4f}")
    print(f"   Perplexity: {perplexity.item():.4f}")
    print(f"   Unique tokens: {struct_tokens.unique().numel()}")
    
    # Step 3: Decode with ESM-Fold
    print("\n3. ESM-Fold Decoding...")
    with torch.no_grad():
        decoder_out = model.decode(
            quant=quant,
            aatype=batch["aatype"],
            mask=batch["res_mask"],
        )
    
    print(f"   Decoder output keys: {list(decoder_out.keys())}")
    if "final_atom_positions" in decoder_out:
        print(f"   Reconstructed positions shape: {decoder_out['final_atom_positions'].shape}")
    
    return struct_tokens, decoder_out


def demo_detokenization(model, struct_tokens, batch):
    """Demonstrate the detokenization process"""
    print("\n=== Structure Detokenization Demo ===")
    
    with torch.no_grad():
        reconstructed = model.detokenize(
            struct_tokens=struct_tokens,
            res_mask=batch["res_mask"],
        )
    
    print(f"   Input tokens shape: {struct_tokens.shape}")
    print(f"   Reconstructed positions shape: {reconstructed['atom37_positions'].shape}")
    print(f"   Reconstructed mask shape: {reconstructed['atom37_mask'].shape}")
    
    return reconstructed


def compute_reconstruction_metrics(original, reconstructed):
    """Compute reconstruction quality metrics"""
    print("\n=== Reconstruction Metrics ===")
    
    # Extract positions
    orig_pos = original["all_atom_positions"]
    recon_pos = reconstructed["atom37_positions"]
    
    # Extract masks
    orig_mask = original["all_atom_mask"]
    recon_mask = reconstructed["atom37_mask"]
    
    # Combined mask
    mask = orig_mask & recon_mask
    
    if mask.sum() == 0:
        print("   No valid atoms for comparison")
        return
    
    # Compute RMSD
    squared_diff = (orig_pos - recon_pos) ** 2
    squared_diff = squared_diff.sum(dim=-1)  # Sum over xyz dimensions
    
    masked_diff = squared_diff * mask.float()
    rmsd = torch.sqrt(masked_diff.sum() / (mask.sum() + 1e-8))
    
    print(f"   RMSD: {rmsd.item():.4f} Å")
    
    # Compute per-residue RMSD
    ca_mask = mask[:, :, 1]  # CA atoms
    if ca_mask.sum() > 0:
        ca_orig = orig_pos[:, :, 1]  # CA positions
        ca_recon = recon_pos[:, :, 1]
        
        ca_squared_diff = (ca_orig - ca_recon) ** 2
        ca_squared_diff = ca_squared_diff.sum(dim=-1)
        
        ca_masked_diff = ca_squared_diff * ca_mask.float()
        ca_rmsd = torch.sqrt(ca_masked_diff.sum() / (ca_mask.sum() + 1e-8))
        
        print(f"   CA RMSD: {ca_rmsd.item():.4f} Å")


def save_sample_output(decoder_out, output_dir="demo_output"):
    """Save sample output structures"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n=== Saving Output ===")
    print(f"   Output directory: {output_path}")
    
    # Save as numpy arrays for now
    # In practice, you'd save as PDB files
    np.save(output_path / "final_atom_positions.npy", 
            decoder_out["final_atom_positions"].cpu().numpy())
    np.save(output_path / "final_atom_mask.npy", 
            decoder_out["final_atom_mask"].cpu().numpy())
    
    print(f"   Saved atom positions and masks")


def main():
    parser = argparse.ArgumentParser(description='Demo ProteiNet + VAE + ESM-Fold model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=50, help='Sequence length')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create model
    model_config = config['model']
    model = ProNetVAEModel(**model_config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    
    model.to(device)
    model.eval()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create sample batch
    batch = create_sample_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device
    )
    
    # Run tokenization demo
    struct_tokens, decoder_out = demo_tokenization(model, batch)
    
    # Run detokenization demo
    reconstructed = demo_detokenization(model, struct_tokens, batch)
    
    # Compute metrics
    compute_reconstruction_metrics(batch, reconstructed)
    
    # Save output
    save_sample_output(decoder_out)
    
    print("\n=== Demo Completed Successfully! ===")


if __name__ == '__main__':
    main() 