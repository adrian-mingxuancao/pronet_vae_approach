#!/usr/bin/env python3
"""
Simplified training script for ProteiNet + VAE (without ESM-Fold decoder)

This script tests the ProteiNet encoder and VAE discretization parts
without the complex ESM-Fold decoder that has OpenFold dependencies.

Usage:
    python train_simple.py --config configs/pronet_vae.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import random
from pathlib import Path
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.pronet_converter import ProNetConverter
from modules.vqvae import VectorQuantizer2

class NaNStoppingCallback(pl.Callback):
    """Callback to stop training when NaN loss is detected"""
    
    def __init__(self, patience=3):
        super().__init__()
        self.patience = patience
        self.nan_count = 0
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is None:  # NaN detected in training step
            self.nan_count += 1
            if self.nan_count >= self.patience:
                print(f"Stopping training due to {self.patience} consecutive NaN batches")
                trainer.should_stop = True
        else:
            self.nan_count = 0
            
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is None:  # NaN detected in validation step
            self.nan_count += 1
            if self.nan_count >= self.patience:
                print(f"Stopping training due to {self.patience} consecutive NaN batches")
                trainer.should_stop = True
        else:
            self.nan_count = 0

class SimpleProNetVAEModel(nn.Module):
    """
    Simplified ProteiNet + VAE model without ESM-Fold decoder
    """
    
    def __init__(self, 
                 pronet_hidden_dim=256,
                 pronet_num_layers=4,
                 vae_num_embeddings=512,
                 vae_embedding_dim=256,
                 vae_commitment_cost=0.25,
                 vae_decay=0.99):
        super().__init__()
        
        # ProteiNet encoder
        self.pronet_converter = ProNetConverter()
        self.pronet_encoder = nn.Sequential(
            nn.Linear(3, pronet_hidden_dim),  # xyz coordinates
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(pronet_hidden_dim, pronet_hidden_dim),
                nn.ReLU()
            ) for _ in range(pronet_num_layers - 1)],
            nn.Linear(pronet_hidden_dim, vae_embedding_dim)
        )
        
        # VAE discretization
        self.vae = VectorQuantizer2(
            n_e=vae_num_embeddings,
            e_dim=vae_embedding_dim,
            beta=vae_commitment_cost
        )
        
        # Simple decoder (just for testing)
        self.decoder = nn.Sequential(
            nn.Linear(vae_embedding_dim, pronet_hidden_dim),
            nn.ReLU(),
            nn.Linear(pronet_hidden_dim, pronet_hidden_dim),
            nn.ReLU(),
            nn.Linear(pronet_hidden_dim, 3)  # xyz coordinates
        )
    
    def forward(self, batch):
        """
        Forward pass through ProteiNet + VAE
        
        Args:
            batch: PyTorch Geometric batch with atom positions
            
        Returns:
            dict with reconstructed positions and VAE loss
        """
        # Use CA coordinates as input to the encoder
        if hasattr(batch, 'coords_ca'):
            atom_positions = batch.coords_ca  # [N, 3]
        else:
            raise ValueError("Input batch does not have 'coords_ca' attribute. Please check your dataset.")
        atom_positions = atom_positions.float()  # Ensure float dtype for AMP compatibility

        # Convert to dense batch: [batch_size, max_num_nodes, 3], mask: [batch_size, max_num_nodes]
        atom_positions_padded, mask = to_dense_batch(atom_positions, batch.batch)
        # Pass through encoder (process each protein in batch)
        encoded = self.pronet_encoder(atom_positions_padded)  # [batch_size, max_num_nodes, embedding_dim]

        # VAE discretization
        # Mask needs to be [batch_size, max_num_nodes, 1] to match encoded
        mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
        
        # Add gradient clipping to prevent explosion
        encoded = torch.clamp(encoded, min=-10, max=10)
        
        quantized, vae_loss, (perplexity, min_encodings, min_encoding_indices) = self.vae(encoded, mask_expanded)

        # Simple decoding
        reconstructed = self.decoder(quantized)  # [batch_size, max_num_nodes, 3]

        return {
            'reconstructed_positions': reconstructed,
            'vae_loss': vae_loss,
            'perplexity': perplexity,
            'encoded': encoded,
            'quantized': quantized,
            'mask': mask,
            'min_encoding_indices': min_encoding_indices
        }

class ProteinDataset(InMemoryDataset):
    def __init__(self, data_path: str, transform=None):
        super().__init__(None, transform)
        self.data_path = data_path
        self._data, self.slices = torch.load(data_path)
        self.num_proteins = len(self.slices['x']) - 1

    def len(self):
        return self.num_proteins

    def get(self, idx):
        start_idx = self.slices['x'][idx]
        end_idx = self.slices['x'][idx + 1]
        protein_data = Data()
        for key in self._data.keys():
            if isinstance(self._data[key], torch.Tensor):
                protein_data[key] = self._data[key][start_idx:end_idx]
            else:
                protein_data[key] = self._data[key]
        if not hasattr(protein_data, 'node_mask'):
            protein_data.node_mask = torch.ones(protein_data.num_nodes, dtype=torch.bool)
        return protein_data

class ProteinDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, test_path=None, batch_size=2, num_workers=2):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = ProteinDataset(self.train_path)
        self.val_dataset = ProteinDataset(self.val_path)
        self.test_dataset = ProteinDataset(self.test_path) if self.test_path else None

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return None

class SimpleProNetVAELightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for simplified ProteiNet + VAE training
    """
    
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters()
    
    def forward(self, batch):
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        outputs = self.model(batch)
        pred_positions = outputs['reconstructed_positions']  # [B, L, 3]
        # Densify target positions to match pred_positions
        target_positions, mask = to_dense_batch(batch.coords_ca.float(), batch.batch)
        
        # Check for NaN/Inf values and handle them
        if torch.isnan(pred_positions).any() or torch.isinf(pred_positions).any():
            print(f"Warning: NaN/Inf detected in predictions at batch {batch_idx}")
            return None
        
        # Compute loss only on valid (unpadded) positions
        reconstruction_loss = F.mse_loss(pred_positions[mask], target_positions[mask])
        vae_loss = outputs['vae_loss']
        
        # Check for NaN/Inf in losses
        if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
            print(f"Warning: NaN/Inf in reconstruction loss at batch {batch_idx}")
            return None
        
        if torch.isnan(vae_loss) or torch.isinf(vae_loss):
            print(f"Warning: NaN/Inf in VAE loss at batch {batch_idx}")
            return None
        
        # Scale losses to prevent explosion
        reconstruction_loss = torch.clamp(reconstruction_loss, max=1e6)
        vae_loss = torch.clamp(vae_loss, max=1e6)
        
        total_loss = reconstruction_loss + vae_loss
        
        # Final NaN check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN/Inf in total loss at batch {batch_idx}")
            return None
        
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_reconstruction_loss', reconstruction_loss)
        self.log('train_vae_loss', vae_loss)
        if outputs['perplexity'] is not None:
            self.log('train_perplexity', outputs['perplexity'])
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch)
        pred_positions = outputs['reconstructed_positions']  # [B, L, 3]
        target_positions, mask = to_dense_batch(batch.coords_ca.float(), batch.batch)
        
        # Check for NaN/Inf values and handle them
        if torch.isnan(pred_positions).any() or torch.isinf(pred_positions).any():
            print(f"Warning: NaN/Inf detected in validation predictions at batch {batch_idx}")
            return None
        
        reconstruction_loss = F.mse_loss(pred_positions[mask], target_positions[mask])
        vae_loss = outputs['vae_loss']
        
        # Check for NaN/Inf in losses
        if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
            print(f"Warning: NaN/Inf in validation reconstruction loss at batch {batch_idx}")
            return None
        
        if torch.isnan(vae_loss) or torch.isinf(vae_loss):
            print(f"Warning: NaN/Inf in validation VAE loss at batch {batch_idx}")
            return None
        
        # Scale losses to prevent explosion
        reconstruction_loss = torch.clamp(reconstruction_loss, max=1e6)
        vae_loss = torch.clamp(vae_loss, max=1e6)
        
        total_loss = reconstruction_loss + vae_loss
        
        # Final NaN check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN/Inf in validation total loss at batch {batch_idx}")
            return None
        
        # Safe RMSD calculation
        try:
            rmsd = torch.sqrt(torch.mean((pred_positions[mask] - target_positions[mask]) ** 2))
            if torch.isnan(rmsd) or torch.isinf(rmsd):
                rmsd = torch.tensor(0.0, device=total_loss.device)
        except:
            rmsd = torch.tensor(0.0, device=total_loss.device)
        
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_reconstruction_loss', reconstruction_loss)
        self.log('val_vae_loss', vae_loss)
        self.log('val_rmsd', rmsd)
        if outputs['perplexity'] is not None:
            self.log('val_perplexity', outputs['perplexity'])
        return total_loss
    
    def configure_optimizers(self):
        # Use AdamW with weight decay for better stability
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4,  # Add weight decay to prevent overfitting
            eps=1e-8  # Increase epsilon for numerical stability
        )
        
        # Use cosine annealing with warm restarts for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart interval each time
            eta_min=1e-6  # Minimum learning rate
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_model(config):
    """Create the simplified ProteiNet + VAE model"""
    model_config = config.get('model', {})
    
    # Extract VAE config
    vae_config = model_config.get('vae', {})
    
    model = SimpleProNetVAEModel(
        pronet_hidden_dim=model_config.get('pronet_hidden_dim', 256),
        pronet_num_layers=model_config.get('pronet_num_layers', 4),
        vae_num_embeddings=vae_config.get('num_embeddings', 512),
        vae_embedding_dim=vae_config.get('embedding_dim', 256),
        vae_commitment_cost=vae_config.get('commitment_cost', 0.25),
        vae_decay=vae_config.get('decay', 0.99)
    )
    return model

def create_datamodule(config):
    data_config = config.get('data', {})
    datamodule = ProteinDataModule(
        train_path=data_config.get('train_data_path'),
        val_path=data_config.get('val_data_path'),
        test_path=data_config.get('test_data_path'),
        batch_size=data_config.get('batch_size', 2),
        num_workers=data_config.get('num_workers', 2)
    )
    return datamodule

def create_trainer(config):
    """Create PyTorch Lightning trainer"""
    trainer_config = config.get('trainer', {})
    
    # Callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=trainer_config.get('checkpoint_dir', 'checkpoints'),
        filename='pronet_vae-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # NaN stopping callback
    nan_stopping = NaNStoppingCallback(patience=3)
    callbacks.append(nan_stopping)
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=trainer_config.get('log_dir', 'logs'),
        name='pronet_vae'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=trainer_config.get('max_epochs', 100),
        accelerator=trainer_config.get('accelerator', 'auto'),
        devices=trainer_config.get('devices', 'auto'),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=trainer_config.get('log_every_n_steps', 10),
        val_check_interval=trainer_config.get('val_check_interval', 1.0),
        gradient_clip_val=trainer_config.get('gradient_clip_val', 0.5),  # Reduce gradient clipping
        precision=trainer_config.get('precision', 32),  # Use FP32 for better stability
        deterministic=trainer_config.get('deterministic', False),
        accumulate_grad_batches=2  # Gradient accumulation for stability
    )
    
    return trainer

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ProteiNet + VAE model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Set random seeds
    pl.seed_everything(42)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create model
    model = create_model(config)
    
    # Create data module
    datamodule = create_datamodule(config)
    
    # Create lightning module
    lightning_module = SimpleProNetVAELightningModule(
        model=model,
        learning_rate=config.get('training', {}).get('learning_rate', 1e-4)
    )
    
    # Create trainer
    trainer = create_trainer(config)
    
    # Train
    trainer.fit(lightning_module, datamodule)
    
    # Test
    if datamodule.test_dataloader():
        trainer.test(lightning_module, datamodule)

if __name__ == '__main__':
    main() 