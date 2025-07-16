#!/usr/bin/env python3
"""
Training script for ProteiNet + VAE + ESM-Fold Structure Tokenization Model

This script trains the new structure tokenization model that uses:
1. ProteiNet for structure encoding (replaces GVP)
2. VAE (VectorQuantizer2) for discretization (replaces LFQ)
3. ESM-Fold for structure decoding (same as DPLM-2)

Usage:
    python train.py --config configs/pronet_vae.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.structok_pronet_vae import ProNetVAEModel
from byprot.datamodules.pdb_dataset.pdb_datamodule import PdbDataModule
from byprot.tasks.struct_tokenizer.structok import StrucTok


class ProNetVAETask(StrucTok):
    """
    Task wrapper for ProteiNet + VAE + ESM-Fold model
    """
    
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)
    
    def training_step(self, batch, batch_idx):
        """Training step with custom loss computation"""
        # Forward pass
        decoder_out, vae_loss, struct_tokens = self.model(batch)
        
        # Structure reconstruction loss
        structure_loss = self.compute_structure_loss(decoder_out, batch)
        
        # Total loss
        total_loss = structure_loss + vae_loss
        
        # Logging
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_structure_loss', structure_loss)
        self.log('train_vae_loss', vae_loss)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        # Forward pass
        decoder_out, vae_loss, struct_tokens = self.model(batch)
        
        # Structure reconstruction loss
        structure_loss = self.compute_structure_loss(decoder_out, batch)
        
        # Total loss
        total_loss = structure_loss + vae_loss
        
        # Compute additional metrics
        metrics = self.compute_metrics(decoder_out, batch)
        
        # Logging
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_structure_loss', structure_loss)
        self.log('val_vae_loss', vae_loss)
        
        for name, value in metrics.items():
            self.log(f'val_{name}', value)
        
        return total_loss
    
    def compute_structure_loss(self, decoder_out, batch):
        """Compute structure reconstruction loss"""
        # Extract predicted and target atom positions
        pred_positions = decoder_out["final_atom_positions"]
        target_positions = batch["all_atom_positions"]
        
        # Extract masks
        pred_mask = decoder_out["final_atom_mask"]
        target_mask = batch["all_atom_mask"]
        
        # Compute RMSD loss
        mask = pred_mask & target_mask
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_positions.device)
        
        # Compute squared distance
        squared_diff = (pred_positions - target_positions) ** 2
        squared_diff = squared_diff.sum(dim=-1)  # Sum over xyz dimensions
        
        # Apply mask and compute mean
        masked_diff = squared_diff * mask.float()
        loss = masked_diff.sum() / (mask.sum() + 1e-8)
        
        return loss
    
    def compute_metrics(self, decoder_out, batch):
        """Compute additional metrics"""
        metrics = {}
        
        # RMSD
        pred_positions = decoder_out["final_atom_positions"]
        target_positions = batch["all_atom_positions"]
        pred_mask = decoder_out["final_atom_mask"]
        target_mask = batch["all_atom_mask"]
        
        mask = pred_mask & target_mask
        if mask.sum() > 0:
            squared_diff = (pred_positions - target_positions) ** 2
            squared_diff = squared_diff.sum(dim=-1)
            masked_diff = squared_diff * mask.float()
            rmsd = torch.sqrt(masked_diff.sum() / (mask.sum() + 1e-8))
            metrics['rmsd'] = rmsd
        
        return metrics


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config):
    """Create the ProteiNet + VAE + ESM-Fold model"""
    model_config = config['model']
    model = ProNetVAEModel(**model_config)
    return model


def create_datamodule(config):
    """Create data module"""
    data_config = config['data']
    
    datamodule = PdbDataModule(
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        max_length=data_config['max_length'],
        train_data_path=data_config['train_data_path'],
        val_data_path=data_config['val_data_path'],
        test_data_path=data_config.get('test_data_path'),
    )
    
    return datamodule


def create_trainer(config):
    """Create PyTorch Lightning trainer"""
    trainer_config = config['trainer']
    
    # Callbacks
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/pronet_vae',
        filename='pronet_vae-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Logger
    logger = TensorBoardLogger('logs', name='pronet_vae')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=trainer_config['max_epochs'],
        accelerator=trainer_config['accelerator'],
        devices=trainer_config['devices'],
        precision=trainer_config['precision'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        gradient_clip_val=1.0,  # Gradient clipping for stability
    )
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description='Train ProteiNet + VAE + ESM-Fold model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Create model
    model = create_model(config)
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create task
    task = ProNetVAETask(
        model=model,
        optimizer_config=config['trainer']['optimizer'],
        scheduler_config=config['trainer']['scheduler'],
    )
    
    # Create data module
    datamodule = create_datamodule(config)
    
    # Create trainer
    trainer = create_trainer(config)
    
    # Train
    if args.resume:
        trainer.fit(task, datamodule, ckpt_path=args.resume)
    else:
        trainer.fit(task, datamodule)
    
    # Test
    trainer.test(task, datamodule)
    
    print("Training completed!")


if __name__ == '__main__':
    main() 