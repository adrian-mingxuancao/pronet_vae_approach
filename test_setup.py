#!/usr/bin/env python3
"""
Test script to verify the ProteiNet + VAE + ESM-Fold setup

This script tests that all components are properly imported and can be instantiated.
"""

import os
import sys
import torch
import yaml

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from models.structok_pronet_vae import ProNetVAEModel
        print("✓ ProNetVAEModel imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ProNetVAEModel: {e}")
        return False
    
    try:
        from modules.pronet_converter import ProNetConverter
        print("✓ ProNetConverter imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ProNetConverter: {e}")
        return False
    
    try:
        from modules.vqvae import VectorQuantizer2
        print("✓ VectorQuantizer2 imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import VectorQuantizer2: {e}")
        return False
    
    return True

def test_config_loading():
    """Test that configuration can be loaded"""
    print("\nTesting configuration loading...")
    
    try:
        with open('configs/pronet_vae.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False

def test_model_creation():
    """Test that the model can be created"""
    print("\nTesting model creation...")
    
    try:
        # Load config
        with open('configs/pronet_vae.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        model = ProNetVAEModel(**config['model'])
        print(f"✓ Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        return True
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False

def test_sample_forward_pass():
    """Test a sample forward pass"""
    print("\nTesting sample forward pass...")
    
    try:
        # Load config
        with open('configs/pronet_vae.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        model = ProNetVAEModel(**config['model'])
        model.eval()
        
        # Create sample batch
        batch_size, seq_len = 2, 50
        batch = {
            "all_atom_positions": torch.randn(batch_size, seq_len, 37, 3),
            "all_atom_mask": torch.ones(batch_size, seq_len, 37),
            "aatype": torch.randint(0, 20, (batch_size, seq_len)),
            "res_mask": torch.ones(batch_size, seq_len),
            "seq_length": torch.full((batch_size,), seq_len),
        }
        
        # Forward pass
        with torch.no_grad():
            decoder_out, vae_loss, struct_tokens = model(batch)
        
        print(f"✓ Forward pass successful")
        print(f"  - Decoder output keys: {list(decoder_out.keys())}")
        print(f"  - VAE loss: {vae_loss.item():.4f}")
        print(f"  - Structure tokens shape: {struct_tokens.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Failed forward pass: {e}")
        return False

def main():
    """Run all tests"""
    print("=== ProteiNet + VAE + ESM-Fold Setup Test ===\n")
    
    tests = [
        test_imports,
        test_config_loading,
        test_model_creation,
        test_sample_forward_pass,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! The setup is working correctly.")
        print("\nYou can now:")
        print("1. Run the demo: python scripts/demo.py --config configs/pronet_vae.yaml")
        print("2. Start training: python scripts/train.py --config configs/pronet_vae.yaml")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nCommon issues:")
        print("1. Missing dependencies - run: pip install -r requirements.txt")
        print("2. Incorrect paths - make sure you're in the pronet_vae_approach directory")
        print("3. Missing ProteiNet - check the pronet_prepoc folder")

if __name__ == '__main__':
    main() 