# ProteiNet + VAE + ESM-Fold Structure Tokenization

A new approach to protein structure tokenization that improves upon DPLM-2's current pipeline.

## What This Does

**Current DPLM-2**: `Protein Structure → GVP Encoder → LFQ Quantizer → ESM-Fold Decoder`

**New Approach**: `Protein Structure → ProteiNet Encoder → VAE Quantizer → ESM-Fold Decoder`

## Key Improvements

1. **ProteiNet Encoder** (replaces GVP)
   - Better 3D geometric modeling using complete graph networks
   - Captures protein interactions more effectively

2. **VAE Quantizer** (replaces LFQ) 
   - More stable training dynamics
   - Better codebook utilization

3. **ESM-Fold Decoder** (same as DPLM-2)
   - Keeps proven high-quality structure prediction
   - Uses IPA (Invariant Point Attention) internally to convert embeddings to 3D coordinates

## Quick Start

### Installation
```bash
cd dplm/pronet_vae_approach
pip install -r requirements.txt
```

### Training
```bash
python scripts/train.py --config configs/pronet_vae.yaml
```

### Demo
```bash
python scripts/demo.py --config configs/pronet_vae.yaml --device cuda
```

## What the Training Script Does

The `train.py` script trains the **entire encode-decode pipeline**:

1. **ProteiNet** encodes protein structures into continuous representations
2. **VAE** discretizes these into tokens 
3. **ESM-Fold** decodes tokens back to protein structures:
   - Converts discrete tokens to continuous embeddings
   - Processes through FoldingTrunk (transformer blocks)
   - Uses Structure Module with IPA to generate 3D coordinates

It's training the complete tokenization system, not just a single component.

## Project Structure
```
pronet_vae_approach/
├── models/structok_pronet_vae.py    # Main model
├── modules/pronet_converter.py      # Data conversion
├── configs/pronet_vae.yaml          # Configuration
├── scripts/
│   ├── train.py                     # Training script
│   └── demo.py                      # Demo script
└── requirements.txt                 # Dependencies
```

## Expected Benefits

- **Better Structure Quality**: More accurate geometric modeling from ProteiNet's 3D graph networks
- **Stable Training**: More predictable convergence with VAE vs LFQ
- **Richer Features**: Better protein interaction modeling
- **Proven Decoding**: ESM-Fold's IPA-based structure module ensures high-quality 3D reconstruction

## Configuration

Key parameters in `configs/pronet_vae.yaml`:
- `model.pronet.*`: ProteiNet encoder settings
- `model.vae.*`: VAE quantizer settings  
- `model.decoder.*`: ESM-Fold decoder settings
- `data.*`: Dataset and training settings

## Troubleshooting

**Memory Issues**: Reduce batch size in config
**Training Instability**: Lower learning rate, check data quality
**Poor Results**: Verify data preprocessing and atom coordinates 