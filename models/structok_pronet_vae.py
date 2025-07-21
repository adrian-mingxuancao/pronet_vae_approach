# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import torch
from torch import nn
import torch.nn.functional as F

# Add pronet_prepoc to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../pronet_prepoc'))
from pronet.pronet import ProNet

from byprot.datamodules.pdb_dataset import protein
from byprot.datamodules.pdb_dataset.pdb_datamodule import (
    PdbDataset,
    aatype_to_seq,
    collate_fn,
    seq_to_aatype,
    struct_ids_to_seq,
    struct_seq_to_ids,
)
from byprot.models import register_model

from modules.ema import LitEma
from modules.vqvae import VectorQuantizer2
from modules.folding_utils.decoder import ESMFoldStructureDecoder as Decoder
from modules.pronet_converter import ProNetConverter, create_pronet_input_from_batch
from modules.nn import TransformerEncoder


def exists(o):
    return o is not None


@register_model("structok_pronet_vae")
class ProNetVAEModel(nn.Module):
    """
    Structure Tokenization using ProteiNet + VAE + ESM-Fold
    
    Pipeline:
    1. ProteiNet: Structure -> Continuous embeddings (replaces GVP)
    2. VAE: Continuous embeddings -> Discrete tokens (replaces LFQ)
    3. ESM-Fold: Discrete tokens -> Structure (same as DPLM-2)
    
    This follows the DPLM-2 architecture but replaces:
    - GVP encoder with ProteiNet encoder
    - LFQ quantizer with VAE quantizer
    - Keeps ESM-Fold decoder (same as DPLM-2)
    """
    
    def __init__(
        self,
        encoder_config,
        decoder_config,
        codebook_config,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        batch_resize_range=None,
        scheduler_config=None,
        lr_g_factor=1.0,
        remap=None,
        sane_index_shape=True,
        use_ema=False,
    ):
        super().__init__()
        self.codebook_embed_dim = codebook_config.get('embed_dim', 128)
        self.num_codebook = codebook_config.get('num_codes', 512)
        self.image_key = image_key
        
        # 1. ProteiNet Encoder (replaces GVP)
        self.encoder = ProNet(
            level=encoder_config.get('level', 'aminoacid'),
            num_blocks=encoder_config.get('num_blocks', 4),
            hidden_channels=encoder_config.get('hidden_channels', 128),
            out_channels=encoder_config.get('out_channels', 1),
            mid_emb=encoder_config.get('mid_emb', 64),
            num_radial=encoder_config.get('num_radial', 6),
            num_spherical=encoder_config.get('num_spherical', 2),
            cutoff=encoder_config.get('cutoff', 10.0),
            max_num_neighbors=encoder_config.get('max_num_neighbors', 32),
            int_emb_layers=encoder_config.get('int_emb_layers', 3),
            out_layers=encoder_config.get('out_layers', 2),
            num_pos_emb=encoder_config.get('num_pos_emb', 16),
            dropout=encoder_config.get('dropout', 0),
            data_augment_eachlayer=encoder_config.get('data_augment_eachlayer', False),
            final_pred=encoder_config.get('final_pred', False),
            out_hidden_channels=encoder_config.get('out_hidden_channels', 2048),
            pool=encoder_config.get('pool', False),
        )
        
        # 2. VAE Quantizer (replaces LFQ)
        self.quantize = VectorQuantizer2(
            n_e=self.num_codebook,
            e_dim=self.codebook_embed_dim,
            beta=codebook_config.get('beta', 0.25),
            remap=remap,
            unknown_index="random",
            sane_index_shape=sane_index_shape,
            legacy=False,
        )
        
        # 3. ESM-Fold Decoder (same as DPLM-2)
        self.decoder = Decoder(**decoder_config)
        
        # Pre-quantization projection
        encoder_output_dim = encoder_config.get('hidden_channels', 128)
        self.pre_quant = nn.Sequential(
            nn.LayerNorm(encoder_output_dim),
            nn.Linear(encoder_output_dim, self.codebook_embed_dim),
            nn.ReLU(),
            nn.Linear(self.codebook_embed_dim, self.codebook_embed_dim),
        )
        
        # Post-quantization projection (same as DPLM-2)
        self.post_quant = nn.ModuleDict({
            "mlp": nn.Sequential(
                nn.LayerNorm(self.codebook_embed_dim),
                nn.Linear(self.codebook_embed_dim, self.decoder.input_dim),
                nn.ReLU(),
                nn.Linear(self.decoder.input_dim, self.decoder.input_dim),
            ),
            "transformer": TransformerEncoder(
                self.decoder.input_dim, 8, 4
            ),
        })
        
        if codebook_config.get("freeze"):
            self.quantize.requires_grad_(False)
            self.pre_quant.requires_grad_(False)
            
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(
                f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}."
            )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

        self.struct_seq_to_ids = struct_seq_to_ids
        self.struct_ids_to_seq = struct_ids_to_seq
        self.aatype_to_seq = aatype_to_seq
        self.seq_to_aatype = seq_to_aatype

        self.process_chain = PdbDataset.process_chain
        
        # Initialize converter
        self.converter = ProNetConverter()

    def forward(self, batch, return_pred_indices=True, decoder_kwargs={}):
        pre_quant, encoder_feats = self.encode(
            atom_positions=batch["all_atom_positions"],
            mask=batch["res_mask"],
            seq_length=batch["seq_length"],
        )

        quant, loss, (_, _, struct_tokens) = self.quantize(
            pre_quant, mask=batch["res_mask"].bool()
        )

        struct_feat = quant

        decoder_out = self.decode(
            quant=struct_feat,
            aatype=batch["aatype"],
            mask=batch["res_mask"],
            decoder_kwargs=decoder_kwargs,
        )
        
        if return_pred_indices:
            return decoder_out, loss, struct_tokens
        else:
            return decoder_out, loss

    def encode(self, atom_positions, mask, seq_length=None, gvp_feat=None):
        """
        Encode protein structure using ProteiNet (replaces GVP)
        """
        batch_size, seq_len, _, _ = atom_positions.shape
        
        # Create batch dictionary for converter
        batch_dict = {
            "all_atom_positions": atom_positions,
            "all_atom_mask": torch.ones_like(atom_positions[..., 0]),  # Assume all atoms exist
            "aatype": torch.zeros(batch_size, seq_len, dtype=torch.long),  # Placeholder aatype
            "res_mask": mask,
        }
        
        # Convert to ProteiNet graphs
        graphs = self.converter.atom37_to_pronet_graph(
            atom_positions=atom_positions,
            atom_mask=batch_dict["all_atom_mask"],
            aatype=batch_dict["aatype"],
            mask=mask,
        )
        
        if not graphs:
            # No valid graphs, return zeros
            encoder_output_dim = self.encoder.hidden_channels
            dummy_output = torch.zeros(batch_size, seq_len, encoder_output_dim, 
                                     device=atom_positions.device)
            pre_quant = self.pre_quant(dummy_output)
            return pre_quant, dummy_output
        
        # Batch graphs for ProteiNet
        batched_graph = self.converter.batch_graphs(graphs)
        
        # Encode with ProteiNet
        encoder_feats = self.encoder(batched_graph)
        
        # Convert back to batch format
        encoder_feats = self.converter.pronet_output_to_embeddings(
            pronet_output=encoder_feats,
            batch=batched_graph.batch,
            original_mask=mask,
        )
        
        # Project to quantization space
        pre_quant = self.pre_quant(encoder_feats)
        
        # Mask out missing positions
        pre_quant = pre_quant * mask[..., None]
        
        return pre_quant, encoder_feats

    def decode(self, quant, aatype, mask, decoder_kwargs={}):
        """
        Decode using ESM-Fold (same as DPLM-2)
        """
        def _post_quant(x, mask):
            x = self.post_quant["mlp"](x)
            x = self.post_quant["transformer"](x, padding_mask=1 - mask)["out"]
            return x

        quant = _post_quant(quant, mask)
        
        decoder_out = self.decoder(
            emb_s=quant,
            emb_z=None,
            mask=mask,
            aa=aatype,
            esmaa=aatype,
            **decoder_kwargs,
        )
        return decoder_out

    def quantize_and_decode(
        self, pre_quant, mask=None, aatype=None, decoder_kwargs={}
    ):
        if not exists(mask):
            mask = torch.ones(
                *pre_quant.shape[:2],
                dtype=torch.float32,
                device=pre_quant.device,
            )
        aatype = torch.zeros_like(mask, dtype=torch.int64)

        quant, loss, (_, _, struct_tokens) = self.quantize(
            pre_quant, mask=mask.bool()
        )
        decoder_out = self.decode(quant, aatype, mask, decoder_kwargs)
        return decoder_out, struct_tokens

    def get_decoder_features(self, struct_tokens, res_mask, unk_mask):
        # use 0 as unk/mask id
        struct_tokens = struct_tokens.masked_fill(unk_mask, 0)
        quant = self.quantize.get_codebook_entry(struct_tokens)
        res_mask = res_mask.float()
        quant = self._post_quant(quant, res_mask)

        _aatypes = torch.zeros_like(struct_tokens, dtype=torch.int64)
        decoder_out = self.decoder(
            emb_s=quant,
            emb_z=None,
            mask=res_mask,
            aa=_aatypes,
            esmaa=_aatypes,
            return_features_only=True,
        )
        single_feats, pair_feats = decoder_out["s_s"], decoder_out["s_z"]
        return single_feats, pair_feats

    def tokenize(self, atom_positions, res_mask, seq_length=None):
        pre_quant, _ = self.encode(
            atom_positions=atom_positions,
            mask=res_mask,
            seq_length=seq_length,
        )
        quant, loss, (_, _, struct_tokens) = self.quantize(
            pre_quant, mask=res_mask.bool()
        )
        return struct_tokens

    def detokenize(self, struct_tokens, res_mask=None, **kwargs):
        if struct_tokens.ndim == 2:
            quant = self.quantize.get_codebook_entry(struct_tokens)
        elif struct_tokens.ndim == 3:
            quant = struct_tokens
        else:
            raise ValueError(
                f"Invalid struct_tokens shape: {struct_tokens.shape}"
            )

        device = struct_tokens.device

        if not exists(res_mask):
            res_mask = torch.ones(
                struct_tokens.shape[:2], dtype=torch.float32, device=device
            )
        _aatypes = torch.zeros(
            struct_tokens.shape[:2], dtype=torch.int64, device=device
        )

        decoder_out = self.decode(
            quant=quant, aatype=_aatypes, mask=res_mask, decoder_kwargs=kwargs
        )
        decoder_out = dict(
            atom37_positions=decoder_out["final_atom_positions"],
            atom37_mask=decoder_out["final_atom_mask"],
        )
        return decoder_out

    def string_to_tensor(self, aatype_str, struct_token_str):
        aatype = torch.tensor([self.aatype_to_seq[aa] for aa in aatype_str])
        struct_tokens = torch.tensor([int(t) for t in struct_token_str.split()])
        return aatype, struct_tokens

    def init_data(self, raw_batch):
        return self.process_chain(raw_batch)

    def output_to_pdb(self, decoder_out, output_dir, is_trajectory=False):
        return protein.output_to_pdb(decoder_out, output_dir, is_trajectory)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}") 