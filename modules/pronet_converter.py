# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import torch
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional


class ProNetConverter:
    """
    Convert protein structure data from DPLM atom37 format to ProteiNet graph format
    """
    
    def __init__(self):
        # Atom type mapping for atom37 format
        self.atom37_atom_types = [
            'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
            'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
            'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
            'CZ3', 'NZ', 'OXT'
        ]
        
        # Amino acid type mapping
        self.aa_types = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F',
            'P', 'S', 'T', 'W', 'Y', 'V', 'X', 'U', 'O', 'B', 'Z'
        ]
    
    def atom37_to_pronet_graph(
        self,
        atom_positions: torch.Tensor,  # [batch, seq_len, 37, 3]
        atom_mask: torch.Tensor,       # [batch, seq_len, 37]
        aatype: torch.Tensor,          # [batch, seq_len]
        mask: torch.Tensor,            # [batch, seq_len]
    ) -> List[Data]:
        """
        Convert atom37 format to list of ProteiNet graph Data objects
        
        Args:
            atom_positions: Atom coordinates in atom37 format
            atom_mask: Atom existence mask
            aatype: Amino acid type indices
            mask: Residue mask
            
        Returns:
            List of PyTorch Geometric Data objects for ProteiNet
        """
        batch_size = atom_positions.shape[0]
        graphs = []
        
        for b in range(batch_size):
            # Extract valid residues for this batch item
            valid_mask = mask[b].bool()
            if not valid_mask.any():
                continue
                
            # Get valid positions and types
            valid_positions = atom_positions[b][valid_mask]  # [valid_res, 37, 3]
            valid_mask_atoms = atom_mask[b][valid_mask]      # [valid_res, 37]
            valid_aatype = aatype[b][valid_mask]             # [valid_res]
            
            # Extract CA coordinates (backbone)
            ca_positions = valid_positions[:, 1]  # CA is at index 1
            
            # Create node features (amino acid type embeddings)
            node_features = self._create_node_features(valid_aatype)
            
            # Create edge features
            edge_index, edge_features = self._create_edge_features(ca_positions)
            
            # Create graph data
            graph_data = Data(
                x=node_features,           # Node features (amino acid types)
                coords_ca=ca_positions,    # CA coordinates
                edge_index=edge_index,     # Edge connectivity
                edge_attr=edge_features,   # Edge features
                batch=torch.zeros(len(valid_aatype), dtype=torch.long),  # Single graph
                sequence=self._aatype_to_sequence(valid_aatype),
                protein_id=f"batch_{b}",
            )
            
            graphs.append(graph_data)
        
        return graphs
    
    def _create_node_features(self, aatype: torch.Tensor) -> torch.Tensor:
        """
        Create node features from amino acid types
        
        Args:
            aatype: Amino acid type indices [seq_len]
            
        Returns:
            Node features [seq_len, num_aa_types]
        """
        # One-hot encode amino acid types
        num_aa_types = len(self.aa_types)
        node_features = torch.zeros(len(aatype), num_aa_types, dtype=torch.float)
        node_features.scatter_(1, aatype.unsqueeze(1), 1.0)
        return node_features
    
    def _create_edge_features(self, ca_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create edge features from CA positions
        
        Args:
            ca_positions: CA coordinates [seq_len, 3]
            
        Returns:
            edge_index: Edge connectivity [2, num_edges]
            edge_features: Edge features [num_edges, feature_dim]
        """
        seq_len = len(ca_positions)
        
        # Create all pairwise edges within cutoff distance
        cutoff = 10.0  # ProteiNet default cutoff
        
        # Calculate pairwise distances
        dist_matrix = torch.cdist(ca_positions, ca_positions)
        
        # Find edges within cutoff
        edge_mask = (dist_matrix < cutoff) & (dist_matrix > 0)  # Exclude self-loops
        
        # Get edge indices
        edge_indices = torch.nonzero(edge_mask, as_tuple=False).t()
        
        if edge_indices.numel() == 0:
            # No edges found, create empty tensors
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_features = torch.zeros(0, 6)  # Basic distance features
        else:
            # Calculate edge features (distance, direction)
            src_pos = ca_positions[edge_indices[0]]
            dst_pos = ca_positions[edge_indices[1]]
            
            # Distance
            distances = torch.norm(dst_pos - src_pos, dim=1, keepdim=True)
            
            # Normalized direction vector
            directions = (dst_pos - src_pos) / (distances + 1e-8)
            
            # Combine features
            edge_features = torch.cat([distances, directions], dim=1)
        
        return edge_indices, edge_features
    
    def _aatype_to_sequence(self, aatype: torch.Tensor) -> str:
        """
        Convert amino acid type indices to sequence string
        
        Args:
            aatype: Amino acid type indices
            
        Returns:
            Amino acid sequence string
        """
        sequence = ""
        for aa_idx in aatype:
            if aa_idx < len(self.aa_types):
                sequence += self.aa_types[aa_idx]
            else:
                sequence += 'X'  # Unknown amino acid
        return sequence
    
    def batch_graphs(self, graphs: List[Data]) -> Data:
        """
        Batch multiple graphs into a single Data object
        
        Args:
            graphs: List of individual graph Data objects
            
        Returns:
            Batched graph Data object
        """
        if not graphs:
            raise ValueError("Cannot batch empty list of graphs")
        
        return Batch.from_data_list(graphs)
    
    def pronet_output_to_embeddings(
        self,
        pronet_output: torch.Tensor,  # [total_nodes, hidden_dim]
        batch: torch.Tensor,          # [total_nodes] - batch assignment
        original_mask: torch.Tensor,  # [batch_size, seq_len] - original mask
    ) -> torch.Tensor:
        """
        Convert ProteiNet output back to batch format
        
        Args:
            pronet_output: ProteiNet model output
            batch: Batch assignment for each node
            original_mask: Original residue mask
            
        Returns:
            Embeddings in batch format [batch_size, max_seq_len, hidden_dim]
        """
        batch_size, max_seq_len = original_mask.shape
        hidden_dim = pronet_output.shape[-1]
        
        # Initialize output tensor
        embeddings = torch.zeros(batch_size, max_seq_len, hidden_dim, 
                               device=pronet_output.device, dtype=pronet_output.dtype)
        
        # Fill embeddings for each batch
        for b in range(batch_size):
            # Get nodes belonging to this batch
            batch_mask = (batch == b)
            if batch_mask.any():
                # Get embeddings for this batch
                batch_embeddings = pronet_output[batch_mask]
                
                # Count how many residues we have for this batch
                num_residues = batch_embeddings.shape[0]
                
                # Fill the embeddings (only for valid residues)
                embeddings[b, :num_residues] = batch_embeddings
        
        return embeddings


def create_pronet_input_from_batch(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Convenience function to create ProteiNet input from DPLM batch
    
    Args:
        batch: DPLM batch dictionary
        
    Returns:
        ProteiNet input tensor
    """
    converter = ProNetConverter()
    
    # Convert to ProteiNet graphs
    graphs = converter.atom37_to_pronet_graph(
        atom_positions=batch["all_atom_positions"],
        atom_mask=batch["all_atom_mask"],
        aatype=batch["aatype"],
        mask=batch["res_mask"],
    )
    
    # Batch graphs
    if graphs:
        batched_graph = converter.batch_graphs(graphs)
        return batched_graph
    else:
        # Return empty tensor if no valid graphs
        return torch.zeros(0, 128)  # Placeholder 