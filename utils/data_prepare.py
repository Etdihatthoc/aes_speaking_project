# -*- coding: utf-8 -*-
import numpy as np
import torch
from utils.utils import get_logger, normalize_scores

logger = get_logger("Data Preparation")


def create_pairwise_data(data, config):
    """
    Create pairwise training data following NPCR approach
    
    Strategy:
    1. For adjacent essays, create pairs
    2. Include accumulation pairs (if we have (e_i, e_j) and (e_j, e_k), add (e_i, e_k))
    """
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    scores = data['scores']
    
    pairs = []
    relative_scores = []
    
    n_samples = len(input_ids)
    
    # Create adjacent pairs with different scores
    for i in range(n_samples - 3):
        # Pairs with immediate neighbors
        if scores[i] != scores[i + 1]:
            pairs.append((i, i + 1))
            relative_scores.append(scores[i] - scores[i + 1])
        
        # Pairs with second neighbors
        if scores[i] != scores[i + 2]:
            pairs.append((i, i + 2))
            relative_scores.append(scores[i] - scores[i + 2])
        
        # Pairs with third neighbors
        if scores[i] != scores[i + 3]:
            pairs.append((i, i + 3))
            relative_scores.append(scores[i] - scores[i + 3])
    
    # Convert to tensors
    if len(pairs) > 0:
        pairs_array = np.array(pairs)
        
        # Get pairwise data
        input_ids1 = input_ids[pairs_array[:, 0]]
        attention_mask1 = attention_mask[pairs_array[:, 0]]
        input_ids2 = input_ids[pairs_array[:, 1]]
        attention_mask2 = attention_mask[pairs_array[:, 1]]
        
        # Normalize relative scores to [0, 1]
        # Since relative scores can be negative, we need to handle this
        relative_scores = torch.FloatTensor(relative_scores)
        # Map to [0, 1] using sigmoid-like transformation
        relative_scores = (relative_scores + 1) / 2  # Maps [-1, 1] to [0, 1]
        
        pairwise_data = {
            'input_ids1': input_ids1,
            'attention_mask1': attention_mask1,
            'input_ids2': input_ids2,
            'attention_mask2': attention_mask2,
            'relative_scores': relative_scores
        }
        
        logger.info(f"Created {len(pairs)} training pairs")
        
        return pairwise_data
    else:
        logger.warning("No valid pairs created!")
        return None


def prepare_reference_samples(train_data, config):
    """
    Prepare reference samples for multi-sample voting during inference
    
    Strategy: Select diverse samples from training data
    """
    example_size = config['training']['example_size']
    
    # Get training data
    input_ids = train_data['input_ids']
    attention_mask = train_data['attention_mask']
    scores = train_data['scores']
    
    n_samples = len(input_ids)
    
    # Strategy 1: Select evenly distributed samples across score range
    sorted_indices = torch.argsort(scores)
    
    # Select indices evenly distributed
    step = max(1, n_samples // example_size)
    selected_indices = sorted_indices[::step][:example_size]
    
    # If we don't have enough samples, randomly select more
    if len(selected_indices) < example_size:
        remaining = example_size - len(selected_indices)
        remaining_indices = torch.randperm(n_samples)[:remaining]
        selected_indices = torch.cat([selected_indices, remaining_indices])
    
    reference_data = {
        'input_ids': input_ids[selected_indices],
        'attention_mask': attention_mask[selected_indices],
        'scores': scores[selected_indices]
    }
    
    logger.info(f"Selected {len(selected_indices)} reference samples")
    
    return reference_data


class PairwiseDataset(torch.utils.data.Dataset):
    """Dataset for pairwise training"""
    
    def __init__(self, pairwise_data):
        self.input_ids1 = pairwise_data['input_ids1']
        self.attention_mask1 = pairwise_data['attention_mask1']
        self.input_ids2 = pairwise_data['input_ids2']
        self.attention_mask2 = pairwise_data['attention_mask2']
        self.relative_scores = pairwise_data['relative_scores']
    
    def __len__(self):
        return len(self.relative_scores)
    
    def __getitem__(self, idx):
        return (
            self.input_ids1[idx],
            self.attention_mask1[idx],
            self.input_ids2[idx],
            self.attention_mask2[idx],
            self.relative_scores[idx]
        )