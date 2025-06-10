# -*- coding: utf-8 -*-
import numpy as np
import torch
from utils.utils import get_logger, normalize_scores

logger = get_logger("Data Preparation")


def create_pairwise_data(data, config):
    """
    Create pairwise training data following NPCR approach with robust error handling
    
    Strategy:
    1. For adjacent essays, create pairs
    2. Include accumulation pairs (if we have (e_i, e_j) and (e_j, e_k), add (e_i, e_k))
    """
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    scores = data['scores']
    
    # Validate input data
    if len(input_ids) == 0 or len(scores) == 0:
        logger.error("Empty input data for pairwise creation")
        return None
    
    # Check for NaN or invalid scores
    if torch.any(torch.isnan(scores)) or torch.any(torch.isinf(scores)):
        logger.warning("NaN/Inf scores found, cleaning before pairwise creation...")
        scores = torch.clamp(scores, 0.0, 1.0)
        scores[torch.isnan(scores)] = 0.5
        scores[torch.isinf(scores)] = 0.5
    
    pairs = []
    relative_scores = []
    
    n_samples = len(input_ids)
    logger.info(f"Creating pairwise data from {n_samples} samples")
    
    # Create adjacent pairs with different scores
    pairs_created = 0
    for i in range(n_samples - 3):
        try:
            # Get scores for comparison
            score_i = scores[i].item()
            
            # Pairs with immediate neighbors
            if i + 1 < n_samples:
                score_j = scores[i + 1].item()
                if abs(score_i - score_j) > 1e-6:  # Only create pairs with meaningful differences
                    pairs.append((i, i + 1))
                    relative_score = score_i - score_j
                    # Ensure relative score is finite
                    if np.isnan(relative_score) or np.isinf(relative_score):
                        relative_score = 0.0
                    relative_scores.append(relative_score)
                    pairs_created += 1
            
            # Pairs with second neighbors
            if i + 2 < n_samples:
                score_k = scores[i + 2].item()
                if abs(score_i - score_k) > 1e-6:
                    pairs.append((i, i + 2))
                    relative_score = score_i - score_k
                    if np.isnan(relative_score) or np.isinf(relative_score):
                        relative_score = 0.0
                    relative_scores.append(relative_score)
                    pairs_created += 1
            
            # Pairs with third neighbors
            if i + 3 < n_samples:
                score_l = scores[i + 3].item()
                if abs(score_i - score_l) > 1e-6:
                    pairs.append((i, i + 3))
                    relative_score = score_i - score_l
                    if np.isnan(relative_score) or np.isinf(relative_score):
                        relative_score = 0.0
                    relative_scores.append(relative_score)
                    pairs_created += 1
                    
        except Exception as e:
            logger.warning(f"Error creating pairs for index {i}: {e}")
            continue
    
    # Convert to tensors if we have pairs
    if len(pairs) > 0 and len(relative_scores) > 0:
        try:
            pairs_array = np.array(pairs)
            
            # Validate pairs indices
            max_idx = max(pairs_array.flatten())
            if max_idx >= n_samples:
                logger.error(f"Invalid pair index {max_idx} >= {n_samples}")
                return None
            
            # Get pairwise data
            input_ids1 = input_ids[pairs_array[:, 0]]
            attention_mask1 = attention_mask[pairs_array[:, 0]]
            input_ids2 = input_ids[pairs_array[:, 1]]
            attention_mask2 = attention_mask[pairs_array[:, 1]]
            
            # Process relative scores with better normalization
            relative_scores_array = np.array(relative_scores)
            
            # Check for invalid relative scores
            if np.any(np.isnan(relative_scores_array)) or np.any(np.isinf(relative_scores_array)):
                logger.warning("Invalid relative scores found, cleaning...")
                valid_mask = ~(np.isnan(relative_scores_array) | np.isinf(relative_scores_array))
                if np.any(valid_mask):
                    mean_rel_score = np.mean(relative_scores_array[valid_mask])
                    relative_scores_array[~valid_mask] = mean_rel_score
                else:
                    relative_scores_array = np.zeros_like(relative_scores_array)
            
            # Improved normalization for relative scores
            # Since scores are already in [0,1], relative scores are in [-1,1]
            # Map to [0,1] using (x + 1) / 2
            relative_scores_normalized = (relative_scores_array + 1.0) / 2.0
            
            # Ensure normalized scores are in [0,1]
            relative_scores_normalized = np.clip(relative_scores_normalized, 0.0, 1.0)
            
            # Final validation
            if np.any(np.isnan(relative_scores_normalized)) or np.any(np.isinf(relative_scores_normalized)):
                logger.error("Still have invalid relative scores after normalization!")
                relative_scores_normalized = np.clip(relative_scores_normalized, 0.0, 1.0)
                relative_scores_normalized[np.isnan(relative_scores_normalized)] = 0.5
                relative_scores_normalized[np.isinf(relative_scores_normalized)] = 0.5
            
            relative_scores_tensor = torch.FloatTensor(relative_scores_normalized)
            
            pairwise_data = {
                'input_ids1': input_ids1,
                'attention_mask1': attention_mask1,
                'input_ids2': input_ids2,
                'attention_mask2': attention_mask2,
                'relative_scores': relative_scores_tensor
            }
            
            logger.info(f"Created {len(pairs)} training pairs successfully")
            logger.info(f"Relative score range: {relative_scores_tensor.min().item():.4f} - {relative_scores_tensor.max().item():.4f}")
            
            return pairwise_data
            
        except Exception as e:
            logger.error(f"Error processing pairwise data: {e}")
            return None
    else:
        logger.warning("No valid pairs created!")
        return None


def prepare_reference_samples(train_data, config):
    """
    Prepare reference samples for multi-sample voting during inference with robust handling
    
    Strategy: Select diverse samples from training data
    """
    try:
        example_size = config['training']['example_size']
        
        # Get training data
        input_ids = train_data['input_ids']
        attention_mask = train_data['attention_mask']
        scores = train_data['scores']
        
        n_samples = len(input_ids)
        
        if n_samples == 0:
            logger.error("No training data available for reference samples")
            return None
        
        # Validate scores
        if torch.any(torch.isnan(scores)) or torch.any(torch.isinf(scores)):
            logger.warning("Invalid scores in reference data, cleaning...")
            scores = torch.clamp(scores, 0.0, 1.0)
            scores[torch.isnan(scores)] = 0.5
            scores[torch.isinf(scores)] = 0.5
        
        # Strategy: Select evenly distributed samples across score range
        sorted_indices = torch.argsort(scores)
        
        # Ensure we don't exceed available samples
        actual_example_size = min(example_size, n_samples)
        
        # Select indices evenly distributed
        if actual_example_size >= n_samples:
            # Use all samples if we don't have enough
            selected_indices = sorted_indices
        else:
            step = max(1, n_samples // actual_example_size)
            selected_indices = sorted_indices[::step][:actual_example_size]
        
        # If we still don't have enough samples, pad with random selection
        if len(selected_indices) < actual_example_size and len(selected_indices) < n_samples:
            remaining_needed = min(actual_example_size - len(selected_indices), n_samples - len(selected_indices))
            if remaining_needed > 0:
                all_indices = torch.arange(n_samples)
                available_indices = torch.tensor([i for i in all_indices if i not in selected_indices])
                if len(available_indices) > 0:
                    additional_indices = available_indices[torch.randperm(len(available_indices))[:remaining_needed]]
                    selected_indices = torch.cat([selected_indices, additional_indices])
        
        # Validate selected indices
        if len(selected_indices) == 0:
            logger.error("No valid reference indices selected")
            return None
        
        # Ensure indices are within bounds
        selected_indices = selected_indices[selected_indices < n_samples]
        
        if len(selected_indices) == 0:
            logger.error("All selected indices are out of bounds")
            return None
        
        reference_data = {
            'input_ids': input_ids[selected_indices],
            'attention_mask': attention_mask[selected_indices],
            'scores': scores[selected_indices]
        }
        
        # Final validation of reference data
        ref_scores = reference_data['scores']
        if torch.any(torch.isnan(ref_scores)) or torch.any(torch.isinf(ref_scores)):
            logger.warning("Invalid reference scores found after selection, cleaning...")
            ref_scores = torch.clamp(ref_scores, 0.0, 1.0)
            ref_scores[torch.isnan(ref_scores)] = 0.5
            ref_scores[torch.isinf(ref_scores)] = 0.5
            reference_data['scores'] = ref_scores
        
        logger.info(f"Selected {len(selected_indices)} reference samples")
        logger.info(f"Reference score range: {ref_scores.min().item():.4f} - {ref_scores.max().item():.4f}")
        
        return reference_data
        
    except Exception as e:
        logger.error(f"Error preparing reference samples: {e}")
        return None


class PairwiseDataset(torch.utils.data.Dataset):
    """Dataset for pairwise training with error handling"""
    
    def __init__(self, pairwise_data):
        if pairwise_data is None:
            raise ValueError("Pairwise data cannot be None")
        
        self.input_ids1 = pairwise_data['input_ids1']
        self.attention_mask1 = pairwise_data['attention_mask1']
        self.input_ids2 = pairwise_data['input_ids2']
        self.attention_mask2 = pairwise_data['attention_mask2']
        self.relative_scores = pairwise_data['relative_scores']
        
        # Validate all tensors have the same length
        lengths = [
            len(self.input_ids1),
            len(self.attention_mask1), 
            len(self.input_ids2),
            len(self.attention_mask2),
            len(self.relative_scores)
        ]
        
        if not all(l == lengths[0] for l in lengths):
            raise ValueError(f"Inconsistent tensor lengths: {lengths}")
        
        logger.info(f"PairwiseDataset initialized with {len(self.relative_scores)} pairs")
    
    def __len__(self):
        return len(self.relative_scores)
    
    def __getitem__(self, idx):
        try:
            return (
                self.input_ids1[idx],
                self.attention_mask1[idx],
                self.input_ids2[idx],
                self.attention_mask2[idx],
                self.relative_scores[idx]
            )
        except IndexError as e:
            logger.error(f"Index error in PairwiseDataset: {e}")
            # Return a safe default
            return (
                torch.zeros_like(self.input_ids1[0]),
                torch.zeros_like(self.attention_mask1[0]),
                torch.zeros_like(self.input_ids2[0]),
                torch.zeros_like(self.attention_mask2[0]),
                torch.tensor(0.5)
            )