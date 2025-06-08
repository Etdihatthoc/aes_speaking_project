# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class NPCRModel(nn.Module):
    """Neural Pairwise Contrastive Regression Model for AES"""
    
    def __init__(self, config):
        super(NPCRModel, self).__init__()
        
        # Load pretrained model
        self.pretrained_model = config['model']['pretrained_model']
        self.embedding = AutoModel.from_pretrained(self.pretrained_model)
        
        # Get hidden dimension from pretrained model
        self.hidden_dim = self.embedding.config.hidden_size
        
        # Neural network layers
        self.dropout = nn.Dropout(config['model']['dropout'])
        
        # Feature extractor (shared between nn1 and nn2 in the paper)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            self.dropout
        )
        
        # Output layer (nn3 in the paper) - no bias for antisymmetry
        self.output = nn.Linear(self.hidden_dim, 1, bias=False)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights following the paper"""
        # Initialize linear layers with Xavier uniform
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def get_essay_representation(self, input_ids, attention_mask):
        """Get essay representation using pretrained model"""
        outputs = self.embedding(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        essay_repr = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_dim)
        
        return essay_repr
    
    def forward_single(self, input_ids, attention_mask):
        """Forward pass for single essay (for regression)"""
        essay_repr = self.get_essay_representation(input_ids, attention_mask)
        features = self.feature_extractor(essay_repr)
        score = torch.sigmoid(self.output(features))
        return score.squeeze(-1)
    
    def forward_pair(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        """Forward pass for essay pair (for contrastive learning)"""
        # Get representations for both essays
        essay_repr1 = self.get_essay_representation(input_ids1, attention_mask1)
        essay_repr2 = self.get_essay_representation(input_ids2, attention_mask2)
        
        # Extract features using shared feature extractor
        features1 = self.feature_extractor(essay_repr1)
        features2 = self.feature_extractor(essay_repr2)
        
        # Calculate difference vector
        diff_vector = features1 - features2
        
        # Get relative score
        relative_score = self.output(diff_vector)
        
        # Apply sigmoid to bound the output
        relative_score = torch.sigmoid(relative_score)
        
        return relative_score.squeeze(-1)
    
    def forward(self, input_ids, attention_mask, input_ids2=None, attention_mask2=None):
        """Unified forward pass"""
        if input_ids2 is not None and attention_mask2 is not None:
            # Pairwise mode
            return self.forward_pair(input_ids, attention_mask, input_ids2, attention_mask2)
        else:
            # Single essay mode
            return self.forward_single(input_ids, attention_mask)


class NPCRModelWithMultiSampleVoting(nn.Module):
    """NPCR Model with multi-sample voting for inference"""
    
    def __init__(self, base_model, config):
        super(NPCRModelWithMultiSampleVoting, self).__init__()
        self.base_model = base_model
        self.example_size = config['training']['example_size']
    
    def forward(self, input_ids, attention_mask, reference_ids=None, reference_masks=None, reference_scores=None):
        """
        Forward pass with multi-sample voting
        
        Args:
            input_ids: Input essay token IDs
            attention_mask: Input essay attention mask
            reference_ids: Reference essays token IDs (for inference)
            reference_masks: Reference essays attention masks
            reference_scores: Reference essays scores
        """
        if reference_ids is None:
            # Training mode - single essay prediction
            return self.base_model(input_ids, attention_mask)
        else:
            # Inference mode - multi-sample voting
            batch_size = input_ids.size(0)
            num_references = reference_ids.size(0)
            
            predictions = []
            
            for i in range(batch_size):
                # Get current essay
                curr_input_ids = input_ids[i:i+1].expand(num_references, -1)
                curr_attention_mask = attention_mask[i:i+1].expand(num_references, -1)
                
                # Predict relative scores with all references
                relative_scores = self.base_model.forward_pair(
                    curr_input_ids,
                    curr_attention_mask,
                    reference_ids,
                    reference_masks
                )
                
                # Add reference scores to get absolute scores
                absolute_scores = relative_scores + reference_scores
                
                # Average across all references
                avg_score = absolute_scores.mean()
                predictions.append(avg_score)
            
            return torch.stack(predictions)