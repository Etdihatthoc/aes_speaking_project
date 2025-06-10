# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import sys
import os

# Add the current directory to path to import Extract_emb
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Extract_emb import MultiScaleEssayEncoder


class NPCRModel(nn.Module):
    """Neural Pairwise Contrastive Regression Model for AES with Multi-Scale Essay Representation"""
    
    def __init__(self, config):
        super(NPCRModel, self).__init__()
        
        # Load pretrained model info
        self.pretrained_model = config['model']['pretrained_model']
        
        # Check if using Stella or other sentence transformer
        if 'stella' in self.pretrained_model.lower() or 'sentence-transformers' in self.pretrained_model:
            self.use_sentence_transformer = True
            self.embedding = SentenceTransformer(self.pretrained_model)
            # For Stella, use simpler representation for now
            self.hidden_dim = self.embedding.get_sentence_embedding_dimension()
            self.multi_scale_encoder = None
        else:
            self.use_sentence_transformer = False
            # Initialize multi-scale encoder
            self.multi_scale_encoder = MultiScaleEssayEncoder(config)
            self.hidden_dim = self.multi_scale_encoder.get_combined_hidden_dim()
        
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
        """Get essay representation using multi-scale encoder or sentence transformer"""
        if self.use_sentence_transformer:
            # For Stella/SentenceTransformer (fallback to original implementation)
            batch_size = input_ids.size(0)
            embeddings = []
            
            # Process each sample in batch
            for i in range(batch_size):
                # Get the actual length (remove padding)
                mask = attention_mask[i].bool()
                tokens = input_ids[i][mask]
                
                # Stella expects raw text or tokens, we'll use the embedding directly
                with torch.no_grad():
                    # Get embeddings using the model's encode method
                    outputs = self.embedding._modules['0'].auto_model(
                        input_ids=input_ids[i:i+1],
                        attention_mask=attention_mask[i:i+1]
                    )
                    # Apply pooling
                    embeddings_i = self.embedding._modules['0'].pooling(
                        outputs, 
                        attention_mask[i:i+1]
                    )
                    embeddings.append(embeddings_i['sentence_embedding'])
            
            essay_repr = torch.cat(embeddings, dim=0)
        else:
            # Use multi-scale encoder
            try:
                essay_repr = self.multi_scale_encoder(input_ids, attention_mask)
            except Exception as e:
                print(f"Warning: Multi-scale encoding failed, falling back to simple BERT: {e}")
                # Fallback to simple BERT [CLS] token
                simple_bert = AutoModel.from_pretrained(self.pretrained_model)
                outputs = simple_bert(input_ids=input_ids, attention_mask=attention_mask)
                essay_repr = outputs.last_hidden_state[:, 0, :]
        
        return essay_repr
    
    def forward_single(self, input_ids, attention_mask):
        """Forward pass for single essay (for regression)"""
        essay_repr = self.get_essay_representation(input_ids, attention_mask)
        features = self.feature_extractor(essay_repr)
        score = torch.sigmoid(self.output(features))
        
        # Ensure score is in valid range [0, 1]
        score = torch.clamp(score, min=0.0, max=1.0)
        
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
        
        # Ensure relative score is in valid range [0, 1]
        relative_score = torch.clamp(relative_score, min=0.0, max=1.0)
        
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
                try:
                    relative_scores = self.base_model.forward_pair(
                        curr_input_ids,
                        curr_attention_mask,
                        reference_ids,
                        reference_masks
                    )
                    
                    # Ensure relative_scores are valid
                    relative_scores = torch.clamp(relative_scores, min=0.0, max=1.0)
                    
                    # Convert relative scores from [0,1] to [-1,1] for proper calculation
                    relative_scores = (relative_scores * 2) - 1
                    
                    # Add reference scores to get absolute scores
                    absolute_scores = relative_scores + reference_scores
                    
                    # Ensure absolute scores are in valid range
                    absolute_scores = torch.clamp(absolute_scores, min=0.0, max=1.0)
                    
                    # Average across all references
                    avg_score = absolute_scores.mean()
                    
                    # Ensure final score is valid
                    avg_score = torch.clamp(avg_score, min=0.0, max=1.0)
                    
                    predictions.append(avg_score.cpu().numpy())
                    
                except Exception as e:
                    print(f"Warning: Error in multi-sample voting for sample {i}: {e}")
                    # Fallback to single mode prediction
                    single_score = self.base_model(input_ids[i:i+1], attention_mask[i:i+1])
                    predictions.append(single_score.squeeze().cpu().numpy())
            
            return torch.stack([torch.tensor(p) for p in predictions])