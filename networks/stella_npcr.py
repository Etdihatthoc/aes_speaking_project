# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np


class StellaNPCRModel(nn.Module):
    """NPCR Model using Stella embeddings"""
    
    def __init__(self, config):
        super(StellaNPCRModel, self).__init__()
        
        # Load Stella model
        self.pretrained_model = config['model']['pretrained_model']
        self.base_model = AutoModel.from_pretrained(self.pretrained_model, trust_remote_code=True)
        
        # Stella uses 1024-dim embeddings
        self.hidden_dim = 1024
        
        # Dropout
        self.dropout = nn.Dropout(config['model']['dropout'])
        
        # Feature extractor layers (following NPCR design)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            self.dropout,
            nn.Linear(self.hidden_dim, 512),
            nn.Tanh(),
            self.dropout
        )
        
        # Output layer for score prediction (no bias for antisymmetry)
        self.output = nn.Linear(512, 1, bias=False)
        
        # Additional regularization for Stella
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling - Take attention mask into account for correct averaging"""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_essay_embedding(self, input_ids, attention_mask):
        """Get essay embedding using Stella"""
        # Get model output
        model_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Apply mean pooling
        sentence_embeddings = self.mean_pooling(model_output, attention_mask)
        
        # Normalize embeddings (important for Stella)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        # Apply layer norm for stability
        sentence_embeddings = self.layer_norm(sentence_embeddings)
        
        return sentence_embeddings
    
    def forward_single(self, input_ids, attention_mask):
        """Single essay scoring"""
        # Get essay embedding
        essay_embedding = self.get_essay_embedding(input_ids, attention_mask)
        
        # Extract features
        features = self.feature_extractor(essay_embedding)
        
        # Predict score
        score = torch.sigmoid(self.output(features))
        
        return score.squeeze(-1)
    
    def forward_pair(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        """Pairwise comparison for contrastive learning"""
        # Get embeddings for both essays
        embedding1 = self.get_essay_embedding(input_ids1, attention_mask1)
        embedding2 = self.get_essay_embedding(input_ids2, attention_mask2)
        
        # Extract features
        features1 = self.feature_extractor(embedding1)
        features2 = self.feature_extractor(embedding2)
        
        # Calculate difference (for relative scoring)
        diff_vector = features1 - features2
        
        # Predict relative score
        relative_score = torch.sigmoid(self.output(diff_vector))
        
        return relative_score.squeeze(-1)
    
    def forward(self, input_ids, attention_mask, input_ids2=None, attention_mask2=None):
        """Unified forward pass"""
        if input_ids2 is not None and attention_mask2 is not None:
            return self.forward_pair(input_ids, attention_mask, input_ids2, attention_mask2)
        else:
            return self.forward_single(input_ids, attention_mask)
    
    def get_similarity_score(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        """Get cosine similarity between two essays (useful for analysis)"""
        embedding1 = self.get_essay_embedding(input_ids1, attention_mask1)
        embedding2 = self.get_essay_embedding(input_ids2, attention_mask2)
        
        # Cosine similarity
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
        
        return similarity