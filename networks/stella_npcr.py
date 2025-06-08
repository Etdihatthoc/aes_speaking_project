# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
import math


class AttentionPooling(nn.Module):
    """Attention pooling layer for segment-scale representation"""
    def __init__(self, hidden_size, attention_size):
        super(AttentionPooling, self).__init__()
        self.w_omega = nn.Linear(hidden_size, attention_size, bias=False)
        self.b_omega = nn.Parameter(torch.zeros(attention_size))
        self.u_omega = nn.Linear(attention_size, 1, bias=False)
        
    def forward(self, hidden_states, mask=None):
        # hidden_states: (batch_size, seq_len, hidden_size)
        attention_w = torch.tanh(self.w_omega(hidden_states) + self.b_omega)
        attention_scores = self.u_omega(attention_w).squeeze(-1)  # (batch_size, seq_len)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)
        weighted_output = (hidden_states * attention_weights).sum(dim=1)
        
        return weighted_output


class StellaNPCRModel(nn.Module):
    """NPCR Model using Stella embeddings with Multi-Scale Representation"""
    
    def __init__(self, config):
        super(StellaNPCRModel, self).__init__()
        
        # Load Stella model
        self.pretrained_model = config['model']['pretrained_model']
        self.base_model = AutoModel.from_pretrained(self.pretrained_model, trust_remote_code=True)
        
        # Stella uses 1024-dim embeddings
        self.base_hidden_dim = 1024
        self.max_length = int(config['model']['max_length'])
        
        # Multi-scale settings
        self.use_multi_scale = config.get('multi_scale', {}).get('use_multi_scale', True)
        self.segment_sizes = config.get('multi_scale', {}).get('segment_sizes', [30, 50, 70, 90, 110])
        
        # Calculate total hidden dimension
        if self.use_multi_scale:
            # Document + Token scales
            self.hidden_dim = self.base_hidden_dim * 2
            
            # LSTM for segment processing
            lstm_hidden = int(config.get('multi_scale', {}).get('lstm_hidden_size', 256))
            lstm_layers = int(config.get('multi_scale', {}).get('lstm_num_layers', 1))
            lstm_dropout = float(config.get('multi_scale', {}).get('lstm_dropout', 0.1))
            attention_size = int(config.get('multi_scale', {}).get('attention_size', 256))
            
            self.segment_lstm = nn.LSTM(
                self.base_hidden_dim,
                lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=lstm_dropout if lstm_layers > 1 else 0,
                bidirectional=True
            )
            
            self.attention_pooling = AttentionPooling(lstm_hidden * 2, attention_size)
            
            # Add segment dimensions
            self.hidden_dim += lstm_hidden * 2 * len(self.segment_sizes)
        else:
            self.hidden_dim = self.base_hidden_dim
        
        # Dropout
        self.dropout = nn.Dropout(float(config['model']['dropout']))
        
        # Feature extractor layers (following NPCR design)
        feature_hidden = min(512, self.hidden_dim // 2)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            self.dropout,
            nn.Linear(self.hidden_dim, feature_hidden),
            nn.Tanh(),
            self.dropout
        )
        
        # Output layer for score prediction (no bias for antisymmetry)
        self.output = nn.Linear(feature_hidden, 1, bias=False)
        
        # Additional regularization for Stella
        self.layer_norm = nn.LayerNorm(self.base_hidden_dim)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling - Take attention mask into account for correct averaging"""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_base_representations(self, input_ids, attention_mask):
        """Get base model representations"""
        model_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get last hidden state
        last_hidden_state = model_output.last_hidden_state
        
        # Apply mean pooling for document representation
        sentence_embeddings = self.mean_pooling(model_output, attention_mask)
        
        # Normalize embeddings (important for Stella)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        # Apply layer norm for stability
        sentence_embeddings = self.layer_norm(sentence_embeddings)
        
        return last_hidden_state, sentence_embeddings
    
    def get_token_scale_representation(self, last_hidden_state, attention_mask):
        """Extract token-scale representation using max-pooling"""
        # Apply layer norm first
        last_hidden_state = self.layer_norm(last_hidden_state)
        
        # Mask out padding tokens
        expanded_mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state)
        last_hidden_state = last_hidden_state.masked_fill(expanded_mask == 0, -1e9)
        
        # Max pooling over sequence length
        token_repr, _ = torch.max(last_hidden_state, dim=1)
        
        # Normalize
        token_repr = F.normalize(token_repr, p=2, dim=1)
        
        return token_repr
    
    def get_segment_scale_representation(self, input_ids, attention_mask, segment_size):
        """Extract segment-scale representation"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Calculate number of segments
        num_segments = math.ceil(seq_len / segment_size)
        
        # Prepare to collect segment representations
        segment_reprs = []
        segment_masks = []
        
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, seq_len)
            
            # Extract segment
            segment_input_ids = input_ids[:, start_idx:end_idx]
            segment_attention_mask = attention_mask[:, start_idx:end_idx]
            
            # Pad if necessary
            if end_idx - start_idx < segment_size:
                pad_length = segment_size - (end_idx - start_idx)
                segment_input_ids = F.pad(segment_input_ids, (0, pad_length), value=0)
                segment_attention_mask = F.pad(segment_attention_mask, (0, pad_length), value=0)
            
            # Get segment representation
            with torch.no_grad():
                _, segment_embedding = self.get_base_representations(
                    segment_input_ids, segment_attention_mask
                )
            
            segment_reprs.append(segment_embedding)
            
            # Create mask for this segment
            has_content = segment_attention_mask.sum(dim=1) > 0
            segment_masks.append(has_content)
        
        # Stack segment representations
        segment_reprs = torch.stack(segment_reprs, dim=1)  # (batch_size, num_segments, hidden_size)
        segment_masks = torch.stack(segment_masks, dim=1).float()  # (batch_size, num_segments)
        
        # Process through LSTM
        lstm_output, _ = self.segment_lstm(segment_reprs)
        
        # Apply attention pooling
        segment_final = self.attention_pooling(lstm_output, segment_masks)
        
        return segment_final
    
    def get_multi_scale_representation(self, input_ids, attention_mask):
        """Get multi-scale essay representation"""
        # Get base representations
        last_hidden_state, doc_repr = self.get_base_representations(input_ids, attention_mask)
        
        if not self.use_multi_scale:
            return doc_repr
        
        # Token-scale representation
        token_repr = self.get_token_scale_representation(last_hidden_state, attention_mask)
        
        # Segment-scale representations
        segment_reprs = []
        for segment_size in self.segment_sizes:
            segment_repr = self.get_segment_scale_representation(
                input_ids, attention_mask, segment_size
            )
            segment_reprs.append(segment_repr)
        
        # Concatenate all representations
        multi_scale_repr = torch.cat([doc_repr, token_repr] + segment_reprs, dim=-1)
        
        return multi_scale_repr
    
    def forward_single(self, input_ids, attention_mask):
        """Single essay scoring"""
        # Get multi-scale representation
        essay_embedding = self.get_multi_scale_representation(input_ids, attention_mask)
        
        # Extract features
        features = self.feature_extractor(essay_embedding)
        
        # Predict score
        score = torch.sigmoid(self.output(features))
        
        return score.squeeze(-1)
    
    def forward_pair(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        """Pairwise comparison for contrastive learning"""
        # Get embeddings for both essays
        embedding1 = self.get_multi_scale_representation(input_ids1, attention_mask1)
        embedding2 = self.get_multi_scale_representation(input_ids2, attention_mask2)
        
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
        embedding1 = self.get_multi_scale_representation(input_ids1, attention_mask1)
        embedding2 = self.get_multi_scale_representation(input_ids2, attention_mask2)
        
        # Cosine similarity
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
        
        return similarity