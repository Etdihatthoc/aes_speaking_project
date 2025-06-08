# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
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
            # Use -inf for numerical stability, but not too large
            attention_scores = attention_scores.masked_fill(mask == 0, -1e4)
        
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)
        weighted_output = (hidden_states * attention_weights).sum(dim=1)
        
        return weighted_output


class NPCRModel(nn.Module):
    """Neural Pairwise Contrastive Regression Model with Multi-Scale Representation"""
    
    def __init__(self, config):
        super(NPCRModel, self).__init__()
        
        # Load pretrained model
        self.pretrained_model = config['model']['pretrained_model']
        self.max_length = int(config['model']['max_length'])
        
        # Multi-scale settings
        self.use_multi_scale = config.get('multi_scale', {}).get('use_multi_scale', True)
        self.segment_sizes = config.get('multi_scale', {}).get('segment_sizes', [30, 50, 70, 90, 110])
        
        # Check if using Stella or other sentence transformer
        if 'stella' in self.pretrained_model.lower() or 'sentence-transformers' in self.pretrained_model:
            self.use_sentence_transformer = True
            self.embedding = SentenceTransformer(self.pretrained_model)
            self.base_hidden_dim = self.embedding.get_sentence_embedding_dimension()
            # For segment processing, we need the base transformer
            self.base_transformer = self.embedding[0].auto_model
        else:
            self.use_sentence_transformer = False
            self.embedding = AutoModel.from_pretrained(self.pretrained_model)
            self.base_hidden_dim = self.embedding.config.hidden_size
            self.base_transformer = self.embedding
        
        # Calculate total hidden dimension based on multi-scale
        if self.use_multi_scale:
            # Document-scale (CLS) + Token-scale (max-pool) + Segment-scales
            self.hidden_dim = self.base_hidden_dim * 2  # Document + Token scales
            
            # LSTM for segment-scale processing
            lstm_hidden = int(config.get('multi_scale', {}).get('lstm_hidden_size', 256))
            lstm_layers = int(config.get('multi_scale', {}).get('lstm_num_layers', 1))
            lstm_dropout = float(config.get('multi_scale', {}).get('lstm_dropout', 0.1))
            attention_size = int(config.get('multi_scale', {}).get('attention_size', 256))
            
            # Create separate LSTM for each segment size (to avoid shared parameters issues)
            self.segment_lstms = nn.ModuleList([
                nn.LSTM(
                    self.base_hidden_dim,
                    lstm_hidden,
                    num_layers=lstm_layers,
                    batch_first=True,
                    dropout=lstm_dropout if lstm_layers > 1 else 0,
                    bidirectional=True
                ) for _ in self.segment_sizes
            ])
            
            self.attention_poolings = nn.ModuleList([
                AttentionPooling(lstm_hidden * 2, attention_size)
                for _ in self.segment_sizes
            ])
            
            # Add segment scale dimensions
            self.hidden_dim += lstm_hidden * 2 * len(self.segment_sizes)
        else:
            self.hidden_dim = self.base_hidden_dim
        
        # Neural network layers
        self.dropout = nn.Dropout(float(config['model']['dropout']))
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # Feature extractor (shared between nn1 and nn2 in the paper)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),  # Add layer norm for stability
            nn.Tanh(),
            self.dropout
        )
        
        # Output layer (nn3 in the paper) - no bias for antisymmetry
        self.output = nn.Linear(self.hidden_dim, 1, bias=False)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights following the paper"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization for stability
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=0.5)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def get_base_representations(self, input_ids, attention_mask):
        """Get base BERT representations"""
        outputs = self.base_transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        last_hidden_state = outputs.last_hidden_state
        # Use mean pooling for document representation (more stable than CLS)
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooler_output = sum_embeddings / sum_mask
        
        return last_hidden_state, pooler_output
    
    def get_token_scale_representation(self, last_hidden_state, attention_mask):
        """Extract token-scale representation using max-pooling"""
        # Mask out padding tokens with smaller value for stability
        expanded_mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state)
        last_hidden_state = last_hidden_state.masked_fill(expanded_mask == 0, -100.0)
        
        # Max pooling over sequence length
        token_repr, _ = torch.max(last_hidden_state, dim=1)
        
        # Replace -100 with 0 for masked positions
        token_repr = torch.where(token_repr == -100.0, torch.zeros_like(token_repr), token_repr)
        
        return token_repr
    
    def get_segment_scale_representation(self, input_ids, attention_mask, segment_size, lstm_idx):
        """Extract segment-scale representation"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Calculate number of segments
        num_segments = min(math.ceil(seq_len / segment_size), 20)  # Cap at 20 segments
        
        # Prepare to collect segment representations
        segment_reprs = []
        segment_masks = []
        
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, seq_len)
            
            # Extract segment
            segment_input_ids = input_ids[:, start_idx:end_idx]
            segment_attention_mask = attention_mask[:, start_idx:end_idx]
            
            # Check if segment has any content
            has_content = segment_attention_mask.sum(dim=1) > 0
            
            if not has_content.any():
                # If no content, use zeros
                segment_repr = torch.zeros(batch_size, self.base_hidden_dim, device=device)
            else:
                # Pad if necessary
                if end_idx - start_idx < segment_size:
                    pad_length = segment_size - (end_idx - start_idx)
                    segment_input_ids = F.pad(segment_input_ids, (0, pad_length), value=0)
                    segment_attention_mask = F.pad(segment_attention_mask, (0, pad_length), value=0)
                
                # Get segment representation (WITH gradients)
                _, segment_repr = self.get_base_representations(
                    segment_input_ids, segment_attention_mask
                )
            
            segment_reprs.append(segment_repr)
            segment_masks.append(has_content.float())
        
        # Stack segment representations
        segment_reprs = torch.stack(segment_reprs, dim=1)  # (batch_size, num_segments, hidden_size)
        segment_masks = torch.stack(segment_masks, dim=1)  # (batch_size, num_segments)
        
        # Process through LSTM
        lstm_output, _ = self.segment_lstms[lstm_idx](segment_reprs)
        
        # Apply attention pooling
        segment_final = self.attention_poolings[lstm_idx](lstm_output, segment_masks)
        
        return segment_final
    
    def get_multi_scale_representation(self, input_ids, attention_mask):
        """Get multi-scale essay representation"""
        # Get base representations
        last_hidden_state, pooler_output = self.get_base_representations(input_ids, attention_mask)
        
        if not self.use_multi_scale:
            # Use only document-scale
            return pooler_output
        
        # Document-scale representation
        doc_repr = pooler_output
        
        # Token-scale representation (max-pooling)
        token_repr = self.get_token_scale_representation(last_hidden_state, attention_mask)
        
        # Segment-scale representations
        segment_reprs = []
        for idx, segment_size in enumerate(self.segment_sizes):
            segment_repr = self.get_segment_scale_representation(
                input_ids, attention_mask, segment_size, idx
            )
            segment_reprs.append(segment_repr)
        
        # Concatenate all representations
        multi_scale_repr = torch.cat([doc_repr, token_repr] + segment_reprs, dim=-1)
        
        # Apply layer norm for stability
        multi_scale_repr = self.layer_norm(multi_scale_repr)
        
        return multi_scale_repr
    
    def forward_single(self, input_ids, attention_mask):
        """Forward pass for single essay (for regression)"""
        essay_repr = self.get_multi_scale_representation(input_ids, attention_mask)
        features = self.feature_extractor(essay_repr)
        score = torch.sigmoid(self.output(features))
        return score.squeeze(-1)
    
    def forward_pair(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        """Forward pass for essay pair (for contrastive learning)"""
        # Get representations for both essays
        essay_repr1 = self.get_multi_scale_representation(input_ids1, attention_mask1)
        essay_repr2 = self.get_multi_scale_representation(input_ids2, attention_mask2)
        
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
        # Check for empty inputs
        if (attention_mask == 0).all():
            raise ValueError("Empty input detected (all attention mask values are 0)")
            
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
        self.example_size = int(config['training']['example_size'])
    
    def forward(self, input_ids, attention_mask, reference_ids=None, reference_masks=None, reference_scores=None):
        """
        Forward pass with multi-sample voting
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