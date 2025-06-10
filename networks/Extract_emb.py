# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import math


class MultiScaleEssayEncoder(nn.Module):
    """
    Multi-Scale Essay Representation Encoder
    Based on "On the Use of BERT for Automated Essay Scoring: Joint Learning of Multi-Scale Essay Representation"
    
    Extracts features at three scales:
    1. Token-scale: Max pooling over all token representations
    2. Document-scale: [CLS] token representation  
    3. Segment-scale: LSTM + attention over text segments
    """
    
    def __init__(self, config):
        super(MultiScaleEssayEncoder, self).__init__()
        
        # Load pretrained BERT model
        self.bert_model = AutoModel.from_pretrained(config['model']['pretrained_model'])
        self.hidden_dim = self.bert_model.config.hidden_size
        self.max_length = config['model']['max_length']
        
        # Segment configuration
        self.segment_sizes = [50, 100, 150]  # Different segment sizes as in the paper
        
        # LSTM for segment-scale processing
        self.lstm_hidden_size = self.hidden_dim
        self.segment_lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.lstm_hidden_size,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism for segment pooling
        self.attention_dim = self.hidden_dim
        self.w_omega = nn.Parameter(torch.Tensor(self.lstm_hidden_size, self.attention_dim))
        self.b_omega = nn.Parameter(torch.Tensor(1, self.attention_dim))
        self.u_omega = nn.Parameter(torch.Tensor(self.attention_dim, 1))
        
        # Initialize attention parameters
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.b_omega, -0.1, 0.1)
        
        # Dropout
        self.dropout = nn.Dropout(config['model']['dropout'])
        
    def get_document_and_token_scale(self, input_ids, attention_mask):
        """
        Extract document-scale and token-scale representations
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            document_repr: Document-scale representation [batch_size, hidden_dim]
            token_repr: Token-scale representation [batch_size, hidden_dim]
        """
        # Get BERT output
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Document-scale: [CLS] token representation
        document_repr = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        
        # Token-scale: Max pooling over all token representations
        token_embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
        # Apply attention mask for proper pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings_masked = token_embeddings * mask_expanded
        
        # Max pooling
        token_repr, _ = torch.max(token_embeddings_masked, dim=1)  # [batch_size, hidden_dim]
        
        return document_repr, token_repr
    
    def create_segments(self, input_ids, attention_mask, segment_size):
        """
        Create segments from input text
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            segment_size: Size of each segment
            
        Returns:
            segment_input_ids: List of segment token IDs
            segment_attention_masks: List of segment attention masks
        """
        batch_size, seq_len = input_ids.size()
        
        # Calculate number of segments
        effective_length = seq_len - 2  # Exclude [CLS] and [SEP]
        num_segments = math.ceil(effective_length / segment_size)
        
        segment_input_ids = []
        segment_attention_masks = []
        
        for batch_idx in range(batch_size):
            batch_segments_ids = []
            batch_segments_masks = []
            
            # Find actual sequence length (excluding padding)
            actual_length = attention_mask[batch_idx].sum().item()
            actual_length = min(actual_length - 2, effective_length)  # Exclude [CLS] and [SEP]
            
            for seg_idx in range(num_segments):
                start_idx = seg_idx * segment_size + 1  # +1 to skip [CLS]
                end_idx = min(start_idx + segment_size, actual_length + 1)
                
                if start_idx >= actual_length + 1:
                    # Create empty segment if beyond actual text
                    segment_ids = torch.zeros(segment_size + 2, dtype=torch.long, device=input_ids.device)
                    segment_ids[0] = input_ids[batch_idx, 0]  # [CLS]
                    segment_ids[-1] = input_ids[batch_idx, -1]  # [SEP] or [PAD]
                    
                    segment_mask = torch.zeros(segment_size + 2, dtype=torch.long, device=attention_mask.device)
                    segment_mask[0] = 1  # [CLS]
                else:
                    # Extract segment tokens
                    segment_tokens = input_ids[batch_idx, start_idx:end_idx]
                    
                    # Create segment with [CLS] and [SEP]
                    segment_ids = torch.zeros(segment_size + 2, dtype=torch.long, device=input_ids.device)
                    segment_ids[0] = input_ids[batch_idx, 0]  # [CLS]
                    
                    actual_seg_len = min(len(segment_tokens), segment_size)
                    segment_ids[1:1+actual_seg_len] = segment_tokens[:actual_seg_len]
                    segment_ids[1+actual_seg_len] = input_ids[batch_idx, min(seq_len-1, actual_length+1)]  # [SEP]
                    
                    # Create attention mask
                    segment_mask = torch.zeros(segment_size + 2, dtype=torch.long, device=attention_mask.device)
                    segment_mask[0] = 1  # [CLS]
                    segment_mask[1:1+actual_seg_len] = 1  # Actual tokens
                    if 1+actual_seg_len < segment_size + 2:
                        segment_mask[1+actual_seg_len] = 1  # [SEP]
                
                batch_segments_ids.append(segment_ids)
                batch_segments_masks.append(segment_mask)
            
            segment_input_ids.append(torch.stack(batch_segments_ids))
            segment_attention_masks.append(torch.stack(batch_segments_masks))
        
        return torch.stack(segment_input_ids), torch.stack(segment_attention_masks)
    
    def get_segment_scale_representation(self, input_ids, attention_mask, segment_size):
        """
        Extract segment-scale representation for a specific segment size
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            segment_size: Size of each segment
            
        Returns:
            segment_repr: Segment-scale representation [batch_size, hidden_dim]
        """
        batch_size = input_ids.size(0)
        
        # Create segments
        segment_input_ids, segment_attention_masks = self.create_segments(
            input_ids, attention_mask, segment_size
        )
        # segment_input_ids: [batch_size, num_segments, segment_size+2]
        # segment_attention_masks: [batch_size, num_segments, segment_size+2]
        
        num_segments = segment_input_ids.size(1)
        
        # Encode each segment
        segment_representations = []
        
        for batch_idx in range(batch_size):
            batch_segment_reprs = []
            
            for seg_idx in range(num_segments):
                seg_input_ids = segment_input_ids[batch_idx, seg_idx].unsqueeze(0)
                seg_attention_mask = segment_attention_masks[batch_idx, seg_idx].unsqueeze(0)
                
                # Get BERT representation for this segment
                with torch.no_grad():
                    seg_outputs = self.bert_model(
                        input_ids=seg_input_ids,
                        attention_mask=seg_attention_mask
                    )
                
                # Use [CLS] token of the segment
                seg_repr = seg_outputs.last_hidden_state[:, 0, :]  # [1, hidden_dim]
                batch_segment_reprs.append(seg_repr)
            
            segment_representations.append(torch.cat(batch_segment_reprs, dim=0))
        
        # Stack to get [batch_size, num_segments, hidden_dim]
        segment_representations = torch.stack(segment_representations)
        
        # Apply LSTM
        lstm_output, _ = self.segment_lstm(segment_representations)
        # lstm_output: [batch_size, num_segments, lstm_hidden_size]
        
        # Apply attention pooling
        attention_weights = torch.tanh(torch.matmul(lstm_output, self.w_omega) + self.b_omega)
        attention_weights = torch.matmul(attention_weights, self.u_omega)  # [batch_size, num_segments, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        segment_repr = torch.sum(lstm_output * attention_weights, dim=1)  # [batch_size, lstm_hidden_size]
        
        return segment_repr
    
    def forward(self, input_ids, attention_mask):
        """
        Extract multi-scale essay representation
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            combined_repr: Combined representation [batch_size, combined_hidden_dim]
        """
        # Get document and token scale representations
        document_repr, token_repr = self.get_document_and_token_scale(input_ids, attention_mask)
        
        # Combine document and token representations
        doc_token_repr = torch.cat([document_repr, token_repr], dim=-1)  # [batch_size, 2*hidden_dim]
        
        # Get segment scale representations for different segment sizes
        segment_reprs = []
        for segment_size in self.segment_sizes:
            try:
                segment_repr = self.get_segment_scale_representation(input_ids, attention_mask, segment_size)
                segment_reprs.append(segment_repr)
            except Exception as e:
                # If segment processing fails, create zero representation
                print(f"Warning: Segment processing failed for size {segment_size}: {e}")
                zero_repr = torch.zeros(input_ids.size(0), self.lstm_hidden_size, device=input_ids.device)
                segment_reprs.append(zero_repr)
        
        # Combine all segment representations
        if segment_reprs:
            combined_segment_repr = torch.stack(segment_reprs, dim=0).mean(dim=0)  # Average across segment sizes
        else:
            combined_segment_repr = torch.zeros(input_ids.size(0), self.lstm_hidden_size, device=input_ids.device)
        
        # Final combined representation
        combined_repr = torch.cat([doc_token_repr, combined_segment_repr], dim=-1)
        # [batch_size, 2*hidden_dim + lstm_hidden_size]
        
        return combined_repr
    
    def get_combined_hidden_dim(self):
        """Get the dimension of the combined representation"""
        return 2 * self.hidden_dim + self.lstm_hidden_size  # doc + token + segment