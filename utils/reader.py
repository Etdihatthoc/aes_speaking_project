# -*- coding: utf-8 -*-
import os
# Fix tokenizers parallelism warning - MUST be before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from utils.utils import get_logger, normalize_scores

logger = get_logger("Data Reader")


class SpeakingDataReader:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained_model'])
        self.max_length = config['model']['max_length']
        self.score_type = config['model']['type']
        self.min_score = config['score']['min_score']
        self.max_score = config['score']['max_score']
        
    def read_csv(self, csv_path):
        """Read CSV file and return data"""
        logger.info(f"Reading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Remove any rows with NaN values
        original_len = len(df)
        df = df.dropna(subset=['text', self.score_type])
        if len(df) < original_len:
            logger.warning(f"Dropped {original_len - len(df)} rows with NaN values")
        
        # Remove empty texts
        df = df[df['text'].str.strip() != '']
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Get text and scores based on type
        texts = df['text'].tolist()
        
        if self.score_type == 'vocabulary':
            scores = df['vocabulary'].tolist()
        elif self.score_type == 'grammar':
            scores = df['grammar'].tolist() if 'grammar' in df.columns else df['vocabulary'].tolist()
        elif self.score_type == 'content':
            scores = df['content'].tolist() if 'content' in df.columns else df['vocabulary'].tolist()
        else:
            raise ValueError(f"Unknown score type: {self.score_type}")
        
        # Convert scores to float and validate
        scores = [float(score) for score in scores]
        
        # Validate texts and scores
        valid_indices = []
        for i, (text, score) in enumerate(zip(texts, scores)):
            if isinstance(text, str) and len(text.strip()) > 0 and not np.isnan(score):
                valid_indices.append(i)
        
        texts = [texts[i] for i in valid_indices]
        scores = [scores[i] for i in valid_indices]
        
        logger.info(f"Loaded {len(texts)} valid samples")
        logger.info(f"Score range: {min(scores)} - {max(scores)}")
        
        return texts, scores
    
    def tokenize_texts(self, texts):
        """Tokenize texts using the pretrained tokenizer"""
        # Add validation
        valid_texts = []
        for text in texts:
            if isinstance(text, str) and len(text.strip()) > 0:
                valid_texts.append(text)
            else:
                valid_texts.append("empty text")  # Fallback
                
        encoded = self.tokenizer(
            valid_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        # Ensure no empty sequences
        attention_sum = encoded['attention_mask'].sum(dim=1)
        if (attention_sum == 0).any():
            logger.warning("Found empty sequences after tokenization!")
            # Replace empty sequences with at least [CLS] and [SEP] tokens
            empty_mask = attention_sum == 0
            encoded['attention_mask'][empty_mask, :2] = 1
            
        return encoded['input_ids'], encoded['attention_mask']
    
    def prepare_data(self, csv_path):
        """Prepare data for training/evaluation"""
        texts, scores = self.read_csv(csv_path)
        
        # Tokenize texts
        input_ids, attention_masks = self.tokenize_texts(texts)
        
        # Normalize scores to [0, 1]
        normalized_scores = normalize_scores(
            np.array(scores), 
            self.min_score, 
            self.max_score
        )
        
        # Validate normalized scores
        normalized_scores = np.clip(normalized_scores, 0.0, 1.0)
        
        # Convert to tensors
        scores_tensor = torch.FloatTensor(normalized_scores)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'scores': scores_tensor,
            'original_scores': torch.FloatTensor(scores),
            'texts': texts
        }


def get_data_loaders(config):
    """Get data loaders for training, validation, and testing"""
    reader = SpeakingDataReader(config)
    
    # Prepare datasets
    train_data = reader.prepare_data(config['data']['train_csv_path'])
    val_data = reader.prepare_data(config['data']['val_csv_path'])
    test_data = reader.prepare_data(config['data']['test_csv_path'])
    
    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(
        train_data['input_ids'],
        train_data['attention_mask'],
        train_data['scores'],
        train_data['original_scores']
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        val_data['input_ids'],
        val_data['attention_mask'],
        val_data['scores'],
        val_data['original_scores']
    )
    
    test_dataset = torch.utils.data.TensorDataset(
        test_data['input_ids'],
        test_data['attention_mask'],
        test_data['scores'],
        test_data['original_scores']
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['device']['num_workers'],
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['device']['num_workers'],
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['device']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_data, val_data, test_data