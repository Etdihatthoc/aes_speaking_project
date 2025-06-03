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
        df =  df.sample(frac=1, random_state=42).reset_index(drop=True)
        
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
        
        # Convert scores to float
        scores = [float(score) for score in scores]
        
        logger.info(f"Loaded {len(texts)} samples")
        logger.info(f"Score range: {min(scores)} - {max(scores)}")
        
        return texts, scores
    
    def tokenize_texts(self, texts):
        """Tokenize texts using the pretrained tokenizer"""
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
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
        num_workers=config['device']['num_workers']
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['device']['num_workers']
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['device']['num_workers']
    )
    
    return train_loader, val_loader, test_loader, train_data, val_data, test_data