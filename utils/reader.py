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
        """Read CSV file and return data with robust score handling"""
        logger.info(f"Reading data from: {csv_path}")
        df = pd.read_csv(csv_path)
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
        valid_texts = []
        valid_scores = []
        
        for i, (text, score) in enumerate(zip(texts, scores)):
            try:
                score_float = float(score)
                
                # Check for invalid scores
                if np.isnan(score_float) or np.isinf(score_float):
                    logger.warning(f"Invalid score at index {i}: {score}, skipping...")
                    continue
                
                # Ensure score is within expected range
                if score_float < self.min_score or score_float > self.max_score:
                    logger.warning(f"Score {score_float} at index {i} outside range [{self.min_score}, {self.max_score}], clipping...")
                    score_float = np.clip(score_float, self.min_score, self.max_score)
                
                # Check for valid text
                if not text or pd.isna(text) or len(str(text).strip()) == 0:
                    logger.warning(f"Invalid text at index {i}, skipping...")
                    continue
                
                valid_texts.append(str(text).strip())
                valid_scores.append(score_float)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing score at index {i}: {score}, error: {e}, skipping...")
                continue
        
        logger.info(f"Loaded {len(valid_texts)} valid samples (filtered from {len(texts)} total)")
        logger.info(f"Score range: {min(valid_scores):.2f} - {max(valid_scores):.2f}")
        logger.info(f"Score statistics: mean={np.mean(valid_scores):.2f}, std={np.std(valid_scores):.2f}")
        
        return valid_texts, valid_scores
    
    def tokenize_texts(self, texts):
        """Tokenize texts using the pretrained tokenizer with error handling"""
        try:
            encoded = self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
                add_special_tokens=True
            )
            return encoded['input_ids'], encoded['attention_mask']
        except Exception as e:
            logger.error(f"Error in tokenization: {e}")
            # Fallback: tokenize one by one
            input_ids_list = []
            attention_mask_list = []
            
            for text in texts:
                try:
                    encoded = self.tokenizer(
                        text,
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt',
                        add_special_tokens=True
                    )
                    input_ids_list.append(encoded['input_ids'].squeeze(0))
                    attention_mask_list.append(encoded['attention_mask'].squeeze(0))
                except:
                    # Create empty tensor as fallback
                    empty_ids = torch.zeros(self.max_length, dtype=torch.long)
                    empty_mask = torch.zeros(self.max_length, dtype=torch.long)
                    empty_ids[0] = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id else 101
                    empty_ids[-1] = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id else 102
                    empty_mask[0] = 1
                    empty_mask[-1] = 1
                    input_ids_list.append(empty_ids)
                    attention_mask_list.append(empty_mask)
            
            return torch.stack(input_ids_list), torch.stack(attention_mask_list)
    
    def prepare_data(self, csv_path):
        """Prepare data for training/evaluation with robust error handling"""
        texts, scores = self.read_csv(csv_path)
        
        if len(texts) == 0:
            raise ValueError(f"No valid data found in {csv_path}")
        
        # Tokenize texts
        input_ids, attention_masks = self.tokenize_texts(texts)
        
        # Validate score range before normalization
        scores_array = np.array(scores)
        if np.any(scores_array < self.min_score) or np.any(scores_array > self.max_score):
            logger.warning("Some scores are outside the expected range, clipping...")
            scores_array = np.clip(scores_array, self.min_score, self.max_score)
        
        # Normalize scores to [0, 1] with safety checks
        try:
            normalized_scores = normalize_scores(
                scores_array, 
                self.min_score, 
                self.max_score
            )
            
            # Additional safety check
            if np.any(np.isnan(normalized_scores)) or np.any(np.isinf(normalized_scores)):
                logger.error("NaN or Inf values found in normalized scores!")
                # Replace NaN/Inf with mean value
                valid_mask = ~(np.isnan(normalized_scores) | np.isinf(normalized_scores))
                if np.any(valid_mask):
                    mean_score = np.mean(normalized_scores[valid_mask])
                    normalized_scores[~valid_mask] = mean_score
                else:
                    normalized_scores = np.full_like(normalized_scores, 0.5)  # Use middle value
            
            # Ensure all scores are in [0, 1]
            normalized_scores = np.clip(normalized_scores, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error in score normalization: {e}")
            # Fallback normalization
            normalized_scores = (scores_array - self.min_score) / (self.max_score - self.min_score)
            normalized_scores = np.clip(normalized_scores, 0.0, 1.0)
        
        # Convert to tensors
        scores_tensor = torch.FloatTensor(normalized_scores)
        original_scores_tensor = torch.FloatTensor(scores_array)
        
        # Final validation
        if torch.any(torch.isnan(scores_tensor)) or torch.any(torch.isinf(scores_tensor)):
            logger.error("NaN or Inf found in final score tensor!")
            scores_tensor = torch.clamp(scores_tensor, 0.0, 1.0)
            scores_tensor[torch.isnan(scores_tensor)] = 0.5
            scores_tensor[torch.isinf(scores_tensor)] = 0.5
        
        logger.info(f"Normalized score range: {scores_tensor.min().item():.4f} - {scores_tensor.max().item():.4f}")
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'scores': scores_tensor,
            'original_scores': original_scores_tensor,
            'texts': texts
        }


def get_data_loaders(config):
    """Get data loaders for training, validation, and testing with error handling"""
    reader = SpeakingDataReader(config)
    
    try:
        # Prepare datasets
        train_data = reader.prepare_data(config['data']['train_csv_path'])
        val_data = reader.prepare_data(config['data']['val_csv_path'])
        test_data = reader.prepare_data(config['data']['test_csv_path'])
        
        # Validate data consistency
        for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            logger.info(f"{name.upper()} data validation:")
            logger.info(f"  Input IDs shape: {data['input_ids'].shape}")
            logger.info(f"  Scores shape: {data['scores'].shape}")
            logger.info(f"  Score range: {data['scores'].min().item():.4f} - {data['scores'].max().item():.4f}")
            
            # Check for any remaining issues
            if torch.any(torch.isnan(data['scores'])):
                logger.error(f"NaN values found in {name} scores!")
            if torch.any(torch.isinf(data['scores'])):
                logger.error(f"Inf values found in {name} scores!")
        
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
        
        # Create data loaders with error handling
        def safe_dataloader(dataset, batch_size, shuffle, num_workers):
            try:
                return torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    drop_last=False,
                    pin_memory=False  # Disable pin_memory to avoid potential issues
                )
            except Exception as e:
                logger.warning(f"Error creating DataLoader with {num_workers} workers: {e}")
                logger.warning("Falling back to single-threaded DataLoader")
                return torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=0,  # Single-threaded
                    drop_last=False
                )
        
        train_loader = safe_dataloader(
            train_dataset, config['training']['batch_size'], True, config['device']['num_workers']
        )
        
        val_loader = safe_dataloader(
            val_dataset, config['training']['batch_size'], False, config['device']['num_workers']
        )
        
        test_loader = safe_dataloader(
            test_dataset, config['training']['batch_size'], False, config['device']['num_workers']
        )
        
        return train_loader, val_loader, test_loader, train_data, val_data, test_data
        
    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        raise