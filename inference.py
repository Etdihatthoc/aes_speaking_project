# -*- coding: utf-8 -*-
import os
import sys
import torch
import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.utils import load_config, rescale_to_score_range, round_to_increment, get_logger, normalize_scores
from networks.core_networks import NPCRModel
from utils.data_prepare import prepare_reference_samples

logger = get_logger("Inference")


class SpeakingScoringInference:
    def __init__(self, config_path, checkpoint_path):
        self.config = load_config(config_path)
        self.device = torch.device(self.config['device']['cuda_device'] if torch.cuda.is_available() else 'cpu')
        
        # Initialize model based on type (same as main.py)
        logger.info("Initializing model...")
        model_type = self.config['model'].get('model_type', 'bert')
        
        if model_type == 'stella' or 'stella' in self.config['model']['pretrained_model'].lower():
            logger.info("Using Stella model architecture")
            from networks.stella_npcr import StellaNPCRModel
            self.model = StellaNPCRModel(self.config)
        else:
            logger.info("Using standard BERT architecture")
            self.model = NPCRModel(self.config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['pretrained_model'])
        
        # Load reference data for multi-sample voting
        self._load_reference_data()
        
        logger.info(f"Model loaded from {checkpoint_path}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model type: {model_type}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_reference_data(self):
        """Load reference data from training set"""
        # Load training data for references
        train_df = pd.read_csv(self.config['data']['train_csv_path'])
        
        # Select diverse reference samples
        example_size = self.config['training']['example_size']
        
        # Get scores based on config type
        score_column = self.config['model']['type']
        if score_column not in train_df.columns:
            logger.warning(f"Column '{score_column}' not found, using 'vocabulary' instead")
            score_column = 'vocabulary'
        
        # Sort by score and select evenly distributed samples
        train_df_sorted = train_df.sort_values(by=score_column)
        step = max(1, len(train_df_sorted) // example_size)
        reference_df = train_df_sorted.iloc[::step][:example_size]
        
        # Get reference texts and scores
        ref_texts = reference_df['text'].tolist()
        ref_scores = reference_df[score_column].tolist()
        
        # Tokenize reference texts (no preprocessing)
        encoded = self.tokenizer(
            ref_texts,
            padding='max_length',
            truncation=True,
            max_length=self.config['model']['max_length'],
            return_tensors='pt',
            add_special_tokens=True
        )
        
        # Normalize scores to [0, 1]
        ref_scores_normalized = normalize_scores(
            np.array(ref_scores),
            self.config['score']['min_score'],
            self.config['score']['max_score']
        )
        
        self.ref_input_ids = encoded['input_ids'].to(self.device)
        self.ref_attention_mask = encoded['attention_mask'].to(self.device)
        self.ref_scores = torch.FloatTensor(ref_scores_normalized).to(self.device)
        
        logger.info(f"Loaded {len(ref_texts)} reference samples")
        logger.info(f"Reference score range: {min(ref_scores):.1f} - {max(ref_scores):.1f}")
    
    def score_single_text(self, text, use_voting=True):
        """Score a single text"""
        # Tokenize input (no preprocessing)
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config['model']['max_length'],
            return_tensors='pt',
            add_special_tokens=True
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            if use_voting:
                # Multi-sample voting
                num_refs = self.ref_input_ids.size(0)
                predictions = []
                
                # Process in batches to avoid memory issues
                batch_size = 10
                for i in range(0, num_refs, batch_size):
                    end_idx = min(i + batch_size, num_refs)
                    batch_size_actual = end_idx - i
                    
                    # Expand input for batch
                    input_ids_exp = input_ids.expand(batch_size_actual, -1)
                    attention_mask_exp = attention_mask.expand(batch_size_actual, -1)
                    
                    # Get batch of references
                    ref_input_ids_batch = self.ref_input_ids[i:end_idx]
                    ref_attention_mask_batch = self.ref_attention_mask[i:end_idx]
                    ref_scores_batch = self.ref_scores[i:end_idx]
                    
                    # Predict relative scores
                    relative_scores = self.model.forward_pair(
                        input_ids_exp,
                        attention_mask_exp,
                        ref_input_ids_batch,
                        ref_attention_mask_batch
                    )
                    
                    # Convert relative scores from [0,1] back to [-1,1]
                    relative_scores = (relative_scores * 2) - 1
                    
                    # Add reference scores to get absolute scores
                    absolute_scores = relative_scores + ref_scores_batch
                    predictions.extend(absolute_scores.cpu().numpy())
                
                # Average all predictions
                score = np.mean(predictions)
            else:
                # Single mode prediction
                score = self.model(input_ids, attention_mask)
                score = score.squeeze().cpu().numpy()
        
        # Đảm bảo score trong khoảng [0, 1] trước khi rescale
        score = np.clip(score, 0.0, 1.0)
        
        # Rescale to original score range
        score_rescaled = rescale_to_score_range(
            score,
            self.config['score']['min_score'],
            self.config['score']['max_score']
        )
        
        # Ensure score is within valid range
        score_rescaled = np.clip(
            score_rescaled,
            self.config['score']['min_score'],
            self.config['score']['max_score']
        )
        
        # Round to increment
        score_rounded = round_to_increment(score_rescaled, self.config['score']['increment'])
        
        return float(score_rounded)
    
    def score_batch(self, texts, use_voting=True):
        """Score a batch of texts efficiently"""
        scores = []
        
        if use_voting:
            # Process each text individually with voting
            for text in tqdm(texts, desc="Scoring with voting"):
                score = self.score_single_text(text, use_voting=True)
                scores.append(score)
        else:
            # Batch processing for single mode
            batch_size = 64
            for i in tqdm(range(0, len(texts), batch_size), desc="Batch scoring"):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize batch
                encoded = self.tokenizer(
                    batch_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=self.config['model']['max_length'],
                    return_tensors='pt',
                    add_special_tokens=True
                )
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    batch_scores = self.model(input_ids, attention_mask)
                    batch_scores = batch_scores.cpu().numpy()
                
                # Rescale and round each score
                for score in batch_scores:
                    score_rescaled = rescale_to_score_range(
                        score,
                        self.config['score']['min_score'],
                        self.config['score']['max_score']
                    )
                    score_rounded = round_to_increment(score_rescaled, self.config['score']['increment'])
                    scores.append(float(score_rounded))
        
        return scores
    
    def score_csv(self, csv_path, output_path, use_voting=True):
        """Score all texts in a CSV file"""
        logger.info(f"Scoring texts from {csv_path}")
        
        # Load data
        df = pd.read_csv(csv_path)
        
        if 'text' not in df.columns:
            raise ValueError("CSV must contain 'text' column")
        
        # Get the original score column
        score_column = self.config['model']['type']
        if score_column not in df.columns:
            logger.warning(f"Column '{score_column}' not found in input CSV")
            score_column = None
        
        # Get all texts
        texts = df['text'].tolist()
        logger.info(f"Found {len(texts)} texts to score")
        
        # Score all texts
        scores = self.score_batch(texts, use_voting=use_voting)
        
        # Create output dataframe with only required columns
        output_df = pd.DataFrame()
        
        # Add original score column if it exists
        if score_column:
            output_df[score_column] = df[score_column]
        
        # Add predicted scores
        output_df[f'predicted_{self.config["model"]["type"]}'] = scores
        
        # Save results
        output_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        # Print statistics
        pred_column = f'predicted_{self.config["model"]["type"]}'
        logger.info(f"\nScore statistics for {pred_column}:")
        logger.info(f"  Mean: {output_df[pred_column].mean():.2f}")
        logger.info(f"  Std: {output_df[pred_column].std():.2f}")
        logger.info(f"  Min: {output_df[pred_column].min():.1f}")
        logger.info(f"  Max: {output_df[pred_column].max():.1f}")
        
        # Check for any invalid scores
        invalid_scores = output_df[
            (output_df[pred_column] < self.config['score']['min_score']) | 
            (output_df[pred_column] > self.config['score']['max_score'])
        ]
        if len(invalid_scores) > 0:
            logger.warning(f"Found {len(invalid_scores)} scores outside valid range!")
        
        # Score distribution
        logger.info("\nScore distribution:")
        score_counts = output_df[pred_column].value_counts().sort_index()
        for score, count in score_counts.items():
            logger.info(f"  {score}: {count} ({count/len(output_df)*100:.1f}%)")



def main():
    parser = argparse.ArgumentParser(description="Inference for Speaking Scoring")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input text or CSV file path')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path (for CSV input)')
    parser.add_argument('--no-voting', action='store_true', help='Disable multi-sample voting')
    
    args = parser.parse_args()
    
    # Initialize inference
    scorer = SpeakingScoringInference(args.config, args.checkpoint)
    
    # Check if input is text or file
    if args.input.endswith('.csv'):
        # Score CSV file
        if args.output is None:
            args.output = args.input.replace('.csv', '_scored.csv')
        scorer.score_csv(args.input, args.output, use_voting=not args.no_voting)
    else:
        # Score single text
        score = scorer.score_single_text(args.input, use_voting=not args.no_voting)
        print(f"\nInput text: {args.input[:100]}...")
        print(f"Predicted {scorer.config['model']['type']} score: {score}")
        print(f"Mode: {'Multi-sample voting' if not args.no_voting else 'Single prediction'}")


if __name__ == '__main__':
    main()