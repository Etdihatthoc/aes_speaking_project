# -*- coding: utf-8 -*-
import os
import sys
import torch
import pandas as pd
import argparse
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.utils import load_config, rescale_to_score_range, round_to_increment, get_logger
from networks.core_networks import NPCRModel
from utils.data_prepare import prepare_reference_samples

logger = get_logger("Inference")


class SpeakingScoringInference:
    def __init__(self, config_path, checkpoint_path):
        self.config = load_config(config_path)
        self.device = torch.device(self.config['device']['cuda_device'] if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = NPCRModel(self.config)
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
    
    def _load_reference_data(self):
        """Load reference data from training set"""
        # Load training data for references
        train_df = pd.read_csv(self.config['data']['train_csv_path'])
        
        # Select diverse reference samples
        example_size = self.config['training']['example_size']
        
        # Get scores
        score_column = self.config['model']['type']
        if score_column not in train_df.columns:
            score_column = 'vocabulary'
        
        # Sort by score and select evenly distributed samples
        train_df_sorted = train_df.sort_values(by=score_column)
        step = max(1, len(train_df_sorted) // example_size)
        reference_df = train_df_sorted.iloc[::step][:example_size]
        
        # Tokenize reference texts
        ref_texts = reference_df['text'].tolist()
        ref_scores = reference_df[score_column].tolist()
        
        encoded = self.tokenizer(
            ref_texts,
            padding='max_length',
            truncation=True,
            max_length=self.config['model']['max_length'],
            return_tensors='pt'
        )
        
        # Normalize scores
        from utils.utils import normalize_scores
        ref_scores_normalized = normalize_scores(
            torch.FloatTensor(ref_scores),
            self.config['score']['min_score'],
            self.config['score']['max_score']
        )
        
        self.ref_input_ids = encoded['input_ids'].to(self.device)
        self.ref_attention_mask = encoded['attention_mask'].to(self.device)
        self.ref_scores = ref_scores_normalized.to(self.device)
        
        logger.info(f"Loaded {len(ref_texts)} reference samples")
    
    def score_single_text(self, text, use_voting=True):
        """Score a single text"""
        # Tokenize input
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config['model']['max_length'],
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            if use_voting:
                # Multi-sample voting
                num_refs = self.ref_input_ids.size(0)
                input_ids_exp = input_ids.expand(num_refs, -1)
                attention_mask_exp = attention_mask.expand(num_refs, -1)
                
                # Predict relative scores
                relative_scores = self.model.forward_pair(
                    input_ids_exp,
                    attention_mask_exp,
                    self.ref_input_ids,
                    self.ref_attention_mask
                )
                
                # Convert relative scores from [0,1] back to [-1,1]
                relative_scores = (relative_scores * 2) - 1
                
                # Add reference scores
                absolute_scores = relative_scores + self.ref_scores
                
                # Average
                score = absolute_scores.mean()
            else:
                # Single mode prediction
                score = self.model(input_ids, attention_mask)
                score = score.squeeze()
        
        # Rescale to original score range
        score_rescaled = rescale_to_score_range(
            score.cpu().numpy(),
            self.config['score']['min_score'],
            self.config['score']['max_score']
        )
        
        # Round to increment
        score_rounded = round_to_increment(score_rescaled, self.config['score']['increment'])
        
        return float(score_rounded)
    
    def score_csv(self, csv_path, output_path, use_voting=True):
        """Score all texts in a CSV file"""
        logger.info(f"Scoring texts from {csv_path}")
        
        # Load data
        df = pd.read_csv(csv_path)
        
        if 'text' not in df.columns:
            raise ValueError("CSV must contain 'text' column")
        
        # Score each text
        scores = []
        for idx, row in df.iterrows():
            text = row['text']
            score = self.score_single_text(text, use_voting=use_voting)
            scores.append(score)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} texts")
        
        # Add predictions to dataframe
        score_column = f"predicted_{self.config['model']['type']}"
        df[score_column] = scores
        
        # Save results
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        # Print statistics
        logger.info(f"Score statistics:")
        logger.info(f"  Mean: {df[score_column].mean():.2f}")
        logger.info(f"  Std: {df[score_column].std():.2f}")
        logger.info(f"  Min: {df[score_column].min():.1f}")
        logger.info(f"  Max: {df[score_column].max():.1f}")


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
        print(f"\nScore: {score}")


if __name__ == '__main__':
    main()