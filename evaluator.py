# -*- coding: utf-8 -*-
import torch
import numpy as np
from utils.metrics import calculate_metrics, mean_absolute_error, kappa
from utils.utils import get_logger, rescale_to_score_range, round_to_increment, save_checkpoint
import os

logger = get_logger("Evaluator")


class Evaluator:
    def __init__(self, config, model, reference_data=None):
        self.config = config
        self.model = model
        self.reference_data = reference_data
        self.device = torch.device(config['device']['cuda_device'] if torch.cuda.is_available() else 'cpu')
        
        # Best metrics tracking
        self.best_val_l1 = float('inf')
        self.best_val_qwk = -1.0
        self.best_val_metrics = {}
        
        # Score settings
        self.min_score = config['score']['min_score']
        self.max_score = config['score']['max_score']
        self.increment = config['score']['increment']
        
    def evaluate_single_mode(self, data_loader, phase='val'):
        """Evaluate model in single essay mode (without pairwise)"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_original_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids, attention_mask, scores, original_scores = batch
                
                # Move to device
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                # Predict
                predictions = self.model(input_ids, attention_mask)
                
                # Collect predictions
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(scores.numpy())
                all_original_targets.extend(original_scores.numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_original_targets = np.array(all_original_targets)
        
        # Rescale predictions to original score range
        predictions_rescaled = rescale_to_score_range(
            all_predictions, self.min_score, self.max_score
        )
        
        # Round to increment
        predictions_rounded = round_to_increment(predictions_rescaled, self.increment)
        
        # Calculate metrics
        metrics = calculate_metrics(all_original_targets, predictions_rounded)
        
        # Log results
        logger.info(f"{phase.upper()} Results:")
        logger.info(f"  MAE (L1): {metrics['mae']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  QWK: {metrics['qwk']:.4f}")
        logger.info(f"  Pearson: {metrics['pearson']:.4f}")
        logger.info(f"  Spearman: {metrics['spearman']:.4f}")
        
        return metrics, predictions_rounded
    
    def evaluate_with_voting(self, data_loader, phase='test'):
        """Evaluate model with multi-sample voting"""
        if self.reference_data is None:
            logger.warning("No reference data available, using single mode evaluation")
            return self.evaluate_single_mode(data_loader, phase)
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_original_targets = []
        
        # Get reference data
        ref_input_ids = self.reference_data['input_ids'].to(self.device)
        ref_attention_mask = self.reference_data['attention_mask'].to(self.device)
        ref_scores = self.reference_data['scores'].to(self.device)
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids, attention_mask, scores, original_scores = batch
                
                # Move to device
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                batch_size = input_ids.size(0)
                batch_predictions = []
                
                # For each essay in batch
                for i in range(batch_size):
                    # Get current essay
                    curr_input_ids = input_ids[i:i+1]
                    curr_attention_mask = attention_mask[i:i+1]
                    
                    # Expand to match number of references
                    num_refs = ref_input_ids.size(0)
                    curr_input_ids_exp = curr_input_ids.expand(num_refs, -1)
                    curr_attention_mask_exp = curr_attention_mask.expand(num_refs, -1)
                    
                    # Predict relative scores with all references
                    relative_scores = self.model.forward_pair(
                        curr_input_ids_exp,
                        curr_attention_mask_exp,
                        ref_input_ids,
                        ref_attention_mask
                    )
                    
                    # Convert relative scores from [0,1] back to [-1,1]
                    relative_scores = (relative_scores * 2) - 1
                    
                    # Add reference scores to get absolute scores
                    absolute_scores = relative_scores + ref_scores
                    
                    # Average across all references
                    avg_score = absolute_scores.mean()
                    batch_predictions.append(avg_score.cpu().numpy())
                
                # Collect predictions
                all_predictions.extend(batch_predictions)
                all_targets.extend(scores.numpy())
                all_original_targets.extend(original_scores.numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_original_targets = np.array(all_original_targets)
        
        # Rescale predictions to original score range
        predictions_rescaled = rescale_to_score_range(
            all_predictions, self.min_score, self.max_score
        )
        
        # Round to increment
        predictions_rounded = round_to_increment(predictions_rescaled, self.increment)
        
        # Calculate metrics
        metrics = calculate_metrics(all_original_targets, predictions_rounded)
        
        # Log results
        logger.info(f"{phase.upper()} Results (with voting):")
        logger.info(f"  MAE (L1): {metrics['mae']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  QWK: {metrics['qwk']:.4f}")
        logger.info(f"  Pearson: {metrics['pearson']:.4f}")
        logger.info(f"  Spearman: {metrics['spearman']:.4f}")
        
        return metrics, predictions_rounded
    
    def save_best_model(self, model, optimizer, epoch, metrics):
        """Save best model based on validation metrics"""
        checkpoint_dir = self.config['training']['checkpoint_dir']
        
        # Check if best L1 (MAE)
        if metrics['mae'] < self.best_val_l1:
            self.best_val_l1 = metrics['mae']
            if self.config['training']['save_best_l1']:
                filepath = os.path.join(checkpoint_dir, 'best_model_l1.pt')
                save_checkpoint(model, optimizer, epoch, metrics, filepath)
                logger.info(f"Saved best L1 model with MAE: {metrics['mae']:.4f}")
        
        # Check if best QWK
        if metrics['qwk'] > self.best_val_qwk:
            self.best_val_qwk = metrics['qwk']
            self.best_val_metrics = metrics
            if self.config['training']['save_best_qwk']:
                filepath = os.path.join(checkpoint_dir, 'best_model_qwk.pt')
                save_checkpoint(model, optimizer, epoch, metrics, filepath)
                logger.info(f"Saved best QWK model with QWK: {metrics['qwk']:.4f}")
    
    def print_best_results(self):
        """Print best validation results"""
        logger.info("\nBest Validation Results:")
        logger.info(f"  Best L1 (MAE): {self.best_val_l1:.4f}")
        logger.info(f"  Best QWK: {self.best_val_qwk:.4f}")
        if self.best_val_metrics:
            logger.info(f"  Best QWK model metrics:")
            for metric, value in self.best_val_metrics.items():
                logger.info(f"    {metric}: {value:.4f}")