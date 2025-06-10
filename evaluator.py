# -*- coding: utf-8 -*-
import torch
import numpy as np
from utils.metrics import calculate_metrics, mean_absolute_error, kappa
from utils.utils import get_logger, rescale_to_score_range, round_to_increment, save_checkpoint
import os
from tqdm import tqdm

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
        """Evaluate model in single essay mode (without pairwise) with robust error handling"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_original_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"{phase.upper()} Evaluation (Single)")):
                try:
                    input_ids, attention_mask, scores, original_scores = batch
                    
                    # Move to device
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    
                    # Validate inputs
                    if torch.any(torch.isnan(input_ids)) or torch.any(torch.isnan(attention_mask)):
                        logger.warning(f"NaN detected in inputs for batch {batch_idx}, skipping...")
                        continue
                    
                    # Predict
                    predictions = self.model(input_ids, attention_mask)
                    
                    # Validate predictions
                    if torch.any(torch.isnan(predictions)) or torch.any(torch.isinf(predictions)):
                        logger.warning(f"NaN/Inf predictions in batch {batch_idx}, replacing with 0.5...")
                        predictions = torch.where(
                            torch.isnan(predictions) | torch.isinf(predictions),
                            torch.tensor(0.5, device=predictions.device),
                            predictions
                        )
                    
                    # Ensure predictions are in valid range
                    predictions = torch.clamp(predictions, 0.0, 1.0)
                    
                    # Collect predictions
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(scores.numpy())
                    all_original_targets.extend(original_scores.numpy())
                    
                except Exception as e:
                    logger.warning(f"Error in batch {batch_idx}: {e}, skipping...")
                    continue
        
        if len(all_predictions) == 0:
            logger.error("No valid predictions collected!")
            return {
                'mae': float('inf'),
                'rmse': float('inf'),
                'qwk': 0.0,
                'pearson': 0.0,
                'spearman': 0.0
            }, []
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_original_targets = np.array(all_original_targets)
        
        # Validate arrays
        if np.any(np.isnan(all_predictions)) or np.any(np.isinf(all_predictions)):
            logger.warning("NaN/Inf found in predictions, cleaning...")
            valid_mask = ~(np.isnan(all_predictions) | np.isinf(all_predictions))
            if np.any(valid_mask):
                mean_pred = np.mean(all_predictions[valid_mask])
                all_predictions[~valid_mask] = mean_pred
            else:
                all_predictions = np.full_like(all_predictions, 0.5)
        
        # Ensure predictions are in [0, 1]
        all_predictions = np.clip(all_predictions, 0.0, 1.0)
        
        # Rescale predictions to original score range
        try:
            predictions_rescaled = rescale_to_score_range(
                all_predictions, self.min_score, self.max_score
            )
            
            # Ensure rescaled predictions are in valid range
            predictions_rescaled = np.clip(predictions_rescaled, self.min_score, self.max_score)
            
            # Round to increment
            predictions_rounded = round_to_increment(predictions_rescaled, self.increment)
            
            # Ensure rounded predictions are still in valid range
            predictions_rounded = np.clip(predictions_rounded, self.min_score, self.max_score)
            
        except Exception as e:
            logger.error(f"Error in score rescaling: {e}")
            # Fallback rescaling
            predictions_rescaled = all_predictions * (self.max_score - self.min_score) + self.min_score
            predictions_rounded = np.round(predictions_rescaled / self.increment) * self.increment
            predictions_rounded = np.clip(predictions_rounded, self.min_score, self.max_score)
        
        # Calculate metrics with error handling
        try:
            metrics = calculate_metrics(all_original_targets, predictions_rounded)
            
            # Validate metrics
            for key, value in metrics.items():
                if np.isnan(value) or np.isinf(value):
                    logger.warning(f"Invalid {key} value: {value}, setting to default")
                    if key == 'mae' or key == 'rmse':
                        metrics[key] = float('inf')
                    else:
                        metrics[key] = 0.0
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            metrics = {
                'mae': float('inf'),
                'rmse': float('inf'),
                'qwk': 0.0,
                'pearson': 0.0,
                'spearman': 0.0
            }
        
        # Log results
        logger.info(f"{phase.upper()} Results:")
        logger.info(f"  MAE (L1): {metrics['mae']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  QWK: {metrics['qwk']:.4f}")
        logger.info(f"  Pearson: {metrics['pearson']:.4f}")
        logger.info(f"  Spearman: {metrics['spearman']:.4f}")
        logger.info(f"  Prediction range: {predictions_rounded.min():.1f} - {predictions_rounded.max():.1f}")
        logger.info(f"  Target range: {all_original_targets.min():.1f} - {all_original_targets.max():.1f}")
        
        return metrics, predictions_rounded
    
    def evaluate_with_voting(self, data_loader, phase='test'):
        """Evaluate model with multi-sample voting with robust error handling"""
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
        
        # Validate reference data
        if torch.any(torch.isnan(ref_scores)) or torch.any(torch.isinf(ref_scores)):
            logger.warning("Invalid reference scores found, cleaning...")
            ref_scores = torch.clamp(ref_scores, 0.0, 1.0)
            ref_scores[torch.isnan(ref_scores)] = 0.5
            ref_scores[torch.isinf(ref_scores)] = 0.5
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"{phase.upper()} Evaluation (Voting)")):
                try:
                    input_ids, attention_mask, scores, original_scores = batch
                    
                    # Move to device
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    
                    batch_size = input_ids.size(0)
                    batch_predictions = []
                    
                    # For each essay in batch
                    for i in range(batch_size):
                        try:
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
                            
                            # Validate relative scores
                            if torch.any(torch.isnan(relative_scores)) or torch.any(torch.isinf(relative_scores)):
                                logger.warning(f"Invalid relative scores for sample {i}, using fallback...")
                                # Fallback to single mode
                                single_score = self.model(curr_input_ids, curr_attention_mask)
                                single_score = torch.clamp(single_score, 0.0, 1.0)
                                batch_predictions.append(single_score.cpu().numpy())
                                continue
                            
                            # Ensure relative scores are in valid range
                            relative_scores = torch.clamp(relative_scores, 0.0, 1.0)
                            
                            # Convert relative scores from [0,1] back to [-1,1]
                            relative_scores = (relative_scores * 2) - 1
                            
                            # Add reference scores to get absolute scores
                            absolute_scores = relative_scores + ref_scores
                            
                            # Ensure absolute scores are in valid range
                            absolute_scores = torch.clamp(absolute_scores, 0.0, 1.0)
                            
                            # Average across all references
                            avg_score = absolute_scores.mean()
                            
                            # Final validation
                            if torch.isnan(avg_score) or torch.isinf(avg_score):
                                avg_score = torch.tensor(0.5)
                            else:
                                avg_score = torch.clamp(avg_score, 0.0, 1.0)
                            
                            batch_predictions.append(avg_score.cpu().numpy())
                            
                        except Exception as e:
                            logger.warning(f"Error processing sample {i} in batch {batch_idx}: {e}")
                            # Use fallback prediction
                            batch_predictions.append(0.5)
                    
                    # Collect predictions
                    all_predictions.extend(batch_predictions)
                    all_targets.extend(scores.numpy())
                    all_original_targets.extend(original_scores.numpy())
                    
                except Exception as e:
                    logger.warning(f"Error in voting batch {batch_idx}: {e}, skipping...")
                    continue
        
        if len(all_predictions) == 0:
            logger.error("No valid predictions collected in voting mode!")
            return self.evaluate_single_mode(data_loader, phase)
        
        # Convert to numpy arrays and process same as single mode
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_original_targets = np.array(all_original_targets)
        
        # Clean and validate
        if np.any(np.isnan(all_predictions)) or np.any(np.isinf(all_predictions)):
            logger.warning("Cleaning invalid predictions in voting mode...")
            valid_mask = ~(np.isnan(all_predictions) | np.isinf(all_predictions))
            if np.any(valid_mask):
                mean_pred = np.mean(all_predictions[valid_mask])
                all_predictions[~valid_mask] = mean_pred
            else:
                all_predictions = np.full_like(all_predictions, 0.5)
        
        all_predictions = np.clip(all_predictions, 0.0, 1.0)
        
        # Rescale and calculate metrics (same as single mode)
        try:
            predictions_rescaled = rescale_to_score_range(
                all_predictions, self.min_score, self.max_score
            )
            predictions_rescaled = np.clip(predictions_rescaled, self.min_score, self.max_score)
            predictions_rounded = round_to_increment(predictions_rescaled, self.increment)
            predictions_rounded = np.clip(predictions_rounded, self.min_score, self.max_score)
            
            metrics = calculate_metrics(all_original_targets, predictions_rounded)
            
            # Validate metrics
            for key, value in metrics.items():
                if np.isnan(value) or np.isinf(value):
                    logger.warning(f"Invalid {key} value in voting: {value}, setting to default")
                    if key == 'mae' or key == 'rmse':
                        metrics[key] = float('inf')
                    else:
                        metrics[key] = 0.0
                        
        except Exception as e:
            logger.error(f"Error in voting evaluation: {e}")
            metrics = {
                'mae': float('inf'),
                'rmse': float('inf'),
                'qwk': 0.0,
                'pearson': 0.0,
                'spearman': 0.0
            }
            predictions_rounded = all_predictions
        
        # Log results
        logger.info(f"{phase.upper()} Results (with voting):")
        logger.info(f"  MAE (L1): {metrics['mae']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  QWK: {metrics['qwk']:.4f}")
        logger.info(f"  Pearson: {metrics['pearson']:.4f}")
        logger.info(f"  Spearman: {metrics['spearman']:.4f}")
        
        return metrics, predictions_rounded
    
    def save_best_model(self, model, optimizer, epoch, metrics):
        """Save best model based on validation metrics with error handling"""
        try:
            checkpoint_dir = self.config['training']['checkpoint_dir']
            
            # Validate metrics before saving
            mae = metrics.get('mae', float('inf'))
            qwk = metrics.get('qwk', 0.0)
            
            if np.isnan(mae) or np.isinf(mae):
                mae = float('inf')
            if np.isnan(qwk) or np.isinf(qwk):
                qwk = 0.0
            
            # Check if best L1 (MAE)
            if mae < self.best_val_l1:
                self.best_val_l1 = mae
                if self.config['training']['save_best_l1']:
                    filepath = os.path.join(checkpoint_dir, 'best_model_l1.pt')
                    save_checkpoint(model, optimizer, epoch, metrics, filepath)
                    logger.info(f"Saved best L1 model with MAE: {mae:.4f}")
            
            # Check if best QWK
            if qwk > self.best_val_qwk:
                self.best_val_qwk = qwk
                self.best_val_metrics = metrics
                if self.config['training']['save_best_qwk']:
                    filepath = os.path.join(checkpoint_dir, 'best_model_qwk.pt')
                    save_checkpoint(model, optimizer, epoch, metrics, filepath)
                    logger.info(f"Saved best QWK model with QWK: {qwk:.4f}")
                    
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def print_best_results(self):
        """Print best validation results"""
        logger.info("\nBest Validation Results:")
        logger.info(f"  Best L1 (MAE): {self.best_val_l1:.4f}")
        logger.info(f"  Best QWK: {self.best_val_qwk:.4f}")
        if self.best_val_metrics:
            logger.info(f"  Best QWK model metrics:")
            for metric, value in self.best_val_metrics.items():
                if not (np.isnan(value) or np.isinf(value)):
                    logger.info(f"    {metric}: {value:.4f}")