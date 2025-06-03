# -*- coding: utf-8 -*-
import wandb
import torch
import numpy as np
from collections import defaultdict


class TrainingMonitor:
    """Enhanced training monitor with WandB integration"""
    
    def __init__(self, config):
        self.config = config
        self.metrics_history = defaultdict(list)
        self.best_metrics = {
            'val_mae': float('inf'),
            'val_qwk': -1.0,
            'val_mae_voting': float('inf'),
            'val_qwk_voting': -1.0
        }
        
    def log_step_metrics(self, metrics, step):
        """Log metrics at each training step"""
        wandb.log(metrics, step=step)
        
    def log_epoch_metrics(self, epoch, train_metrics, val_metrics_single, val_metrics_voting):
        """Log comprehensive epoch metrics"""
        epoch_metrics = {
            'epoch': epoch,
        }
        
        # Training metrics
        if train_metrics:
            for key, value in train_metrics.items():
                epoch_metrics[f'train_{key}'] = value
        
        # Validation metrics - single mode
        for key, value in val_metrics_single.items():
            epoch_metrics[f'val_{key}_single'] = value
            self.metrics_history[f'val_{key}_single'].append(value)
        
        # Validation metrics - voting mode
        for key, value in val_metrics_voting.items():
            epoch_metrics[f'val_{key}_voting'] = value
            self.metrics_history[f'val_{key}_voting'].append(value)
        
        # Check for best metrics
        if val_metrics_single['mae'] < self.best_metrics['val_mae']:
            self.best_metrics['val_mae'] = val_metrics_single['mae']
            epoch_metrics['best_val_mae_single'] = val_metrics_single['mae']
        
        if val_metrics_single['qwk'] > self.best_metrics['val_qwk']:
            self.best_metrics['val_qwk'] = val_metrics_single['qwk']
            epoch_metrics['best_val_qwk_single'] = val_metrics_single['qwk']
        
        if val_metrics_voting['mae'] < self.best_metrics['val_mae_voting']:
            self.best_metrics['val_mae_voting'] = val_metrics_voting['mae']
            epoch_metrics['best_val_mae_voting'] = val_metrics_voting['mae']
        
        if val_metrics_voting['qwk'] > self.best_metrics['val_qwk_voting']:
            self.best_metrics['val_qwk_voting'] = val_metrics_voting['qwk']
            epoch_metrics['best_val_qwk_voting'] = val_metrics_voting['qwk']
        
        wandb.log(epoch_metrics)
        
    def create_comparison_plot(self, epoch):
        """Create comparison plots for single vs voting mode"""
        if epoch > 1:
            # MAE comparison
            mae_data = [[x, y1, y2] for x, y1, y2 in zip(
                range(1, epoch + 1),
                self.metrics_history['val_mae_single'][:epoch],
                self.metrics_history['val_mae_voting'][:epoch]
            )]
            
            mae_table = wandb.Table(
                data=mae_data,
                columns=["Epoch", "MAE Single", "MAE Voting"]
            )
            
            wandb.log({
                "mae_comparison": wandb.plot.line(
                    mae_table, "Epoch", ["MAE Single", "MAE Voting"],
                    title="MAE: Single vs Voting Mode"
                )
            })
            
            # QWK comparison
            qwk_data = [[x, y1, y2] for x, y1, y2 in zip(
                range(1, epoch + 1),
                self.metrics_history['val_qwk_single'][:epoch],
                self.metrics_history['val_qwk_voting'][:epoch]
            )]
            
            qwk_table = wandb.Table(
                data=qwk_data,
                columns=["Epoch", "QWK Single", "QWK Voting"]
            )
            
            wandb.log({
                "qwk_comparison": wandb.plot.line(
                    qwk_table, "Epoch", ["QWK Single", "QWK Voting"],
                    title="QWK: Single vs Voting Mode"
                )
            })
    
    def log_learning_rate(self, lr, step):
        """Log learning rate changes"""
        wandb.log({"learning_rate": lr}, step=step)
    
    def log_gradient_stats(self, model, step):
        """Log gradient statistics for monitoring"""
        total_norm = 0
        param_count = 0
        grad_norms = []
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                grad_norms.append(param_norm.item())
                param_count += 1
        
        total_norm = total_norm ** (1. / 2.)
        
        if param_count > 0:
            wandb.log({
                "gradient_norm": total_norm,
                "gradient_norm_mean": np.mean(grad_norms),
                "gradient_norm_max": np.max(grad_norms),
                "gradient_norm_min": np.min(grad_norms)
            }, step=step)
    
    def log_model_stats(self, model):
        """Log model parameter statistics"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.log({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/frozen_parameters": total_params - trainable_params
        })
    
    def create_final_summary(self):
        """Create final training summary"""
        summary = {
            "best_val_mae_single": self.best_metrics['val_mae'],
            "best_val_qwk_single": self.best_metrics['val_qwk'],
            "best_val_mae_voting": self.best_metrics['val_mae_voting'],
            "best_val_qwk_voting": self.best_metrics['val_qwk_voting']
        }
        
        wandb.summary.update(summary)
        
        # Create final comparison table
        final_table = wandb.Table(
            columns=["Metric", "Single Mode", "Voting Mode", "Improvement"],
            data=[
                ["Best MAE", self.best_metrics['val_mae'], 
                 self.best_metrics['val_mae_voting'],
                 f"{(self.best_metrics['val_mae'] - self.best_metrics['val_mae_voting']):.4f}"],
                ["Best QWK", self.best_metrics['val_qwk'],
                 self.best_metrics['val_qwk_voting'],
                 f"{(self.best_metrics['val_qwk_voting'] - self.best_metrics['val_qwk']):.4f}"]
            ]
        )
        
        wandb.log({"final_comparison": final_table})