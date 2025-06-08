# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import wandb
from datetime import datetime
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.utils import get_logger, load_config, create_directories, save_checkpoint
from utils.reader import get_data_loaders
from utils.metrics import calculate_metrics, mean_absolute_error
from utils.data_prepare import create_pairwise_data, prepare_reference_samples, PairwiseDataset
from utils.loss_monitor import LossMonitor
from utils.train_monitor import TrainingMonitor
from networks.core_networks import NPCRModel, NPCRModelWithMultiSampleVoting
from evaluator import Evaluator


def train_epoch(model, train_loader, optimizer, criterion, scheduler, device, epoch, config, loss_monitor, global_step, logger):
    """Train for one epoch with enhanced logging and techniques"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        step_loss = 0
        
        if len(batch) == 5:  # Pairwise mode
            input_ids1, attention_mask1, input_ids2, attention_mask2, relative_scores = batch
            
            # Move to device
            input_ids1 = input_ids1.to(device)
            attention_mask1 = attention_mask1.to(device)
            input_ids2 = input_ids2.to(device)
            attention_mask2 = attention_mask2.to(device)
            relative_scores = relative_scores.to(device)
            
            # Add label smoothing if configured
            label_smoothing = float(config['training'].get('label_smoothing', 0.0))
            if label_smoothing > 0:
                relative_scores = relative_scores * (1 - label_smoothing) + 0.5 * label_smoothing
            
            # Forward pass
            outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            
            # Check for NaN in outputs
            if torch.isnan(outputs).any():
                logger.warning(f"NaN detected in model outputs at batch {batch_idx}")
                continue
                
            loss = criterion(outputs, relative_scores)
            
        else:  # Single mode
            input_ids, attention_mask, scores, _ = batch
            
            # Move to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            scores = scores.to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Check for NaN in outputs
            if torch.isnan(outputs).any():
                logger.warning(f"NaN detected in model outputs at batch {batch_idx}")
                continue
                
            loss = criterion(outputs, scores)
            
            # Store for metrics (using MAE for evaluation)
            predictions.extend(outputs.detach().cpu().numpy())
            targets.extend(scores.detach().cpu().numpy())
        
        # Check for NaN in loss
        if torch.isnan(loss):
            logger.warning(f"NaN loss detected at batch {batch_idx}, skipping...")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_clip_norm = float(config['training'].get('gradient_clip_norm', 1.0))
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                grad_clip_norm
            )
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        step_loss = loss.item()
        total_loss += step_loss
        global_step += 1
        
        # Update progress bar
        current_lr = scheduler.get_last_lr()[0] if scheduler else float(config['training']['learning_rate'])
        progress_bar.set_postfix({
            'loss': step_loss,
            'lr': f'{current_lr:.2e}'
        })
        
        # Log to wandb every N steps
        log_every_n = int(config['training'].get('log_every_n_steps', 10))
        if global_step % log_every_n == 0:
            wandb.log({
                'train_step_loss': step_loss,
                'learning_rate': current_lr,
                'global_step': global_step,
                'epoch': epoch
            })
    
    # Calculate average loss
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Calculate training metrics if in single mode
    train_metrics = None
    if predictions:
        # Rescale to original score range
        from utils.utils import rescale_to_score_range, round_to_increment
        predictions_rescaled = rescale_to_score_range(
            np.array(predictions), 
            config['score']['min_score'], 
            config['score']['max_score']
        )
        targets_rescaled = rescale_to_score_range(
            np.array(targets), 
            config['score']['min_score'], 
            config['score']['max_score']
        )
        predictions_rounded = round_to_increment(predictions_rescaled, config['score']['increment'])
        
        # Use MAE for evaluation
        mae = mean_absolute_error(targets_rescaled, predictions_rounded)
        train_metrics = calculate_metrics(targets_rescaled, predictions_rounded)
        train_metrics['mae'] = mae
    
    return avg_loss, train_metrics, global_step

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="NPCR Training for Speaking Scoring")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for testing')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    create_directories(config)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config['training']['log_dir'], f'training_{timestamp}.log')
    logger = get_logger("Main", log_file=log_file)
    
    # Initialize wandb
    wandb.login(key=config['wandb']['key'], relogin=True)
    wandb.init(
        project=config['wandb']['project_name'],
        config=config,
        name=f"npcr_{config['model']['type']}_{timestamp}"
    )
    
    # Set device
    device = torch.device(config['device']['cuda_device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, train_data, val_data, test_data = get_data_loaders(config)
    logger.info(f"Train samples: {len(train_data['input_ids'])}")
    logger.info(f"Val samples: {len(val_data['input_ids'])}")
    logger.info(f"Test samples: {len(test_data['input_ids'])}")
    
    # Create pairwise training data
    logger.info("Creating pairwise training data...")
    pairwise_train_data = create_pairwise_data(train_data, config)
    
    if pairwise_train_data is not None:
        pairwise_dataset = PairwiseDataset(pairwise_train_data)
        pairwise_train_loader = torch.utils.data.DataLoader(
            pairwise_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['device']['num_workers']
        )
        logger.info(f"Created {len(pairwise_dataset)} pairwise training samples")
    else:
        logger.warning("No pairwise data created, using single mode only")
        pairwise_train_loader = None
    
    # Prepare reference samples for multi-sample voting
    logger.info("Preparing reference samples...")
    reference_data = prepare_reference_samples(train_data, config)
    
    # Initialize model
    logger.info("Initializing model...")
    
    # Check model type
    model_type = config['model'].get('model_type', 'bert')
    
    if model_type == 'stella' or 'stella' in config['model']['pretrained_model'].lower():
        logger.info("Using Stella model architecture")
        from networks.stella_npcr import StellaNPCRModel
        model = StellaNPCRModel(config)
    else:
        logger.info("Using standard BERT architecture")
        model = NPCRModel(config)
    
    model = model.to(device)
    logger.info(f"Model loaded: {config['model']['pretrained_model']}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize loss function (MSE as in original paper)
    if config['training']['loss_function'] == 'mse':
        criterion = nn.MSELoss()
    elif config['training']['loss_function'] == 'l1':
        criterion = nn.L1Loss()
    elif config['training']['loss_function'] == 'smooth_l1':
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()  # Default to MSE
    
    logger.info(f"Using {config['training']['loss_function'].upper()} loss for training")
    
    # Calculate total training steps
    steps_per_epoch = len(train_loader)
    if pairwise_train_loader:
        steps_per_epoch += len(pairwise_train_loader)
    total_steps = steps_per_epoch * config['training']['num_epochs']
    
    # Initialize scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Initialize evaluator
    evaluator = Evaluator(config, model, reference_data)
    
    # Initialize loss monitor
    loss_monitor = LossMonitor(save_dir=config['training']['log_dir'])
    
    # Initialize training monitor
    train_monitor = TrainingMonitor(config)
    train_monitor.log_model_stats(model)
    
    if args.mode == 'train':
        # Training loop
        logger.info("Starting training...")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Warmup steps: {config['training']['warmup_steps']}")
        
        global_step = 0
        best_val_mae = float('inf')
        early_stopping_counter = 0
        
        for epoch in range(1, config['training']['num_epochs'] + 1):
            logger.info(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
            
            # Train with pairwise data if available
            if pairwise_train_loader is not None:
                logger.info("Training with pairwise data...")
                train_loss_pair, _, global_step = train_epoch(
                    model, pairwise_train_loader, optimizer, criterion, 
                    scheduler, device, epoch, config, loss_monitor, global_step, logger
                )
                logger.info(f"Pairwise training loss: {train_loss_pair:.4f}")
                
                # Monitor pairwise loss
                loss_monitor.add_pairwise_loss(epoch, train_loss_pair)
                loss_status = loss_monitor.get_loss_status(train_loss_pair, loss_monitor.pairwise_losses)
                logger.info(f"Loss Status: {loss_status}")
                
                wandb.log({
                    'train_loss_pairwise': train_loss_pair,
                    'epoch': epoch
                })
            
            # Also train with single mode for better convergence
            logger.info("Training with single mode...")
            train_loss_single, train_metrics, global_step = train_epoch(
                model, train_loader, optimizer, criterion,
                scheduler, device, epoch, config, loss_monitor, global_step, logger
            )
            logger.info(f"Single mode training loss: {train_loss_single:.4f}")
            
            # Monitor single loss
            loss_monitor.add_single_loss(epoch, train_loss_single)
            
            # Log training metrics
            if train_metrics:
                logger.info(f"Training MAE: {train_metrics['mae']:.4f}")
                logger.info(f"Training QWK: {train_metrics['qwk']:.4f}")
                wandb.log({
                    'train_loss_single': train_loss_single,
                    'train_mae': train_metrics['mae'],
                    'train_qwk': train_metrics['qwk'],
                    'train_rmse': train_metrics['rmse'],
                    'epoch': epoch
                })
            
            # Validation with both modes
            logger.info("Evaluating on validation set...")
            
            # Single mode evaluation
            logger.info("- Single mode evaluation")
            val_metrics_single, _ = evaluator.evaluate_single_mode(val_loader, phase='val')
            
            # Multi-sample voting evaluation
            logger.info("- Multi-sample voting evaluation")
            val_metrics_voting, _ = evaluator.evaluate_with_voting(val_loader, phase='val')
            
            # Monitor validation metrics
            loss_monitor.add_val_metrics(epoch, val_metrics_single)
            
            # Log comprehensive metrics
            train_monitor.log_epoch_metrics(
                epoch, train_metrics, val_metrics_single, val_metrics_voting
            )
            
            # Create comparison plots
            train_monitor.create_comparison_plot(epoch)
            
            # Save best model based on voting MAE
            evaluator.save_best_model(model, optimizer, epoch, val_metrics_single)
            
            # Early stopping check
            if val_metrics_single['mae'] < best_val_mae:
                best_val_mae = val_metrics_single['mae']
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter >= config['training']['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Print training summary every 5 epochs
            if epoch % 5 == 0:
                logger.info("\n" + loss_monitor.get_summary())
                
            # Check convergence
            if epoch > 20:
                converged, msg = loss_monitor.check_convergence()
                if converged:
                    logger.info(f"Training converged: {msg}")
        
        # Create final training summary
        train_monitor.create_final_summary()
        
        # Print best results
        evaluator.print_best_results()
        
        # Save loss history and plot
        loss_monitor.save_history()
        loss_monitor.plot_losses(save_path=os.path.join(config['training']['log_dir'], 'training_curves.png'))
        logger.info(f"Training curves saved to {config['training']['log_dir']}/training_curves.png")
        
        # Test evaluation with best model
        logger.info("\nEvaluating on test set with best QWK model...")
        best_checkpoint_path = os.path.join(config['training']['checkpoint_dir'], 'best_model_qwk.pt')
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Test with single mode
            logger.info("Testing with single mode...")
            test_metrics_single, _ = evaluator.evaluate_single_mode(test_loader, phase='test')
            
            # Test with multi-sample voting
            logger.info("Testing with multi-sample voting...")
            test_metrics_voting, _ = evaluator.evaluate_with_voting(test_loader, phase='test')
            
            # Log test results
            wandb.log({
                'test_mae_single': test_metrics_single['mae'],
                'test_qwk_single': test_metrics_single['qwk'],
                'test_rmse_single': test_metrics_single['rmse'],
                'test_mae_voting': test_metrics_voting['mae'],
                'test_qwk_voting': test_metrics_voting['qwk'],
                'test_rmse_voting': test_metrics_voting['rmse']
            })
            
            # Create summary table
            summary_data = [
                ["Metric", "Val Single", "Val Voting", "Test Single", "Test Voting"],
                ["MAE", f"{val_metrics_single['mae']:.4f}", f"{val_metrics_voting['mae']:.4f}", 
                 f"{test_metrics_single['mae']:.4f}", f"{test_metrics_voting['mae']:.4f}"],
                ["QWK", f"{val_metrics_single['qwk']:.4f}", f"{val_metrics_voting['qwk']:.4f}",
                 f"{test_metrics_single['qwk']:.4f}", f"{test_metrics_voting['qwk']:.4f}"],
                ["RMSE", f"{val_metrics_single['rmse']:.4f}", f"{val_metrics_voting['rmse']:.4f}",
                 f"{test_metrics_single['rmse']:.4f}", f"{test_metrics_voting['rmse']:.4f}"]
            ]
            
            wandb.log({"summary_table": wandb.Table(data=summary_data)})
    
    else:  # Test mode
        if args.checkpoint is None:
            raise ValueError("Checkpoint path required for test mode")
        
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test evaluation
        logger.info("Evaluating on test set...")
        test_metrics_single, _ = evaluator.evaluate_single_mode(test_loader, phase='test')
        test_metrics_voting, _ = evaluator.evaluate_with_voting(test_loader, phase='test')
    
    # Finish wandb
    wandb.finish()
    logger.info("Training completed!")


if __name__ == '__main__':
    main()