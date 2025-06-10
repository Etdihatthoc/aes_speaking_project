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

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from utils.utils import get_logger, load_config, create_directories, save_checkpoint
from utils.reader import get_data_loaders
from utils.metrics import calculate_metrics, mean_absolute_error
from utils.data_prepare import create_pairwise_data, prepare_reference_samples, PairwiseDataset
from utils.loss_monitor import LossMonitor
from utils.train_monitor import TrainingMonitor
from evaluator import Evaluator


def clear_gpu_memory():
    """Clear GPU memory to prevent OOM errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def initialize_model(config, device):
    """Initialize model with enhanced error handling for multi-scale architecture"""
    logger = get_logger("Model Init")
    
    try:
        # Check model type
        model_type = config['model'].get('model_type', 'bert')
        
        if model_type == 'stella' or 'stella' in config['model']['pretrained_model'].lower():
            logger.info("Using Stella model architecture")
            try:
                from networks.stella_npcr import StellaNPCRModel
                model = StellaNPCRModel(config)
            except ImportError:
                logger.warning("Stella model not available, falling back to BERT")
                from networks.core_networks import NPCRModel
                model = NPCRModel(config)
        else:
            logger.info("Using multi-scale BERT architecture")
            from networks.core_networks import NPCRModel
            model = NPCRModel(config)
        
        model = model.to(device)
        
        # Log model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model loaded: {config['model']['pretrained_model']}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Estimate model size
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        logger.info(f"Estimated model size: {model_size_mb:.1f} MB")
        
        # Test model with dummy input to ensure it works
        try:
            dummy_input_ids = torch.ones(2, 10, dtype=torch.long).to(device)
            dummy_attention_mask = torch.ones(2, 10, dtype=torch.long).to(device)
            
            model.eval()
            with torch.no_grad():
                # Test single mode
                test_output = model(dummy_input_ids, dummy_attention_mask)
                logger.info(f"Single mode test successful. Output shape: {test_output.shape}")
                
                # Test pairwise mode
                test_output_pair = model(
                    dummy_input_ids, dummy_attention_mask,
                    dummy_input_ids, dummy_attention_mask
                )
                logger.info(f"Pairwise mode test successful. Output shape: {test_output_pair.shape}")
                
                # Check if outputs are valid
                if torch.isnan(test_output).any() or torch.isinf(test_output).any():
                    logger.warning("Model produces NaN/Inf outputs in single mode!")
                elif torch.isnan(test_output_pair).any() or torch.isinf(test_output_pair).any():
                    logger.warning("Model produces NaN/Inf outputs in pairwise mode!")
                else:
                    logger.info("Model produces valid outputs in both modes")
                    
                # Check output ranges
                single_range = f"{test_output.min().item():.4f} - {test_output.max().item():.4f}"
                pair_range = f"{test_output_pair.min().item():.4f} - {test_output_pair.max().item():.4f}"
                logger.info(f"Output ranges - Single: {single_range}, Pairwise: {pair_range}")
                
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            raise
            
        model.train()
        return model
        
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        logger.error("Attempting fallback to basic BERT model...")
        
        try:
            # Fallback to basic implementation
            from networks.core_networks_2 import NPCRModel
            model = NPCRModel(config)
            model = model.to(device)
            logger.info("Fallback model loaded successfully")
            return model
        except Exception as fallback_error:
            logger.error(f"Fallback model also failed: {fallback_error}")
            raise


def validate_batch_outputs(outputs, batch_idx, mode="single"):
    """Validate and clean model outputs to prevent NaN losses"""
    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
        logger = get_logger("Validation")
        logger.warning(f"NaN/Inf outputs detected in {mode} mode at batch {batch_idx}")
        
        # Count problematic values
        nan_count = torch.isnan(outputs).sum().item()
        inf_count = torch.isinf(outputs).sum().item()
        logger.warning(f"  NaN values: {nan_count}, Inf values: {inf_count}")
        
        # Clean outputs
        outputs = torch.clamp(outputs, 0.0, 1.0)
        outputs[torch.isnan(outputs)] = 0.5
        outputs[torch.isinf(outputs)] = 0.5
        
        logger.warning(f"  Cleaned outputs to range: {outputs.min().item():.4f} - {outputs.max().item():.4f}")
    
    # Ensure outputs are in valid range
    outputs = torch.clamp(outputs, 0.0, 1.0)
    return outputs


def train_epoch(model, train_loader, optimizer, criterion, scheduler, device, epoch, config, loss_monitor, global_step):
    """Train for one epoch with enhanced logging and techniques"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    batch_count = 0
    
    logger = get_logger("Training")
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        step_loss = 0
        
        try:
            if len(batch) == 5:  # Pairwise mode
                input_ids1, attention_mask1, input_ids2, attention_mask2, relative_scores = batch
                
                # Move to device
                input_ids1 = input_ids1.to(device)
                attention_mask1 = attention_mask1.to(device)
                input_ids2 = input_ids2.to(device)
                attention_mask2 = attention_mask2.to(device)
                relative_scores = relative_scores.to(device)
                
                # Validate inputs
                if torch.isnan(relative_scores).any():
                    logger.warning(f"NaN relative scores in batch {batch_idx}, cleaning...")
                    relative_scores[torch.isnan(relative_scores)] = 0.5
                
                # Add label smoothing if configured
                if config['training']['label_smoothing'] > 0:
                    smoothing = config['training']['label_smoothing']
                    relative_scores = relative_scores * (1 - smoothing) + 0.5 * smoothing
                
                # Forward pass
                outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
                
                # Validate outputs
                outputs = validate_batch_outputs(outputs, batch_idx, "pairwise")
                
                loss = criterion(outputs, relative_scores)
                
            else:  # Single mode
                input_ids, attention_mask, scores, _ = batch
                
                # Move to device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                scores = scores.to(device)
                
                # Validate inputs
                if torch.isnan(scores).any():
                    logger.warning(f"NaN scores in batch {batch_idx}, cleaning...")
                    scores[torch.isnan(scores)] = 0.5
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                
                # Validate outputs
                outputs = validate_batch_outputs(outputs, batch_idx, "single")
                
                loss = criterion(outputs, scores)
                
                # Store for metrics (using MAE for evaluation)
                predictions.extend(outputs.detach().cpu().numpy())
                targets.extend(scores.detach().cpu().numpy())
            
            # Validate loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss in batch {batch_idx}, skipping...")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2.)
            
            if total_norm > 100:  # Large gradient warning
                logger.warning(f"Large gradient norm: {total_norm:.2f} in batch {batch_idx}")
            
            # Gradient clipping
            if config['training']['gradient_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['training']['gradient_clip_norm']
                )
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # Update metrics
            step_loss = loss.item()
            total_loss += step_loss
            global_step += 1
            batch_count += 1
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0] if scheduler else config['training']['learning_rate']
            progress_bar.set_postfix({
                'loss': f"{step_loss:.4f}",
                'lr': f'{current_lr:.2e}',
                'grad_norm': f'{total_norm:.2f}'
            })
            
            # Log to wandb every N steps
            if global_step % config['training']['log_every_n_steps'] == 0:
                wandb.log({
                    'train_step_loss': step_loss,
                    'learning_rate': current_lr,
                    'gradient_norm': total_norm,
                    'global_step': global_step,
                    'epoch': epoch
                })
                
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            continue
    
    if batch_count == 0:
        logger.error("No valid batches processed!")
        return float('inf'), None, global_step
    
    avg_loss = total_loss / batch_count
    
    # Calculate training metrics if in single mode
    train_metrics = None
    if predictions:
        try:
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
            
        except Exception as e:
            logger.warning(f"Error calculating training metrics: {e}")
    
    logger.info(f"Epoch {epoch} training completed. Avg loss: {avg_loss:.4f}")
    return avg_loss, train_metrics, global_step


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Multi-Scale NPCR Training for Speaking Scoring")
    parser.add_argument('--config', type=str, default='config/config_multiscale.yaml', 
                       help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
                       help='Mode: train or test')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Path to checkpoint for testing')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode with extra logging')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    create_directories(config)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config['training']['log_dir'], f'training_{timestamp}.log')
    logger = get_logger("Main", log_file=log_file)
    
    logger.info("="*60)
    logger.info("MULTI-SCALE NPCR TRAINING STARTED")
    logger.info("="*60)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Debug mode: {args.debug}")
    
    # Initialize wandb
    try:
        wandb.login(key=config['wandb']['key'], relogin=True)
        run_name = f"multiscale_npcr_{config['model']['type']}_{timestamp}"
        wandb.init(
            project=config['wandb']['project_name'],
            config=config,
            name=run_name
        )
        logger.info(f"Wandb initialized: {run_name}")
    except Exception as e:
        logger.warning(f"Wandb initialization failed: {e}")
    
    # Set device
    device = torch.device(config['device']['cuda_device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    logger.info("Loading data...")
    try:
        train_loader, val_loader, test_loader, train_data, val_data, test_data = get_data_loaders(config)
        logger.info(f"Data loading successful:")
        logger.info(f"  Train samples: {len(train_data['input_ids'])}")
        logger.info(f"  Val samples: {len(val_data['input_ids'])}")
        logger.info(f"  Test samples: {len(test_data['input_ids'])}")
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise
    
    # Create pairwise training data
    logger.info("Creating pairwise training data...")
    pairwise_train_data = create_pairwise_data(train_data, config)
    
    if pairwise_train_data is not None:
        try:
            pairwise_dataset = PairwiseDataset(pairwise_train_data)
            pairwise_train_loader = torch.utils.data.DataLoader(
                pairwise_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=True,
                num_workers=config['device']['num_workers']
            )
            logger.info(f"Created {len(pairwise_dataset)} pairwise training samples")
        except Exception as e:
            logger.error(f"Error creating pairwise loader: {e}")
            pairwise_train_loader = None
    else:
        logger.warning("No pairwise data created, using single mode only")
        pairwise_train_loader = None
    
    # Prepare reference samples for multi-sample voting
    logger.info("Preparing reference samples...")
    reference_data = prepare_reference_samples(train_data, config)
    if reference_data is not None:
        logger.info(f"Reference samples prepared: {len(reference_data['input_ids'])}")
    else:
        logger.warning("No reference data prepared")
    
    # Initialize model
    logger.info("Initializing multi-scale model...")
    model = initialize_model(config, device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize loss function
    loss_function = config['training']['loss_function'].lower()
    if loss_function == 'mse':
        criterion = nn.MSELoss()
    elif loss_function == 'l1':
        criterion = nn.L1Loss()
    elif loss_function == 'smooth_l1':
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()  # Default to MSE
    
    logger.info(f"Using {loss_function.upper()} loss for training")
    
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
    
    # Initialize monitors
    loss_monitor = LossMonitor(save_dir=config['training']['log_dir'])
    train_monitor = TrainingMonitor(config)
    train_monitor.log_model_stats(model)
    
    if args.mode == 'train':
        # Training loop
        logger.info("Starting training...")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Warmup steps: {config['training']['warmup_steps']}")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        
        global_step = 0
        best_val_mae = float('inf')
        early_stopping_counter = 0
        
        for epoch in range(1, config['training']['num_epochs'] + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"EPOCH {epoch}/{config['training']['num_epochs']}")
            logger.info(f"{'='*50}")
            
            # Clear GPU memory at start of epoch
            clear_gpu_memory()
            
            # Train with pairwise data if available
            # if pairwise_train_loader is not None:
            #     logger.info("Training with pairwise data...")
            #     train_loss_pair, _, global_step = train_epoch(
            #         model, pairwise_train_loader, optimizer, criterion, 
            #         scheduler, device, epoch, config, loss_monitor, global_step
            #     )
            #     logger.info(f"Pairwise training loss: {train_loss_pair:.4f}")
                
            #     # Monitor pairwise loss
            #     loss_monitor.add_pairwise_loss(epoch, train_loss_pair)
            #     loss_status = loss_monitor.get_loss_status(train_loss_pair, loss_monitor.pairwise_losses)
            #     logger.info(f"Pairwise Loss Status: {loss_status}")
                
            #     wandb.log({
            #         'train_loss_pairwise': train_loss_pair,
            #         'epoch': epoch
            #     })
            
            # Train with single mode for better convergence
            logger.info("Training with single mode...")
            train_loss_single, train_metrics, global_step = train_epoch(
                model, train_loader, optimizer, criterion,
                scheduler, device, epoch, config, loss_monitor, global_step
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
            
            # Save best model based on single mode MAE
            evaluator.save_best_model(model, optimizer, epoch, val_metrics_single)
            
            # Early stopping check
            if val_metrics_single['mae'] < best_val_mae:
                best_val_mae = val_metrics_single['mae']
                early_stopping_counter = 0
                logger.info(f"New best validation MAE: {best_val_mae:.4f}")
            else:
                early_stopping_counter += 1
                logger.info(f"No improvement. Early stopping counter: {early_stopping_counter}/{config['training']['early_stopping_patience']}")
                
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
                    
            # Memory cleanup every few epochs
            if epoch % 3 == 0:
                clear_gpu_memory()
        
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
            try:
                checkpoint = torch.load(best_checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Best model loaded for final evaluation")
                
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
                
                logger.info("\nFINAL RESULTS SUMMARY:")
                logger.info("="*50)
                logger.info("VALIDATION:")
                logger.info(f"  Single Mode  - MAE: {val_metrics_single['mae']:.4f}, QWK: {val_metrics_single['qwk']:.4f}")
                logger.info(f"  Voting Mode  - MAE: {val_metrics_voting['mae']:.4f}, QWK: {val_metrics_voting['qwk']:.4f}")
                logger.info("TEST:")
                logger.info(f"  Single Mode  - MAE: {test_metrics_single['mae']:.4f}, QWK: {test_metrics_single['qwk']:.4f}")
                logger.info(f"  Voting Mode  - MAE: {test_metrics_voting['mae']:.4f}, QWK: {test_metrics_voting['qwk']:.4f}")
                logger.info("="*50)
                
            except Exception as e:
                logger.error(f"Error loading best model for final evaluation: {e}")
    
    else:  # Test mode
        if args.checkpoint is None:
            raise ValueError("Checkpoint path required for test mode")
        
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Checkpoint loaded successfully")
            
            # Test evaluation
            logger.info("Evaluating on test set...")
            test_metrics_single, _ = evaluator.evaluate_single_mode(test_loader, phase='test')
            test_metrics_voting, _ = evaluator.evaluate_with_voting(test_loader, phase='test')
            
            logger.info("TEST RESULTS:")
            logger.info(f"  Single Mode  - MAE: {test_metrics_single['mae']:.4f}, QWK: {test_metrics_single['qwk']:.4f}")
            logger.info(f"  Voting Mode  - MAE: {test_metrics_voting['mae']:.4f}, QWK: {test_metrics_voting['qwk']:.4f}")
            
        except Exception as e:
            logger.error(f"Error in test mode: {e}")
            raise
    
    # Final cleanup
    clear_gpu_memory()
    
    # Finish wandb
    try:
        wandb.finish()
    except:
        pass
    
    logger.info("="*60)
    logger.info("MULTI-SCALE NPCR TRAINING COMPLETED!")
    logger.info("="*60)


if __name__ == '__main__':
    main()