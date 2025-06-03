# -*- coding: utf-8 -*-
import logging
import sys
import numpy as np
import torch
import torch.nn as nn
import yaml
import os
from datetime import datetime


def get_logger(name, level=logging.INFO, log_file=None):
    """Create logger with console and file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert string values to appropriate types
    config['training']['num_epochs'] = int(config['training']['num_epochs'])
    config['training']['batch_size'] = int(config['training']['batch_size'])
    config['training']['learning_rate'] = float(config['training']['learning_rate'])
    config['training']['weight_decay'] = float(config['training']['weight_decay'])
    config['training']['example_size'] = int(config['training']['example_size'])
    
    # New parameters
    config['training']['warmup_steps'] = int(config['training'].get('warmup_steps', 500))
    config['training']['gradient_clip_norm'] = float(config['training'].get('gradient_clip_norm', 1.0))
    config['training']['dropout_rate'] = float(config['training'].get('dropout_rate', 0.5))
    config['training']['label_smoothing'] = float(config['training'].get('label_smoothing', 0.0))
    config['training']['early_stopping_patience'] = int(config['training'].get('early_stopping_patience', 10))
    config['training']['log_every_n_steps'] = int(config['training'].get('log_every_n_steps', 10))
    
    config['model']['max_length'] = int(config['model']['max_length'])
    config['model']['dropout'] = float(config['model']['dropout'])
    config['model']['hidden_dim'] = int(config['model']['hidden_dim'])
    
    config['score']['min_score'] = float(config['score']['min_score'])
    config['score']['max_score'] = float(config['score']['max_score'])
    config['score']['increment'] = float(config['score']['increment'])
    
    config['device']['num_workers'] = int(config['device']['num_workers'])
    
    return config


def create_directories(config):
    """Create necessary directories"""
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)


def rescale_to_score_range(scaled_scores, min_score, max_score):
    """Rescale normalized scores [0,1] to actual score range"""
    return scaled_scores * (max_score - min_score) + min_score


def normalize_scores(scores, min_score, max_score):
    """Normalize scores to [0,1] range"""
    return (scores - min_score) / (max_score - min_score)


def round_to_increment(scores, increment=0.5):
    """Round scores to nearest increment"""
    return np.round(scores / increment) * increment


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    

def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics']


class TimeDistributed(nn.Module):
    """TimeDistributed wrapper for PyTorch"""
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
        
        input_shape = input_seq.shape
        batch_size = input_shape[0]
        time_steps = input_shape[1]
        
        # Reshape to (batch_size * time_steps, ...)
        reshaped_input = input_seq.contiguous().view(-1, *input_shape[2:])
        output = self.module(reshaped_input)
        
        # Reshape back to (batch_size, time_steps, ...)
        if type(output) == tuple:
            output = output[0]
        output = output.contiguous().view(batch_size, time_steps, -1)
        
        return output