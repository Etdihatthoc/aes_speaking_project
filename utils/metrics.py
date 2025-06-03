# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn


def kappa(y_true, y_pred, weights='quadratic'):
    """
    Calculate Cohen's kappa with quadratic weights
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Round to integers
    y_true = np.round(y_true).astype(int)
    y_pred = np.round(y_pred).astype(int)
    
    # Ensure same length
    assert len(y_true) == len(y_pred)
    
    # Get min and max ratings
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))
    
    # Shift values so that the lowest value is 0
    y_true = y_true - min_rating
    y_pred = y_pred - min_rating
    
    # Build confusion matrix
    num_ratings = max_rating - min_rating + 1
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(num_ratings)))
    num_scored_items = float(len(y_true))
    
    # Build weight matrix
    weights_mat = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            diff = abs(i - j)
            if weights == 'linear':
                weights_mat[i, j] = diff
            elif weights == 'quadratic':
                weights_mat[i, j] = diff ** 2
            else:  # unweighted
                weights_mat[i, j] = bool(diff)
    
    # Calculate expected scores
    hist_true = np.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[:num_ratings] / num_scored_items
    hist_pred = np.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[:num_ratings] / num_scored_items
    expected = np.outer(hist_true, hist_pred)
    
    # Normalize matrices
    conf_mat = conf_mat / num_scored_items
    
    # Calculate kappa
    k = 1.0
    if np.count_nonzero(weights_mat):
        k -= (np.sum(weights_mat * conf_mat) / np.sum(weights_mat * expected))
    
    return k


def pearson(y_true, y_pred):
    """Calculate Pearson correlation coefficient"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) < 2:
        return 0.0
    
    corr, _ = pearsonr(y_true, y_pred)
    return corr if not np.isnan(corr) else 0.0


def spearman(y_true, y_pred):
    """Calculate Spearman's rank correlation coefficient"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) < 2:
        return 0.0
    
    corr, _ = spearmanr(y_true, y_pred)
    return corr if not np.isnan(corr) else 0.0


def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error (L1 Loss)"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.mean(np.abs(y_true - y_pred))


def root_mean_square_error(y_true, y_pred):
    """Calculate Root Mean Square Error"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


class L1Loss(nn.Module):
    """L1 Loss (MAE) for PyTorch"""
    def __init__(self):
        super(L1Loss, self).__init__()
        self.mae = nn.L1Loss()
    
    def forward(self, predictions, targets):
        return self.mae(predictions, targets)


def calculate_metrics(y_true, y_pred):
    """Calculate all metrics"""
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_square_error(y_true, y_pred),
        'qwk': kappa(y_true, y_pred, weights='quadratic'),
        'pearson': pearson(y_true, y_pred),
        'spearman': spearman(y_true, y_pred)
    }
    return metrics