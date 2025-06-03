# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import json
import os

class LossMonitor:
    """Monitor and analyze training losses"""
    
    def __init__(self, window_size=10, save_dir='./logs'):
        self.pairwise_losses = []
        self.single_losses = []
        self.val_metrics = []
        self.window_size = window_size
        self.save_dir = save_dir
        
    def add_pairwise_loss(self, epoch, loss):
        self.pairwise_losses.append({'epoch': epoch, 'loss': loss})
        
    def add_single_loss(self, epoch, loss):
        self.single_losses.append({'epoch': epoch, 'loss': loss})
        
    def add_val_metrics(self, epoch, metrics):
        self.val_metrics.append({'epoch': epoch, **metrics})
        
    def get_loss_status(self, current_loss, loss_history):
        """Determine if loss is good based on history"""
        if len(loss_history) < 2:
            return "Too early to tell"
        
        # Get recent losses
        recent_losses = [item['loss'] for item in loss_history[-self.window_size:]]
        avg_recent = np.mean(recent_losses)
        
        # Check absolute value
        if current_loss < 0.1:
            status = "Excellent"
        elif current_loss < 0.15:
            status = "Very Good"
        elif current_loss < 0.2:
            status = "Good"
        elif current_loss < 0.25:
            status = "Acceptable"
        else:
            status = "Needs Improvement"
        
        # Check trend
        if len(recent_losses) >= 3:
            trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            if trend < -0.01:
                trend_status = "Improving Fast"
            elif trend < -0.001:
                trend_status = "Improving"
            elif trend < 0.001:
                trend_status = "Stable"
            else:
                trend_status = "Getting Worse"
        else:
            trend_status = "Unknown"
        
        return f"{status} (Trend: {trend_status})"
    
    def check_convergence(self):
        """Check if training has converged"""
        if len(self.pairwise_losses) < 20:
            return False, "Not enough epochs"
        
        recent_losses = [item['loss'] for item in self.pairwise_losses[-10:]]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        # Converged if std is very small relative to mean
        if loss_std / loss_mean < 0.01:
            return True, f"Converged at loss ~{loss_mean:.4f}"
        else:
            return False, f"Still improving (std: {loss_std:.4f})"
    
    def get_recommendations(self):
        """Get training recommendations based on loss patterns"""
        recommendations = []
        
        if len(self.pairwise_losses) < 5:
            return ["Too early for recommendations"]
        
        current_loss = self.pairwise_losses[-1]['loss']
        recent_losses = [item['loss'] for item in self.pairwise_losses[-5:]]
        
        # Check if loss is too high
        if current_loss > 0.3:
            recommendations.append("Loss is high. Consider:")
            recommendations.append("- Increasing model capacity")
            recommendations.append("- Checking data quality")
            recommendations.append("- Adjusting learning rate")
        
        # Check if loss is decreasing too slowly
        if len(self.pairwise_losses) > 10:
            early_loss = self.pairwise_losses[5]['loss']
            improvement = (early_loss - current_loss) / early_loss
            if improvement < 0.2:  # Less than 20% improvement
                recommendations.append("Slow improvement. Try:")
                recommendations.append("- Increasing learning rate")
                recommendations.append("- Adding more training pairs")
        
        # Check for overfitting
        if len(self.val_metrics) > 5:
            train_trend = np.polyfit(range(5), recent_losses, 1)[0]
            val_losses = [m['mae'] for m in self.val_metrics[-5:]]
            val_trend = np.polyfit(range(5), val_losses, 1)[0]
            
            if train_trend < 0 and val_trend > 0:
                recommendations.append("Possible overfitting detected!")
                recommendations.append("- Add dropout or regularization")
                recommendations.append("- Reduce model size")
                recommendations.append("- Use early stopping")
        
        # Check correlation with QWK
        if len(self.val_metrics) > 5:
            losses = [item['loss'] for item in self.pairwise_losses[-5:]]
            qwks = [m['qwk'] for m in self.val_metrics[-5:]]
            correlation = np.corrcoef(losses, qwks)[0, 1]
            
            if correlation > -0.5:  # Weak negative correlation
                recommendations.append("Weak loss-QWK correlation.")
                recommendations.append("- Check if pairs represent score distribution")
                recommendations.append("- Verify reference sample diversity")
        
        return recommendations if recommendations else ["Training looks good!"]
    
    def plot_losses(self, save_path=None):
        """Plot training progress"""
        if len(self.pairwise_losses) < 2:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Pairwise loss
        epochs = [item['epoch'] for item in self.pairwise_losses]
        losses = [item['loss'] for item in self.pairwise_losses]
        ax1.plot(epochs, losses, 'b-', label='Pairwise Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Pairwise Training Loss')
        ax1.grid(True)
        
        # Add good/bad regions
        ax1.axhspan(0, 0.1, alpha=0.2, color='green', label='Excellent')
        ax1.axhspan(0.1, 0.2, alpha=0.2, color='yellow', label='Good')
        ax1.axhspan(0.2, 0.3, alpha=0.2, color='orange', label='Acceptable')
        ax1.axhspan(0.3, 1.0, alpha=0.2, color='red', label='Poor')
        ax1.legend()
        
        # Single mode loss if available
        if self.single_losses:
            epochs_s = [item['epoch'] for item in self.single_losses]
            losses_s = [item['loss'] for item in self.single_losses]
            ax2.plot(epochs_s, losses_s, 'g-', label='Single Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Single Mode Training Loss')
            ax2.grid(True)
            ax2.legend()
        
        # Validation metrics
        if self.val_metrics:
            epochs_v = [item['epoch'] for item in self.val_metrics]
            qwks = [item['qwk'] for item in self.val_metrics]
            maes = [item['mae'] for item in self.val_metrics]
            
            ax3.plot(epochs_v, qwks, 'r-', label='QWK')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('QWK')
            ax3.set_title('Validation QWK')
            ax3.grid(True)
            ax3.legend()
            
            ax4.plot(epochs_v, maes, 'm-', label='MAE')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('MAE')
            ax4.set_title('Validation MAE')
            ax4.grid(True)
            ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def save_history(self):
        """Save training history to JSON"""
        history = {
            'pairwise_losses': self.pairwise_losses,
            'single_losses': self.single_losses,
            'val_metrics': self.val_metrics
        }
        
        save_path = os.path.join(self.save_dir, 'training_history.json')
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def get_summary(self):
        """Get training summary"""
        if not self.pairwise_losses:
            return "No training data yet"
        
        current_loss = self.pairwise_losses[-1]['loss']
        status = self.get_loss_status(current_loss, self.pairwise_losses)
        converged, conv_msg = self.check_convergence()
        
        summary = f"""
Training Summary:
- Current Pairwise Loss: {current_loss:.4f} ({status})
- Convergence: {conv_msg}
- Epochs Trained: {len(self.pairwise_losses)}
"""
        
        if self.val_metrics:
            latest_val = self.val_metrics[-1]
            summary += f"""- Latest Validation:
  - MAE: {latest_val['mae']:.4f}
  - QWK: {latest_val['qwk']:.4f}
  - Pearson: {latest_val['pearson']:.4f}
"""
        
        recommendations = self.get_recommendations()
        summary += "\nRecommendations:\n"
        for rec in recommendations:
            summary += f"  {rec}\n"
        
        return summary