import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(log_dir='./logs', name='flov2t', level=logging.INFO):
    """
    Setup logger for training and evaluation
    
    Args:
        log_dir: Directory to save log files
        name: Logger name
        level: Logging level
    
    Returns:
        Logger instance
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_metrics(logger, epoch, metrics, prefix=''):
    """
    Log metrics to logger
    
    Args:
        logger: Logger instance
        epoch: Current epoch/round number
        metrics: Dictionary of metrics
        prefix: Prefix for log message (e.g., 'Train', 'Val', 'Client_0')
    """
    metric_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    
    if prefix:
        logger.info(f'{prefix} - Epoch {epoch} | {metric_str}')
    else:
        logger.info(f'Epoch {epoch} | {metric_str}')


class MetricsTracker:
    """Track metrics across training rounds/epochs"""
    
    def __init__(self):
        self.history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'loss': []
        }
    
    def update(self, metrics):
        """
        Update metrics history
        
        Args:
            metrics: Dictionary of metrics for current round/epoch
        """
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
            else:
                self.history[key] = [value]
    
    def get_best(self, metric='accuracy'):
        """
        Get best value and epoch for a metric
        
        Args:
            metric: Metric name
        
        Returns:
            Tuple of (best_value, best_epoch)
        """
        if metric not in self.history or len(self.history[metric]) == 0:
            return None, None
        
        if metric == 'loss':
            best_value = min(self.history[metric])
            best_epoch = self.history[metric].index(best_value)
        else:
            best_value = max(self.history[metric])
            best_epoch = self.history[metric].index(best_value)
        
        return best_value, best_epoch
    
    def get_history(self):
        """Get full metrics history"""
        return self.history
