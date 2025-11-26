import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def compute_metrics(y_true, y_pred, average='weighted'):
    """
    Compute classification metrics: Accuracy, Precision, Recall, F1-Score
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('weighted', 'macro', 'micro')
    
    Returns:
        Dictionary containing accuracy, precision, recall, and F1 score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def calculate_confusion_matrix(y_true, y_pred, num_classes=8):
    """
    Calculate confusion matrix for classification results
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classification classes
    
    Returns:
        Confusion matrix as numpy array
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return cm


def per_class_metrics(y_true, y_pred, class_names=None):
    """
    Compute per-class Precision, Recall, and F1-Score
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
    
    Returns:
        Dictionary with per-class metrics
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    num_classes = len(precision)
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    results = {}
    for i in range(num_classes):
        results[class_names[i]] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': support[i]
        }
    
    return results
