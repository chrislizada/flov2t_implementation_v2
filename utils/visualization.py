import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_confusion_matrix(cm, class_names, save_path=None, normalize=False):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        save_path: Path to save the figure (optional)
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(history, save_path=None, metrics=None):
    """
    Plot training curves (accuracy, loss, F1-score, etc.)
    
    Args:
        history: Dictionary containing metrics history
        save_path: Path to save the figure (optional)
        metrics: List of metrics to plot (if None, plot all available)
    """
    if metrics is None:
        metrics = ['accuracy', 'f1_score', 'loss']
    
    available_metrics = [m for m in metrics if m in history and len(history[m]) > 0]
    
    if not available_metrics:
        print("No metrics available to plot")
        return
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        epochs = range(1, len(history[metric]) + 1)
        ax.plot(epochs, history[metric], 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} vs Epoch', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_client_comparison(client_metrics, save_path=None):
    """
    Plot comparison of metrics across different clients
    
    Args:
        client_metrics: Dictionary with client_id as key and metrics dict as value
        save_path: Path to save the figure (optional)
    """
    client_ids = list(client_metrics.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    x = np.arange(len(client_ids))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for idx, metric in enumerate(metrics):
        values = [client_metrics[cid].get(metric, 0) for cid in client_ids]
        ax.bar(x + idx * width, values, width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Client Performance Comparison', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(client_ids)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results_summary(results, save_path):
    """
    Save results summary to text file
    
    Args:
        results: Dictionary containing results
        save_path: Path to save the summary
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("FLoV2T Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        
        if 'overall' in results:
            f.write("Overall Metrics:\n")
            f.write("-" * 40 + "\n")
            for key, value in results['overall'].items():
                f.write(f"{key.replace('_', ' ').title():20s}: {value:.4f}\n")
            f.write("\n")
        
        if 'per_class' in results:
            f.write("Per-Class Metrics:\n")
            f.write("-" * 40 + "\n")
            for class_name, metrics in results['per_class'].items():
                f.write(f"\n{class_name}:\n")
                for metric, value in metrics.items():
                    if metric != 'support':
                        f.write(f"  {metric.replace('_', ' ').title():15s}: {value:.4f}\n")
                    else:
                        f.write(f"  {metric.replace('_', ' ').title():15s}: {value}\n")
