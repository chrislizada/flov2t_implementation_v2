import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

from config.model_config import get_model_config
from models.rtfe import RTFEModule
from data.dataset import MaliciousTrafficDataset
from utils.logger import setup_logger
from utils.metrics import compute_metrics, calculate_confusion_matrix, per_class_metrics
from utils.visualization import plot_confusion_matrix, save_results_summary


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='FLoV2T Model Evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = load_config(args.config)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        log_dir=str(output_dir),
        name='flov2t_eval'
    )
    
    logger.info("="*60)
    logger.info("FLoV2T Model Evaluation")
    logger.info("="*60)
    logger.info(f"Model checkpoint: {args.model_path}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*60 + "\n")
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    logger.info("Loading test dataset...")
    test_dataset = MaliciousTrafficDataset(
        root_dir=args.test_data,
        attack_categories=config['dataset']['attack_categories']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    class_names = test_dataset.get_class_names()
    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"Number of classes: {len(class_names)}")
    logger.info(f"Class names: {class_names}\n")
    
    logger.info("Loading model...")
    model = RTFEModule(
        num_classes=config['model']['num_classes'],
        backbone=config['model']['backbone'],
        rank=config['lora']['rank'],
        alpha=config['lora']['alpha'],
        dropout=config['lora']['dropout'],
        pretrained=False
    )
    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'global_model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['global_model_state_dict'])
        logger.info(f"Loaded checkpoint from round {checkpoint.get('round', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded model state dict")
    
    model.to(device)
    model.eval()
    
    logger.info("\nStarting evaluation...")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {(batch_idx + 1) * args.batch_size}/{len(test_dataset)} samples")
    
    logger.info("\n" + "="*60)
    logger.info("Evaluation Results")
    logger.info("="*60 + "\n")
    
    overall_metrics = compute_metrics(all_labels, all_preds, average='weighted')
    
    logger.info("Overall Metrics:")
    logger.info(f"  Accuracy:  {overall_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {overall_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {overall_metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {overall_metrics['f1_score']:.4f}")
    logger.info("")
    
    class_metrics = per_class_metrics(all_labels, all_preds, class_names=class_names)
    
    logger.info("Per-Class Metrics:")
    logger.info(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    logger.info("-" * 70)
    for class_name, metrics in class_metrics.items():
        logger.info(
            f"{class_name:<20} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} "
            f"{metrics['f1_score']:<12.4f} "
            f"{metrics['support']:<10}"
        )
    logger.info("")
    
    cm = calculate_confusion_matrix(
        all_labels,
        all_preds,
        num_classes=len(class_names)
    )
    
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, class_names, save_path=str(cm_path), normalize=False)
    logger.info(f"Confusion matrix saved: {cm_path}")
    
    cm_norm_path = output_dir / 'confusion_matrix_normalized.png'
    plot_confusion_matrix(cm, class_names, save_path=str(cm_norm_path), normalize=True)
    logger.info(f"Normalized confusion matrix saved: {cm_norm_path}")
    
    results = {
        'overall': overall_metrics,
        'per_class': class_metrics
    }
    
    summary_path = output_dir / 'evaluation_summary.txt'
    save_results_summary(results, save_path=str(summary_path))
    logger.info(f"Evaluation summary saved: {summary_path}")
    
    predictions_path = output_dir / 'predictions.npz'
    np.savez(
        predictions_path,
        predictions=np.array(all_preds),
        labels=np.array(all_labels),
        probabilities=np.array(all_probs),
        class_names=np.array(class_names)
    )
    logger.info(f"Predictions saved: {predictions_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Evaluation Complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
