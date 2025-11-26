import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

from config.model_config import get_model_config
from models.rtfe import RTFEModule
from data.dataset import MaliciousTrafficDataset
from data.data_loader import create_federated_dataloaders
from federated.client import FLoV2TClient
from federated.server import RGPAServer
from utils.logger import setup_logger, log_metrics, MetricsTracker
from utils.metrics import compute_metrics, calculate_confusion_matrix
from utils.visualization import plot_training_curves, plot_confusion_matrix


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='FLoV2T Federated Training')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--num_clients', type=int, default=None,
                        help='Number of clients (overrides config)')
    parser.add_argument('--distribution', type=str, default=None,
                        choices=['iid', 'non_iid'],
                        help='Data distribution (overrides config)')
    parser.add_argument('--rounds', type=int, default=None,
                        help='Number of training rounds (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = load_config(args.config)
    
    if args.num_clients:
        config['federated']['num_clients'] = args.num_clients
    if args.distribution:
        config['federated']['distribution'] = args.distribution
    if args.rounds:
        config['federated']['num_rounds'] = args.rounds
    if args.device:
        config['hardware']['device'] = args.device
    
    logger = setup_logger(
        log_dir=config['logging']['log_dir'],
        name='flov2t_train'
    )
    
    logger.info("="*60)
    logger.info("FLoV2T Federated Learning Training")
    logger.info("="*60)
    logger.info(f"Configuration:")
    logger.info(f"  Dataset: {config['dataset']['name']}")
    logger.info(f"  Num clients: {config['federated']['num_clients']}")
    logger.info(f"  Distribution: {config['federated']['distribution']}")
    logger.info(f"  Num rounds: {config['federated']['num_rounds']}")
    logger.info(f"  Device: {config['hardware']['device']}")
    logger.info(f"  Aggregation: {config['federated']['aggregation']['method']}")
    logger.info(f"  Lambda: {config['federated']['aggregation']['lambda_reg']}")
    logger.info("="*60)
    
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if device == 'cuda':
        torch.cuda.manual_seed(config['seed'])
    
    logger.info("Creating federated dataloaders...")
    train_loaders, test_loader, class_names = create_federated_dataloaders(
        root_dir=config['dataset']['root_dir'],
        num_clients=config['federated']['num_clients'],
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        distribution=config['federated']['distribution'],
        non_iid_config=config['federated']['non_iid_config'],
        attack_categories=config['dataset']['attack_categories']
    )
    
    logger.info(f"Created {len(train_loaders)} client dataloaders")
    logger.info(f"Test set size: {len(test_loader.dataset)}")
    
    logger.info("Initializing global model...")
    global_model = RTFEModule(
        num_classes=config['model']['num_classes'],
        backbone=config['model']['backbone'],
        rank=config['lora']['rank'],
        alpha=config['lora']['alpha'],
        dropout=config['lora']['dropout'],
        pretrained=config['model']['pretrained']
    )
    
    logger.info("Initializing server...")
    server = RGPAServer(
        global_model=global_model,
        aggregation_method=config['federated']['aggregation']['method'],
        lambda_reg=config['federated']['aggregation']['lambda_reg'],
        weight_by_samples=config['federated']['aggregation']['weight_by_samples']
    )
    
    logger.info("Initializing clients...")
    clients = []
    for client_id, train_loader in enumerate(train_loaders):
        client_model = RTFEModule(
            num_classes=config['model']['num_classes'],
            backbone=config['model']['backbone'],
            rank=config['lora']['rank'],
            alpha=config['lora']['alpha'],
            dropout=config['lora']['dropout'],
            pretrained=config['model']['pretrained']
        )
        
        client = FLoV2TClient(
            client_id=client_id,
            model=client_model,
            dataloader=train_loader,
            device=device,
            local_epochs=config['training']['local_epochs'],
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        clients.append(client)
    
    logger.info(f"Initialized {len(clients)} clients")
    
    metrics_tracker = MetricsTracker()
    
    start_round = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint_metrics = server.load_checkpoint(args.resume)
        start_round = server.round
        if checkpoint_metrics:
            logger.info(f"Checkpoint metrics: {checkpoint_metrics}")
    
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*60)
    logger.info("Starting Federated Training")
    logger.info("="*60 + "\n")
    
    best_accuracy = 0.0
    
    for round_num in range(start_round, config['federated']['num_rounds']):
        logger.info(f"\n{'='*60}")
        logger.info(f"Round {round_num + 1}/{config['federated']['num_rounds']}")
        logger.info(f"{'='*60}")
        
        global_params = server.get_global_lora_parameters()
        
        for client in clients:
            client.update_model(global_params)
        
        logger.info("Training clients...")
        client_results = []
        for client in clients:
            result = client.train(global_round=round_num + 1)
            client_results.append(result)
            
            log_metrics(
                logger,
                round_num + 1,
                result['metrics'],
                prefix=f'Client {client.client_id}'
            )
        
        logger.info("Aggregating client parameters...")
        aggregated_params = server.aggregate(client_results)
        server.update_global_model(aggregated_params)
        
        logger.info("Evaluating global model...")
        eval_results = server.evaluate_global_model(test_loader, device=device)
        
        metrics = compute_metrics(eval_results['labels'], eval_results['predictions'])
        
        log_metrics(logger, round_num + 1, metrics, prefix='Global Model')
        
        metrics_tracker.update({
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'loss': eval_results['loss']
        })
        
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_checkpoint_path = checkpoint_dir / 'best_model.pth'
            server.save_checkpoint(str(best_checkpoint_path), metrics)
            logger.info(f"New best accuracy: {best_accuracy:.4f} - Saved checkpoint")
        
        if (round_num + 1) % config['logging']['save_every_n_rounds'] == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_round_{round_num + 1}.pth'
            server.save_checkpoint(str(checkpoint_path), metrics)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    
    final_checkpoint_path = checkpoint_dir / 'final_model.pth'
    server.save_checkpoint(str(final_checkpoint_path), metrics)
    logger.info(f"Final model saved: {final_checkpoint_path}")
    
    logger.info("\nFinal Evaluation:")
    final_results = server.evaluate_global_model(test_loader, device=device)
    final_metrics = compute_metrics(final_results['labels'], final_results['predictions'])
    
    for key, value in final_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    cm = calculate_confusion_matrix(
        final_results['labels'],
        final_results['predictions'],
        num_classes=config['model']['num_classes']
    )
    
    cm_path = checkpoint_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, class_names, save_path=str(cm_path))
    logger.info(f"Confusion matrix saved: {cm_path}")
    
    history = metrics_tracker.get_history()
    curves_path = checkpoint_dir / 'training_curves.png'
    plot_training_curves(history, save_path=str(curves_path))
    logger.info(f"Training curves saved: {curves_path}")
    
    logger.info(f"\nBest accuracy: {best_accuracy:.4f}")
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()
