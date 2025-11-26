import torch
import torch.nn as nn
from typing import List, Dict
import logging
from .aggregation import rgpa_aggregate, fedavg_aggregate

logger = logging.getLogger(__name__)


class RGPAServer:
    """
    Federated Learning Server with Regularized Global Parameter Aggregation (RGPA)
    
    Aggregates client LoRA parameters using RGPA mechanism to mitigate
    non-IID data distribution effects.
    
    From the paper (Section 3.4):
    - Aggregation method: RGPA
    - Regularization coefficient λ: 0.1
    - Weight by samples: True (proportional to client data size)
    """
    
    def __init__(self,
                 global_model: nn.Module,
                 aggregation_method: str = 'rgpa',
                 lambda_reg: float = 0.1,
                 weight_by_samples: bool = True):
        """
        Args:
            global_model: Global RTFE model
            aggregation_method: 'rgpa' or 'fedavg'
            lambda_reg: Regularization coefficient for RGPA
            weight_by_samples: Whether to weight clients by number of samples
        """
        self.global_model = global_model
        self.aggregation_method = aggregation_method
        self.lambda_reg = lambda_reg
        self.weight_by_samples = weight_by_samples
        
        self.prev_global_params = None
        self.round = 0
        
        logger.info(f"Server initialized with {aggregation_method} aggregation")
        logger.info(f"Lambda regularization: {lambda_reg}")
        logger.info(f"Weight by samples: {weight_by_samples}")
    
    def aggregate(self, client_results: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client LoRA parameters
        
        Args:
            client_results: List of dictionaries from clients containing:
                - lora_params: LoRA parameters
                - num_samples: Number of training samples
                - metrics: Training metrics
        
        Returns:
            Aggregated global LoRA parameters
        """
        if not client_results:
            raise ValueError("No client results to aggregate")
        
        client_params = [result['lora_params'] for result in client_results]
        num_samples = [result['num_samples'] for result in client_results]
        
        if self.weight_by_samples:
            total_samples = sum(num_samples)
            client_weights = [n / total_samples for n in num_samples]
            logger.info(f"Client weights (by samples): {[f'{w:.3f}' for w in client_weights]}")
        else:
            num_clients = len(client_results)
            client_weights = [1.0 / num_clients] * num_clients
            logger.info(f"Client weights (uniform): {[f'{w:.3f}' for w in client_weights]}")
        
        if self.aggregation_method == 'rgpa':
            aggregated_params = rgpa_aggregate(
                client_params=client_params,
                client_weights=client_weights,
                prev_global_params=self.prev_global_params,
                lambda_reg=self.lambda_reg
            )
        elif self.aggregation_method == 'fedavg':
            aggregated_params = fedavg_aggregate(
                client_params=client_params,
                client_weights=client_weights
            )
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        current_params = self.global_model.get_lora_parameters()
        self.prev_global_params = {k: v.clone() for k, v in current_params.items()}
        
        return aggregated_params
    
    def update_global_model(self, aggregated_params: Dict[str, torch.Tensor]):
        """
        Update global model with aggregated LoRA parameters
        
        Args:
            aggregated_params: Aggregated LoRA parameters
        """
        self.global_model.set_lora_parameters(aggregated_params)
        self.round += 1
        logger.info(f"Global model updated (Round {self.round})")
    
    def get_global_model(self) -> nn.Module:
        """
        Get current global model
        
        Returns:
            Global model
        """
        return self.global_model
    
    def get_global_lora_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get current global LoRA parameters
        
        Returns:
            Global LoRA parameters
        """
        return self.global_model.get_lora_parameters()
    
    def save_checkpoint(self, path: str, metrics: Dict = None):
        """
        Save server checkpoint
        
        Args:
            path: Path to save checkpoint
            metrics: Optional metrics to save with checkpoint
        """
        checkpoint = {
            'round': self.round,
            'global_model_state_dict': self.global_model.state_dict(),
            'prev_global_params': self.prev_global_params,
            'aggregation_method': self.aggregation_method,
            'lambda_reg': self.lambda_reg,
            'metrics': metrics
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load server checkpoint
        
        Args:
            path: Path to checkpoint file
        
        Returns:
            Metrics from checkpoint (if available)
        """
        checkpoint = torch.load(path)
        
        self.global_model.load_state_dict(checkpoint['global_model_state_dict'])
        self.prev_global_params = checkpoint['prev_global_params']
        self.round = checkpoint['round']
        self.aggregation_method = checkpoint.get('aggregation_method', 'rgpa')
        self.lambda_reg = checkpoint.get('lambda_reg', 0.1)
        
        logger.info(f"Checkpoint loaded from {path} (Round {self.round})")
        
        return checkpoint.get('metrics', None)
    
    def evaluate_global_model(self, test_loader, device='cuda'):
        """
        Evaluate global model on test set
        
        Args:
            test_loader: DataLoader for test data
            device: Device to evaluate on
        
        Returns:
            Dictionary containing evaluation metrics
        """
        self.global_model.eval()
        self.global_model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = self.global_model(images)
                loss = criterion(outputs, labels)
                
                _, predicted = outputs.max(1)
                
                total_loss += loss.item()
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        
        logger.info(f"Global Model Evaluation - Round {self.round}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels
        }
