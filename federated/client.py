import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class FLoV2TClient:
    """
    Federated Learning Client for FLoV2T
    
    Performs local training with LoRA fine-tuning on client device.
    Only uploads LoRA parameters (A and B matrices) to reduce communication cost.
    
    From the paper (Section 3.3):
    - Local epochs: 1
    - Batch size: 32
    - Optimizer: AdamW
    - Learning rate: 1e-4
    - Weight decay: 0.01
    """
    
    def __init__(self,
                 client_id: int,
                 model: nn.Module,
                 dataloader: DataLoader,
                 device: str = 'cuda',
                 local_epochs: int = 1,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01):
        """
        Args:
            client_id: Unique client identifier
            model: RTFE model with LoRA
            dataloader: DataLoader for client's local data
            device: Device to train on ('cuda' or 'cpu')
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.local_epochs = local_epochs
        
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.num_samples = len(dataloader.dataset)
        
        logger.info(f"Client {client_id} initialized with {self.num_samples} samples")
    
    def train(self, global_round: int = 0) -> dict:
        """
        Perform local training
        
        Args:
            global_round: Current global training round
        
        Returns:
            Dictionary containing:
                - lora_params: LoRA parameters (A and B matrices)
                - num_samples: Number of training samples
                - metrics: Training metrics (loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            pbar = tqdm(self.dataloader, desc=f'Client {self.client_id} - Round {global_round} - Epoch {epoch+1}/{self.local_epochs}')
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                _, predicted = outputs.max(1)
                
                epoch_loss += loss.item()
                epoch_correct += predicted.eq(labels).sum().item()
                epoch_total += labels.size(0)
                
                pbar.set_postfix({
                    'loss': f'{epoch_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*epoch_correct/epoch_total:.2f}%'
                })
            
            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total
        
        avg_loss = total_loss / (len(self.dataloader) * self.local_epochs)
        accuracy = correct / total
        
        lora_params = self.model.get_lora_parameters()
        
        logger.info(f"Client {self.client_id} - Round {global_round}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        
        return {
            'lora_params': lora_params,
            'num_samples': self.num_samples,
            'metrics': {
                'loss': avg_loss,
                'accuracy': accuracy
            }
        }
    
    def update_model(self, lora_params: dict):
        """
        Update local model with global LoRA parameters
        
        Args:
            lora_params: Global LoRA parameters from server
        """
        self.model.set_lora_parameters(lora_params)
        logger.info(f"Client {self.client_id}: Model updated with global parameters")
    
    def evaluate(self) -> dict:
        """
        Evaluate model on local data
        
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                _, predicted = outputs.max(1)
                
                total_loss += loss.item()
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.dataloader)
        accuracy = correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels
        }
