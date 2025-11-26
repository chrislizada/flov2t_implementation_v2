import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Dict, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class FederatedDataLoader:
    """
    Creates federated data splits for IID and non-IID scenarios.
    
    Implements the data partitioning strategy from the FLoV2T paper:
    - IID: Each client has all classes but imbalanced quantities
    - Non-IID: Each client has different subsets of classes
    """
    
    def __init__(self,
                 dataset,
                 num_clients: int,
                 batch_size: int = 32,
                 distribution: str = 'iid',
                 non_iid_config: Dict = None,
                 num_workers: int = 4,
                 seed: int = 42):
        """
        Args:
            dataset: PyTorch Dataset object
            num_clients: Number of federated clients
            batch_size: Batch size for training
            distribution: 'iid' or 'non_iid'
            non_iid_config: Configuration for non-IID split
            num_workers: Number of data loading workers
            seed: Random seed
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.distribution = distribution
        self.non_iid_config = non_iid_config
        self.num_workers = num_workers
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.client_indices = self._create_splits()
    
    def _get_class_indices(self) -> Dict[int, List[int]]:
        """Get indices for each class"""
        class_indices = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            class_indices[label].append(idx)
        
        return class_indices
    
    def _create_iid_split(self) -> List[List[int]]:
        """
        Create IID split: Each client has all classes but imbalanced.
        
        According to the paper (Table 1), samples are distributed
        proportionally to maintain class imbalance.
        """
        class_indices = self._get_class_indices()
        client_indices = [[] for _ in range(self.num_clients)]
        
        # For each class, split samples across clients
        for class_id, indices in class_indices.items():
            # Shuffle indices
            np.random.shuffle(indices)
            
            # Split into num_clients parts (not necessarily equal)
            split_points = np.linspace(0, len(indices), 
                                      self.num_clients + 1, 
                                      dtype=int)
            
            for client_id in range(self.num_clients):
                start = split_points[client_id]
                end = split_points[client_id + 1]
                client_indices[client_id].extend(indices[start:end])
        
        # Shuffle each client's data
        for indices in client_indices:
            np.random.shuffle(indices)
        
        # Log distribution
        self._log_distribution(client_indices, "IID")
        
        return client_indices
    
    def _create_non_iid_split(self) -> List[List[int]]:
        """
        Create non-IID split: Each client has different classes.
        
        According to the paper (Table 2):
        - 3 clients:
            C1: Botnet, DoS-Slowloris, Web-SQL
            C2: DoS-Goldeneye, SSH-BruteForce
            C3: Web-XSS, DoS-Hulk, Web-Bruteforce
        - 5 clients:
            C1: Botnet, Web-BruteForce
            C2: DoS-Goldeneye, DoS-Hulk
            C3: Web-SQL, Web-XSS
            C4: DoS-Slowloris
            C5: SSH-BruteForce
        """
        if self.non_iid_config is None or str(self.num_clients) not in self.non_iid_config:
            raise ValueError(f"Non-IID configuration not provided for {self.num_clients} clients")
        
        class_indices = self._get_class_indices()
        client_indices = [[] for _ in range(self.num_clients)]
        
        config = self.non_iid_config[str(self.num_clients)]
        
        # Map category names to indices
        if hasattr(self.dataset, 'class_to_idx'):
            class_to_idx = self.dataset.class_to_idx
        else:
            # Default mapping for CICIDS2017
            categories = ["Botnet", "DoS-Slowloris", "DoS-Goldeneye", "DoS-Hulk",
                         "SSH-BruteForce", "Web-SQL", "Web-XSS", "Web-Bruteforce"]
            class_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        
        # Assign classes to clients
        for client_id in range(self.num_clients):
            client_key = f"client_{client_id}"
            
            if client_key in config:
                assigned_classes = config[client_key]
                
                for class_name in assigned_classes:
                    if class_name in class_to_idx:
                        class_id = class_to_idx[class_name]
                        if class_id in class_indices:
                            client_indices[client_id].extend(class_indices[class_id])
                        else:
                            logger.warning(f"No samples for class {class_name}")
                    else:
                        logger.warning(f"Unknown class name: {class_name}")
        
        # Shuffle each client's data
        for indices in client_indices:
            np.random.shuffle(indices)
        
        # Log distribution
        self._log_distribution(client_indices, "Non-IID")
        
        return client_indices
    
    def _create_splits(self) -> List[List[int]]:
        """Create data splits based on distribution type"""
        if self.distribution == 'iid':
            return self._create_iid_split()
        elif self.distribution == 'non_iid':
            return self._create_non_iid_split()
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
    
    def _log_distribution(self, client_indices: List[List[int]], dist_type: str):
        """Log the data distribution"""
        logger.info(f"\n{dist_type} Data Distribution ({self.num_clients} clients):")
        
        for client_id, indices in enumerate(client_indices):
            # Count samples per class
            class_counts = defaultdict(int)
            for idx in indices:
                _, label = self.dataset[idx]
                class_counts[label] += 1
            
            logger.info(f"  Client {client_id}: {len(indices)} samples")
            for class_id, count in sorted(class_counts.items()):
                if hasattr(self.dataset, 'attack_categories'):
                    class_name = self.dataset.attack_categories[class_id]
                else:
                    class_name = f"Class_{class_id}"
                logger.info(f"    {class_name}: {count}")
    
    def get_client_loaders(self) -> List[DataLoader]:
        """
        Create DataLoader for each client.
        
        Returns:
            List of DataLoader objects, one per client
        """
        client_loaders = []
        
        for client_id, indices in enumerate(self.client_indices):
            subset = Subset(self.dataset, indices)
            
            loader = DataLoader(
                subset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            client_loaders.append(loader)
            logger.info(f"Created DataLoader for Client {client_id}: "
                       f"{len(subset)} samples, "
                       f"{len(loader)} batches")
        
        return client_loaders
    
    def get_client_weights(self) -> List[float]:
        """
        Get client weights based on number of samples.
        Used for weighted aggregation.
        
        Returns:
            List of weights (normalized to sum to 1)
        """
        sample_counts = [len(indices) for indices in self.client_indices]
        total_samples = sum(sample_counts)
        weights = [count / total_samples for count in sample_counts]
        
        logger.info(f"Client weights: {weights}")
        return weights


if __name__ == "__main__":
    # Test federated data loader
    logging.basicConfig(level=logging.INFO)
    
    print("FederatedDataLoader test")
    print("This requires a properly initialized dataset")
