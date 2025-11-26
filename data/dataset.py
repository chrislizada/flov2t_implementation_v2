import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import pickle
import logging

logger = logging.getLogger(__name__)


class TrafficDataset(Dataset):
    """Base dataset class for traffic images"""
    
    def __init__(self, 
                 images: List[np.ndarray],
                 labels: List[int],
                 transform=None):
        """
        Args:
            images: List of traffic images (224x224x3)
            labels: List of labels
            transform: Optional torchvision transforms
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        
        assert len(images) == len(labels), "Images and labels must have same length"
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert HWC to CHW format
        image = np.transpose(image, (2, 0, 1))
        
        # Convert to tensor
        image = torch.from_numpy(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CICIDS2017Dataset(Dataset):
    """
    Dataset for CICIDS2017 processed traffic images.
    
    Expected directory structure:
    root_dir/
        train/
            class_0/
                flow_0.npy
                flow_1.npy
                ...
            class_1/
            ...
        test/
            class_0/
            ...
    """
    
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 attack_categories: List[str] = None,
                 transform=None):
        """
        Args:
            root_dir: Root directory containing processed data
            split: 'train' or 'test'
            attack_categories: List of attack category names
            transform: Optional transforms
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        if attack_categories is None:
            attack_categories = [
                "Botnet", "DoS-Slowloris", "DoS-Goldeneye", "DoS-Hulk",
                "SSH-BruteForce", "Web-SQL", "Web-XSS", "Web-Bruteforce"
            ]
        
        self.attack_categories = attack_categories
        self.class_to_idx = {cat: idx for idx, cat in enumerate(attack_categories)}
        
        self.images = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
        """Load processed images from disk"""
        data_dir = self.root_dir / self.split
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = data_dir / class_name
            
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Load all .npy files
            image_files = list(class_dir.glob("*.npy"))
            
            for img_file in image_files:
                try:
                    image = np.load(img_file)
                    self.images.append(image)
                    self.labels.append(class_idx)
                except Exception as e:
                    logger.error(f"Error loading {img_file}: {e}")
        
        logger.info(f"Loaded {len(self.images)} images for {self.split} split")
        
        # Log class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        for cls_idx, count in zip(unique, counts):
            logger.info(f"  {self.attack_categories[cls_idx]}: {count} samples")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0
        
        # Convert HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        # Convert to tensor
        image = torch.from_numpy(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """Get distribution of classes in dataset"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return {
            self.attack_categories[idx]: count 
            for idx, count in zip(unique, counts)
        }


def save_processed_dataset(images: List[np.ndarray],
                          labels: List[int],
                          output_dir: str,
                          attack_categories: List[str],
                          split: str = 'train'):
    """
    Save processed images to disk in class-organized structure.
    
    Args:
        images: List of processed traffic images
        labels: List of labels
        output_dir: Output directory
        attack_categories: List of category names
        split: 'train' or 'test'
    """
    output_path = Path(output_dir) / split
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create class directories
    for category in attack_categories:
        (output_path / category).mkdir(exist_ok=True)
    
    # Save images
    class_counters = {i: 0 for i in range(len(attack_categories))}
    
    for image, label in zip(images, labels):
        class_name = attack_categories[label]
        counter = class_counters[label]
        
        filename = f"flow_{counter:06d}.npy"
        filepath = output_path / class_name / filename
        
        np.save(filepath, image)
        class_counters[label] += 1
    
    logger.info(f"Saved {len(images)} images to {output_path}")
    for label, count in class_counters.items():
        logger.info(f"  {attack_categories[label]}: {count} samples")


if __name__ == "__main__":
    # Test dataset loading
    logging.basicConfig(level=logging.INFO)
    
    print("CICIDS2017Dataset test")
    print("Expected directory structure:")
    print("  root_dir/train/Botnet/flow_000000.npy")
    print("  root_dir/train/DoS-Slowloris/flow_000000.npy")
    print("  ...")
