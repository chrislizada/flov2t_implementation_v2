from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """Configuration for FLoV2T model"""
    
    num_classes: int = 8
    backbone: str = "WinKawaks/vit-tiny-patch16-224"
    pretrained: bool = True
    
    lora_rank: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    
    image_size: int = 224
    patch_size: int = 16
    channels: int = 3
    
    @property
    def num_patches(self) -> int:
        """Calculate number of patches"""
        return (self.image_size // self.patch_size) ** 2
    
    def __post_init__(self):
        assert self.image_size % self.patch_size == 0, \
            f"Image size {self.image_size} must be divisible by patch size {self.patch_size}"
        assert self.num_patches == 196, \
            f"Expected 196 patches (14x14), got {self.num_patches}"
