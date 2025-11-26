import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from .lora import apply_lora_to_model, count_parameters
import logging

logger = logging.getLogger(__name__)


class RTFEModule(nn.Module):
    """
    Raw Traffic Feature Extraction module.
    
    Uses pretrained Vision Transformer with LoRA for efficient fine-tuning.
    
    From the paper:
    - Backbone: ViT-tiny/16
    - LoRA rank: 4, alpha: 8
    - Applied to attention and FFN layers
    """
    
    def __init__(self,
                 num_classes: int = 8,
                 backbone: str = "WinKawaks/vit-tiny-patch16-224",
                 rank: int = 4,
                 alpha: int = 8,
                 dropout: float = 0.0,
                 pretrained: bool = True):
        """
        Args:
            num_classes: Number of traffic classes
            backbone: HuggingFace model name
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: LoRA dropout
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.rank = rank
        self.alpha = alpha
        
        # Load pretrained ViT
        logger.info(f"Loading ViT model: {backbone}")
        self.vit = ViTForImageClassification.from_pretrained(
            backbone,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Apply LoRA
        logger.info(f"Applying LoRA (rank={rank}, alpha={alpha})")
        self.vit = apply_lora_to_model(
            self.vit,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=['query', 'key', 'value', 'dense', 'intermediate']
        )
        
        # Log parameter counts
        params = count_parameters(self.vit)
        logger.info(f"Model parameters:")
        logger.info(f"  Total: {params['total']:,}")
        logger.info(f"  Trainable: {params['trainable']:,} ({params['trainable_percent']:.2f}%)")
        logger.info(f"  Frozen: {params['frozen']:,}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT.
        
        Args:
            x: Input tensor (batch_size, 3, 224, 224)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        outputs = self.vit(x)
        return outputs.logits
    
    def get_lora_parameters(self) -> dict:
        """
        Extract only LoRA parameters (A and B matrices).
        
        Returns:
            Dictionary of LoRA parameter names and values
        """
        lora_params = {}
        for name, param in self.named_parameters():
            if 'lora' in name and param.requires_grad:
                lora_params[name] = param.data.clone()
        return lora_params
    
    def set_lora_parameters(self, lora_params: dict):
        """
        Set LoRA parameters from a dictionary.
        
        Args:
            lora_params: Dictionary of parameter names and values
        """
        state_dict = self.state_dict()
        state_dict.update(lora_params)
        self.load_state_dict(state_dict)
    
    def freeze_base_model(self):
        """Freeze all parameters except LoRA"""
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing RTFE Module...")
    
    # Create model
    model = RTFEModule(num_classes=8, rank=4, alpha=8)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(f"\nForward pass: {x.shape} -> {logits.shape}")
    
    # Test LoRA parameter extraction
    lora_params = model.get_lora_parameters()
    print(f"\nLoRA parameters: {len(lora_params)} tensors")
    
    total_lora_params = sum(p.numel() for p in lora_params.values())
    print(f"Total LoRA parameters: {total_lora_params:,}")
