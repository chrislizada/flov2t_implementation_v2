import torch
import torch.nn as nn
import math
from typing import Optional


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer.
    
    Implements the LoRA method from Hu et al. (2021):
    W = W0 + BA, where rank(BA) << rank(W0)
    
    From the FLoV2T paper:
    - Default rank r = 4
    - Default alpha α = 8
    - Scaling factor = α / r
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 rank: int = 4,
                 alpha: int = 8,
                 dropout: float = 0.0):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank of LoRA matrices (default: 4)
            alpha: Scaling parameter (default: 8)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        # A: (rank, in_features) - initialized with kaiming_uniform
        # B: (out_features, rank) - initialized with zeros
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA parameters"""
        # Initialize A with kaiming_uniform (same as nn.Linear)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B with zeros (so initially ΔW = 0)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute BA*x with scaling
        
        Args:
            x: Input tensor (..., in_features)
            
        Returns:
            Output tensor (..., out_features)
        """
        # x @ A^T @ B^T with dropout and scaling
        result = self.dropout(x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    Implements: h = W0*x + BA*x
    where W0 is frozen and only BA is trainable.
    """
    
    def __init__(self,
                 linear: nn.Linear,
                 rank: int = 4,
                 alpha: int = 8,
                 dropout: float = 0.0):
        """
        Args:
            linear: Original nn.Linear layer (will be frozen)
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: Dropout probability
        """
        super().__init__()
        
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: W0*x + BA*x
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.linear(x) + self.lora(x)
    
    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into the original linear layer.
        Useful for inference.
        
        Returns:
            New nn.Linear with merged weights
        """
        merged = nn.Linear(
            self.linear.in_features,
            self.linear.out_features,
            bias=self.linear.bias is not None
        )
        
        # Compute W = W0 + BA
        with torch.no_grad():
            delta_w = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            merged.weight.data = self.linear.weight.data + delta_w
            
            if self.linear.bias is not None:
                merged.bias.data = self.linear.bias.data
        
        return merged


def apply_lora_to_model(model: nn.Module,
                        rank: int = 4,
                        alpha: int = 8,
                        dropout: float = 0.0,
                        target_modules: Optional[list] = None) -> nn.Module:
    """
    Apply LoRA to specified modules in a model.
    
    For ViT, typically applied to:
    - Attention query, key, value projections
    - Attention output projection
    - MLP intermediate and output layers
    
    Args:
        model: PyTorch model
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout probability
        target_modules: List of module name patterns to apply LoRA to
        
    Returns:
        Model with LoRA applied
    """
    if target_modules is None:
        # Default targets for ViT
        target_modules = ['query', 'key', 'value', 'dense', 'intermediate']
    
    for name, module in model.named_modules():
        # Check if this module should have LoRA applied
        should_apply = any(target in name for target in target_modules)
        
        if should_apply:
            # Find and replace Linear layers
            for child_name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # Replace with LoRALinear
                    lora_linear = LoRALinear(
                        child,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout
                    )
                    setattr(module, child_name, lora_linear)
    
    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters in model.
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'trainable_percent': 100 * trainable_params / total_params if total_params > 0 else 0
    }


if __name__ == "__main__":
    # Test LoRA layer
    print("Testing LoRA implementation...")
    
    # Create a simple linear layer
    linear = nn.Linear(768, 768)
    print(f"Original linear layer: {sum(p.numel() for p in linear.parameters())} parameters")
    
    # Wrap with LoRA
    lora_linear = LoRALinear(linear, rank=4, alpha=8)
    
    # Count parameters
    params = count_parameters(lora_linear)
    print(f"\nLoRA Linear layer:")
    print(f"  Total parameters: {params['total']}")
    print(f"  Trainable parameters: {params['trainable']}")
    print(f"  Frozen parameters: {params['frozen']}")
    print(f"  Trainable percentage: {params['trainable_percent']:.2f}%")
    
    # Test forward pass
    x = torch.randn(8, 768)
    y = lora_linear(x)
    print(f"\nForward pass: {x.shape} -> {y.shape}")
