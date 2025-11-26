import torch
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def rgpa_aggregate(client_params: List[Dict[str, torch.Tensor]],
                   client_weights: List[float],
                   prev_global_params: Dict[str, torch.Tensor] = None,
                   lambda_reg: float = 0.1) -> Dict[str, torch.Tensor]:
    """
    Regularized Global Parameter Aggregation (RGPA).
    
    Implements Algorithm from FLoV2T paper:
    1. Weighted averaging: Ā = Σ(w_k * A_k) / Σw_k (Eq. 7)
    2. Regularization: Ā' = Ā - λ * Σ(w_k * (Ā_prev - A_k)) (Eq. 8-9)
    
    The regularization term uses the PREVIOUS global model to prevent
    extreme updates and maintain stability under non-IID conditions.
    
    Args:
        client_params: List of client parameter dictionaries
        client_weights: List of client weights (should sum to 1)
        prev_global_params: Previous round's global parameters (for regularization)
        lambda_reg: Regularization coefficient (default: 0.1)
        
    Returns:
        Aggregated parameters
    """
    if not client_params:
        raise ValueError("No client parameters provided")
    
    # Normalize weights
    total_weight = sum(client_weights)
    if abs(total_weight - 1.0) > 1e-6:
        logger.warning(f"Client weights sum to {total_weight}, normalizing...")
        client_weights = [w / total_weight for w in client_weights]
    
    # Step 1: Weighted averaging
    aggregated_params = {}
    
    param_names = client_params[0].keys()
    
    for param_name in param_names:
        # Weighted sum: Σ(w_k * params_k)
        weighted_sum = sum(
            w * params[param_name].to(client_params[0][param_name].device)
            for w, params in zip(client_weights, client_params)
        )
        
        aggregated_params[param_name] = weighted_sum
    
    # Step 2: Regularization
    # Ā' = Ā - λ * Σ(w_k * (Ā_prev - A_k))
    # Uses previous global model to prevent extreme drift
    regularized_params = {}
    
    if prev_global_params is None:
        # First round or no regularization - just return weighted average
        logger.info("No previous global params, skipping regularization")
        return aggregated_params
    
    for param_name in param_names:
        avg_param = aggregated_params[param_name]
        prev_param = prev_global_params.get(param_name, avg_param)
        
        # Compute regularization term: λ * Σ(w_k * (Ā_prev - A_k))
        # This pulls the update towards the previous global model
        reg_term = sum(
            w * (prev_param.to(avg_param.device) - params[param_name].to(avg_param.device))
            for w, params in zip(client_weights, client_params)
        )
        
        regularized_params[param_name] = avg_param - lambda_reg * reg_term
    
    logger.debug(f"Aggregated {len(regularized_params)} parameters with λ={lambda_reg}")
    
    return regularized_params


def fedavg_aggregate(client_params: List[Dict[str, torch.Tensor]],
                     client_weights: List[float]) -> Dict[str, torch.Tensor]:
    """
    Standard FedAvg aggregation (for comparison).
    
    Simply: Ā = Σ(w_k * A_k)
    
    Args:
        client_params: List of client parameter dictionaries
        client_weights: List of client weights
        
    Returns:
        Aggregated parameters
    """
    # Normalize weights
    total_weight = sum(client_weights)
    client_weights = [w / total_weight for w in client_weights]
    
    aggregated_params = {}
    param_names = client_params[0].keys()
    
    for param_name in param_names:
        weighted_sum = sum(
            w * params[param_name].to(client_params[0][param_name].device)
            for w, params in zip(client_weights, client_params)
        )
        aggregated_params[param_name] = weighted_sum
    
    return aggregated_params


if __name__ == "__main__":
    # Test aggregation
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing RGPA aggregation...")
    
    # Create dummy parameters
    client1_params = {
        'lora_A': torch.randn(4, 768),
        'lora_B': torch.randn(768, 4)
    }
    
    client2_params = {
        'lora_A': torch.randn(4, 768),
        'lora_B': torch.randn(768, 4)
    }
    
    client3_params = {
        'lora_A': torch.randn(4, 768),
        'lora_B': torch.randn(768, 4)
    }
    
    client_params = [client1_params, client2_params, client3_params]
    client_weights = [0.4, 0.3, 0.3]
    
    # Test RGPA
    rgpa_result = rgpa_aggregate(client_params, client_weights, lambda_reg=0.1)
    print(f"RGPA result: {list(rgpa_result.keys())}")
    
    # Test FedAvg
    fedavg_result = fedavg_aggregate(client_params, client_weights)
    print(f"FedAvg result: {list(fedavg_result.keys())}")
    
    # Compare
    for key in rgpa_result.keys():
        diff = (rgpa_result[key] - fedavg_result[key]).abs().mean()
        print(f"Difference in {key}: {diff:.6f}")
