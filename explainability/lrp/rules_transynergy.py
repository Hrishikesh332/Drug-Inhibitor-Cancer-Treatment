import torch

def explain_transynergy(
    model: torch.nn.Module,
    inputs: torch.Tensor
    ) -> torch.Tensor:
    """Run LRP explanation on Transynergy model for regression task.
    
    Args:
        model: The Transynergy model
        inputs: Input tensor (drug features, cell line features)
        composite: Optional pre-created composite. If None, will create default one
        
    Returns:
        Tuple of relevance scores
    """
    # mass inputs thru model
    model.eval()
    with torch.no_grad():
        out = model(inputs)
    return