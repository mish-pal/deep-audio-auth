import torch

def pad_or_truncate(tensor, max_length, dim=-1):
    """
    Pad or truncate a PyTorch tensor to a fixed size along a specific dimension.
    
    Mathematical Concept:
    For a tensor T, if T.shape[dim] > max_length, we take the slice T[..., :max_length].
    If T.shape[dim] < max_length, we pad the tensor with zeros along that dimension.
    
    Args:
        tensor (torch.Tensor): Input tensor.
        max_length (int): Target size along the target dimension.
        dim (int): Dimension along which to pad or truncate.
        
    Returns:
        torch.Tensor: Padded or truncated tensor.
    """
    length = tensor.shape[dim]
    
    if length > max_length:
        # Construct dynamic slicing for the target dimension
        slices = [slice(None)] * tensor.ndim
        slices[dim] = slice(0, max_length)
        return tensor[tuple(slices)]
        
    elif length < max_length:
        # Calculate padding amount
        pad_amount = max_length - length
        
        # PyTorch's F.pad takes padding pairs in reverse order of dimensions
        # starting from the last dimension.
        # For a 1D padding of the last dimension, pad shape is (0, pad_amount) if dim is -1.
        # Let's construct the pad list:
        pad_list = [0] * (2 * tensor.ndim)
        
        # Map the dimension index to positive and standard format
        if dim < 0:
            dim = tensor.ndim + dim
            
        # Reverse index mapping for F.pad
        rev_dim = tensor.ndim - 1 - dim
        pad_list[2 * rev_dim + 1] = pad_amount
        
        return torch.nn.functional.pad(tensor, tuple(pad_list))
        
    return tensor
