import torch

# Model Output
# It is enough to do model.forward(x) as any other torch module.

# Features

def get_features(dino: torch.nn.Module, x: torch.Tensor, dino_output: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Get the features organised as a 2D image with N channels
    
    # Input

    `dino`: the dino model.
    
    `x` is a square 2d image (torch.Tensor) of shape (B, 3, K, K)
    K must be multiple of 32 if dino is convnext-based, of 16 if it is vit-based.

    x should be already standardized. Developers use the ImageNet statistics for this purpose:
        - IMAGENET_MEAN = (0.485, 0.456, 0.406)
        - IMAGENET_STD = (0.229, 0.224, 0.225)
    which can be imported from `dinov3easy.utils.constants`.

    `dino_output`: (bool) If True, give back also the output of the model as it is.

    # Output

    The feature map of the model.
    If K is the side of the input image, then the output feature map will be:
        - (B, C, K/32, K/32) if dino is convnext-based
        - (B, C, K/16, K/16) if dino is vit-based

    If dino_output is True, outputs a tuple of tensors, secodn tensor is the output of the model.
    """
    # Shape check
    if len(x.shape) != 4:
        raise ValueError("Input tensor must be of shape (B, 3, K, K) with K being the image size.")
    # Check if image is square
    if x.shape[2] != x.shape[3]:
        raise ValueError("Input image must be square.")
    # Check if K is a multiple of 16 or 32
    if "DinoVisionTransformer" in str(type(dino)):
        if x.shape[2] % 16 != 0:
            raise ValueError("Input image size must be a multiple of 16 for ViT-based models.")
    elif "ConvNeXt" in str(type(dino)):
        if x.shape[2] % 32 != 0:
            raise ValueError("Input image size must be a multiple of 32 for ConvNeXt-based models.")
    else:
        raise ValueError(f"Unknown model type: {type(dino)}")
    # Get the output features
    x = x.float()
    out: torch.Tensor = dino.get_intermediate_layers(x)[0]
    features_2d_img_side = int(out.shape[1]**0.5)
    out = out.reshape(out.shape[0], features_2d_img_side, features_2d_img_side, out.shape[2])
    out = out.permute(0, 3, 1, 2).contiguous()
    if not dino_output:
        return out
    else:
        return (out, dino(x))



 # Attention Map

def get_attention_map(dino: torch.nn.Module, x: torch.Tensor, remove_extra_tokens: bool = True) -> torch.Tensor:
    """Get the attention map of the model.
    
    # Input

    `dino`: the dino model. Must be a visual transformer (ViT).

    `x` is a square 2d image (torch.Tensor) of shape (B, 3, K, K)
    K must be multiple of 16.

    x should be already standardized. Developers use the ImageNet statistics for this purpose:
        - IMAGENET_MEAN = (0.485, 0.456, 0.406)
        - IMAGENET_STD = (0.229, 0.224, 0.225)
    which can be imported from `dinov3easy.utils.constants`.

    # Output

    The attention map of the model (the most informative one).
    
    shape: (num_tokens, num_tokens) or (num_patches, num_patches) if remove_extra_tokens is True.

    attn_map[i, j] mean (how much attention is given to patch j when processing patch i).

    Note that the Dino ViT has 5 extra tokens at the beginning that do not come from image patches
    (first one is the CLS token, the remaining ones I do not know).
    """
    # Shape check
    if len(x.shape) != 4:
        raise ValueError("Input tensor must be of shape (B, 3, K, K) with K being the image size.")
    # Check if image is square
    if x.shape[2] != x.shape[3]:
        raise ValueError("Input image must be square.")
    # Check if K is a multiple of 16
    if "DinoVisionTransformer" in str(type(dino)):
        if x.shape[2] % 16 != 0:
            raise ValueError("Input image size must be a multiple of 16 for ViT-based models.")
    else:
        raise ValueError(f"Unknown model type: {type(dino)}")
    # Get the attention map
    # This is possible only because I manually modified the Attention layer 
    # (dinov3 -> layers -> attention.py file)
    # so that, if the flag is true, attention is computed in the "canonical" way 
    # (with the attention matrix)
    # and not with the far more efficient flash attention that the model actually uses.
    x = x.float()
    dino.blocks[-1].attn.save_attention_map = True
    _ = dino(x)
    attn_map = dino.blocks[-1].attn.attn_map
    attn_map = attn_map.detach().cpu()
    attn_map = torch.mean(attn_map.squeeze(0), axis=0) # shape: (num_tokens, num_tokens)
    if remove_extra_tokens:
        attn_map = attn_map[5:, 5:] # shape: (num_patches, num_patches)
    dino.blocks[-1].attn.save_attention_map = False
    return attn_map

