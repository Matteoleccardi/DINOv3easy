import numpy
import torch
from sklearn.decomposition import PCA

def pca_of_features(features: torch.Tensor | numpy.ndarray, n_components=3) -> numpy.ndarray:
    """
    Perform PCA on the features extracted from DINOv3 model or any other model.

    # Input
        `features`: The features to perform PCA on. Accepted shapes are:
        - (T, C) where T is the number of tokens (patches) and C is the feature dimension (channels).
        - (B, T, C) where B is the batch size, T is the number of tokens (patches), and C is the feature dimension (channels).
        - (B, C, K1, K2) where B is the batch size, C is the feature dimension (channels), and K1, K2 are the spatial dimensions.
        - (B, C, K1, K2, K3) where B is the batch size, C is the feature dimension (channels), and K1, K2, K3 are the spatial dimensions.
    """
    # Convert to numpy array
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    # Deal with the accepted input sizes
    original_shape = features.shape
    if len(features.shape) == 2:
        # expected (T, C)
        # transform into (B, T, C) where B=1
        features = features[None, ...]
    elif len(features.shape) == 3:
        # expected (B, T, C) -> this is the target shape
        pass
    elif len(features.shape) == 4:
        # expected (B, C, K1, K2)
        # transform into (B, T, C) where T=K1*K2
        features = features.transpose(0, 2, 3, 1).reshape(features.shape[0], -1, features.shape[1])
    elif len(features.shape) == 5:
        # expected (B, C, K1, K2, K3)
        # transform into (B, T, C) where T=K1*K2*K3
        features = features.transpose(0, 2, 3, 4, 1).reshape(features.shape[0], -1, features.shape[1])
    else:
        raise ValueError("features must be a 2D, 3D, 4D, or 5D array. Read the docstring.")
    # Shape check
    B, T, C = features.shape
    if C < n_components:
        raise ValueError(f"Number of tokens {T} is less than the number of PCA components {n_components}.")
    # Perform PCA
    pca = PCA(n_components=n_components, whiten=True)
    pca_result = pca.fit_transform(features.reshape(B*T, C)) # pca_results shape: (n_in_batch*n_patches, components)
    pca_result = 1 / (1 + numpy.exp(-pca_result * 2.0))  # sigmoid
    # Reshape back to original shape except channel dim, which will be n_components
    if len(original_shape) == 2:
        pca_features = pca_result
    elif len(original_shape) == 3:
        pca_features = pca_result.reshape(B, T, n_components)
    elif len(original_shape) == 4:
        K1, K2 = original_shape[2:4]
        pca_features = pca_result.reshape(B, K1, K2, n_components)
        pca_features = pca_features.transpose(0, 3, 1, 2)  # back to (B, C, K1, K2)
    elif len(original_shape) == 5:
        K1, K2, K3 = original_shape[2:5]
        pca_features = pca_result.reshape(B, K1, K2, K3, n_components)
        pca_features = pca_features.transpose(0, 4, 1, 2, 3)  # back to (B, C, K1, K2, K3)

    return pca_features

