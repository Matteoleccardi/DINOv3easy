import torch



def inference_dino(self, x: torch.Tensor) -> torch.Tensor:
    # Get the features organised as a 2D image with N channels
    # x is a 2d image (B, 3, K1, K2) where it should be that K1 = K2 == K
    # K must be multiple of 32
    # If K is the side of the input image, then the output feature map will be (B, 3, K/32, K/32)
    if len(x.shape) != 4 or x.shape[1] != 3:
        raise ValueError("Input tensor must be of shape (B, 3, K, K) with K being the image size.")
    out: torch.Tensor = self.dinov3_model.get_intermediate_layers(x)[0]
    features_2d_img_side = int(out.shape[1]**0.5)
    out = out.reshape(out.shape[0], features_2d_img_side, features_2d_img_side, out.shape[2])
    out = out.permute(0, 3, 1, 2).contiguous()
    return out

def extract_3d_features_with_2d_dino(self, x: torch.Tensor):
    # Maybe useful for later models
    """Expected shape: (B, 1, I, J, K) with K ranging along the axial direction (z axis)"""
    # Get the features of the 3D medical image with 1 color channel
    # using dino in a 2D manner only.
    #
    # Dino is trained on natural images found across the internet.
    # Of those, very few represent CT images, if any.
    # Of this very small subset, usually CT images are represented in 2D with axial slices
    # (cutting the torso with  aplane perpendicular to the feet-head axis)
    # So, it does not make much sense to also slice along the other axes.
    #
    # Slices are resamled at 1.24 x 1024 to obtain a finer feature map.
    # Axial slices are obtained by ranging through the last tensor axis (B, 1, I, J, K) where K -> Z axis
    if len(x.shape) != 5 or x.shape[1] != 1:
        raise ValueError("Input tensor must be of shape (B, 1, D, H, W) with D being the depth.")
    features = []
    for i in range(x.shape[-1]):
        slice_2d = x[:, :, :, :, i]
        features_2d = self.forward_dino(slice_2d)
        features.append(features_2d)
    features = torch.stack(features, dim=2)  # (B, C, 32, 32, K)
    return features

