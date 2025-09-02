import numpy
import matplotlib.pyplot as plt
from matplotlib import cm

def get_similarity_map(features: numpy.ndarray, k1: int = 0, k2: int = 0) -> numpy.ndarray:
    """ 
    Compute the cosine similarity map of patch in position k1, k2 with all other patches

    # Input

    - `features`: A numpy array of shape (channels, K1, K2) representing the feature maps.
    - `k1`: The first spatial dimension index of the patch of interest.
    - `k2`: The second spatial dimension index of the patch of interest.

    # Output

    - A numpy array of shape (K1, K2) representing the cosine similarity map, with values in [-1;1].
    """
    # features must be of shape (channels, K1, K2)
    # Normalize features
    features = features / numpy.linalg.norm(features, axis=0, keepdims=True)
    # Get query vector to have same shape as features
    query = features[:, k1:k1+1, k2:k2+1]  # (channels, 1, 1)
    # Compute similarity = dot(query, features) / (norm(query) * norm(features))
    cosine_similarity = numpy.sum(features * query, axis=0) # shape: (K1, K2)
    return cosine_similarity

def get_similarity_map_discrete_image(similarity_map: numpy.ndarray):
    """ 
    Input a (N, N) similarity map with values in [-1;1] and outputs an RGBA image of shape (N, N, 4).
    """
    # Define similarity intervals (in [0;1] interval, monotone decreasing)
    intervals = [1.0, 0.95, 0.75, 0.60, 0.50, 0.0]
    # Define some RGBA colors, one for each similarity cathegory
    color_1 = numpy.array([1.0, 0.1, 0.1, 0.5])
    color_2 = numpy.array([1.0, 0.5, 0.1, 0.5])
    color_3 = numpy.array([1.0, 1.0, 0.1, 0.4])
    color_4 = numpy.array([0.5, 1.0, 0.1, 0.3])
    color_5 = numpy.array([0.0, 0.0, 0.0, 0.0])
    colors = [color_1, color_2, color_3, color_4, color_5]
    # Check similarity map shape
    assert len(similarity_map.shape) == 2
    assert similarity_map.shape[0] == similarity_map.shape[1]
    # Check similarity map intensities and normalize it
    similarity_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min() + 1e-5)
    similarity_map = numpy.clip(similarity_map, 0, 1)
    # Create image
    similarity_image = numpy.zeros((*similarity_map.shape, 4), dtype=numpy.float32)
    for int_s, int_e, int_c in zip(intervals[:-1], intervals[1:], colors):
        mask = (similarity_map <= int_s) & (similarity_map > int_e)
        similarity_image[mask] = int_c
    return similarity_image


class SimilarityMapInteractiveVisualizer():
    def __init__(self, image: numpy.ndarray, features: numpy.ndarray, suptitle: str = "Cosine Similarity Map"):
        """
        # Input

        - `image`: (numpy.ndarray) of shape (K, K, 3)
        - `features`: (numpy.ndarray) of shape (C, F, F) where F = K/16 if ViT, F = K/32 if ConvNeXt 
        - `suptitle`: (str) title for the plot
        """
        self.image = image
        self.features = features
        self.features /= numpy.linalg.norm(self.features, axis=0, keepdims=True)  # normalize to unit norm (versors in C dimensions along axis 0)
        self.suptitle = suptitle

        self.patch_size = int(image.shape[0] / features.shape[-1]) # assuming square image and square features

        self.attention_colormap = cm.get_cmap("Reds")

        self.fig, axes = plt.subplots(1, 2)
        self.fig.tight_layout()
        self.fig.suptitle(self.suptitle)

        # Continuous Similarity
        self.cax = axes[0]
        self.cax.axis('off')
        
        self.c_image_artist = None
        self.c_similarity_artist = None
        self.c_patch_position_artist = None

        # Discrete Similarity
        self.dax = axes[1]
        self.dax.axis('off')

        self.d_image_artist = None
        self.d_similarity_artist = None
        self.d_patch_position_artist = None

        self.patch_position_buffer = (-1, -1)

        self._update_figure()
        # when axis is clicked anywhere inside, update the figure
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)


    def _update_figure(self, patch_pos: tuple[int, int] = (0, 0)):
        if patch_pos[0] == self.patch_position_buffer[0] and patch_pos[1] == self.patch_position_buffer[1]:
            return
        self.patch_position_buffer = patch_pos
        # Similarity Map
        similarity_map = get_similarity_map(self.features, patch_pos[1], patch_pos[0])
        similarity_map_discrete = get_similarity_map_discrete_image(similarity_map)
        # - upsample both
        kernel = numpy.ones((self.patch_size, self.patch_size))
        similarity_map = numpy.kron(similarity_map, kernel)
        kernel = kernel[..., None]
        similarity_map_discrete = numpy.kron(similarity_map_discrete, kernel)
        # Patch position in image
        patch_x, patch_y = patch_pos
        patch_position = ((patch_x+0.5) * self.patch_size, (patch_y+0.5) * self.patch_size)

        if self.c_image_artist is None:
            # first initialization
            image_min = self.image.min()
            image_max = self.image.max()
            image_show = (self.image - image_min) / (image_max - image_min + 1e-5)
            self.c_image_artist = self.cax.imshow(image_show, cmap='gray')
            self.d_image_artist = self.dax.imshow(image_show, cmap='gray')
            #
            self.c_similarity_artist = self.cax.imshow(similarity_map, cmap="jet", alpha=0.5)
            self.d_similarity_artist = self.dax.imshow(similarity_map_discrete)
            #
            self.c_patch_position_artist = self.cax.scatter(patch_position[0], patch_position[1], color='red', s=10)
            self.d_patch_position_artist = self.dax.scatter(patch_position[0], patch_position[1], color='red', s=10)
        else:
            # Only update the attention map and the patch indicator
            self.c_similarity_artist.set_array(similarity_map)
            self.d_similarity_artist.set_array(similarity_map_discrete)
            self.c_patch_position_artist.set_offsets(patch_position)
            self.d_patch_position_artist.set_offsets(patch_position)
    
    def _on_click(self, event):
        if event.inaxes is not None and (event.inaxes == self.cax or event.inaxes == self.dax):
            x, y = int(event.xdata), int(event.ydata)
            # find the patch that was clicked
            patch_x = int(x // self.patch_size)
            patch_y = int(y // self.patch_size)
            # update
            self._update_figure((patch_x, patch_y))
            self.fig.canvas.draw()

