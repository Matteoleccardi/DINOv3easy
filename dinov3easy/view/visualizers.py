import numpy
import matplotlib.pyplot as plt


# In this file are declared the clkasses used to interactively (or not)
# visualize stuff.

class AttentionMapInteractiveVisualizer():
    def __init__(self, image: numpy.ndarray, attention_map: numpy.ndarray, suptitle: str = "Attention Map"):
        """
        # Input

        - `image`: (numpy.ndarray) of shape (K, K, 3)
        - `attention_map`: (numpy.ndarray) of shape (N, N)
        - `suptitle`: (str) title for the plot
        """
        self.image = image
        self.attention_map = attention_map
        self.suptitle = suptitle

        self.patch_size = int(image.shape[0] // attention_map.shape[0]) # assuming square image and square attention map

        self.fig, self.ax = plt.subplots()
        self.fig.suptitle(self.suptitle)

        self.image_artist = None
        self.attn_map_artist = None
        self.patch_position_artist = None
        
        self._update_figure()
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        self.fig.tight_layout()
        self.fig.show()

    def _update_figure(self, index_of_interest: int = 0):
        # Attention map
        attention_map_2d = self.attention_map[index_of_interest].reshape(
            int(self.attention_map.shape[0]**0.5), 
            int(self.attention_map.shape[0]**0.5)
        )
        attention_map_2d = (attention_map_2d - attention_map_2d.min()) / (attention_map_2d.max() - attention_map_2d.min() + 1e-5)
        attention_map_2d = numpy.kron(attention_map_2d, numpy.ones((self.patch_size, self.patch_size)))
        # Patch position in 2D grid
        patch_x = index_of_interest % (self.attention_map.shape[0]**0.5)
        patch_y = index_of_interest // (self.attention_map.shape[0]**0.5)
        patch_position = ((patch_x+0.5) * self.patch_size, (patch_y+0.5) * self.patch_size)

        if self.image_artist is None:
            # first initialization
            image_min = self.image.min()
            image_max = self.image.max()
            image_show = (self.image - image_min) / (image_max - image_min + 1e-5)
            self.image_artist = self.ax.imshow(image_show, cmap='gray')
            #
            self.attn_map_artist = self.ax.imshow(attention_map_2d, cmap='jet', alpha=0.5)
            #
            self.patch_position_artist = self.ax.scatter(patch_position[0], patch_position[1], color='red', s=10)
        else:
            # Only update the attention map and the patch indicator
            self.attn_map_artist.set_array(attention_map_2d)
            self.patch_position_artist.set_offsets(patch_position)
    
    def _on_click(self, event):
        if event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            # find the patch that was clicked
            patch_x = int(x // self.patch_size)
            patch_y = int(y // self.patch_size)
            # patches are stored as a sequence, so find the index corresponding to the correct patch
            index_of_interest = patch_y * (self.attention_map.shape[0]**0.5) + patch_x
            self._update_figure(index_of_interest)

