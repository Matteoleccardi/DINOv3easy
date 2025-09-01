import numpy
import matplotlib.pyplot as plt
from matplotlib import cm


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

        self.patch_size = int(image.shape[0] / attention_map.shape[0]**0.5) # assuming square image and square attention map

        self.attention_colormap = cm.get_cmap("Reds")

        self.fig, self.ax = plt.subplots()
        self.fig.tight_layout()
        self.fig.suptitle(self.suptitle)
        self.ax.axis('off')
        
        self.image_artist = None
        self.attn_map_artist = None
        self.patch_position_artist = None
        
        self._update_figure()
        # when axis is clicked anywhere inside, update the figure
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)


    def _update_figure(self, index_of_interest: int = 0):
        # Attention map
        attention_map_2d = self.attention_map[index_of_interest].reshape(
            int(self.attention_map.shape[0]**0.5), 
            int(self.attention_map.shape[0]**0.5)
        )
        attention_map_2d = (attention_map_2d - attention_map_2d.min()) / (attention_map_2d.max() - attention_map_2d.min() + 1e-5)
        attention_map_2d_img = self.attention_colormap(attention_map_2d) # from scalar to RGBA image
        attention_map_2d_img[..., -1] = (attention_map_2d)*0.75
        attention_map_2d_img = numpy.kron(attention_map_2d_img, numpy.ones((self.patch_size, self.patch_size, 1)))
        
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
            self.attn_map_artist = self.ax.imshow(attention_map_2d_img)
            #
            self.patch_position_artist = self.ax.scatter(patch_position[0], patch_position[1], color='red', s=10)
        else:
            # Only update the attention map and the patch indicator
            self.attn_map_artist.set_array(attention_map_2d_img)
            self.patch_position_artist.set_offsets(patch_position)
    
    def _on_click(self, event):
        if event.inaxes is not None and event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            # find the patch that was clicked
            patch_x = int(x // self.patch_size)
            patch_y = int(y // self.patch_size)
            # patches are stored as a sequence, so find the index corresponding to the correct patch
            index_of_interest = int(patch_y * (self.attention_map.shape[0]**0.5) + patch_x)
            self._update_figure(index_of_interest)
            self.fig.canvas.draw()

