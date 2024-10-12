import numpy as np
from tqdm import tqdm
from .tiled_image import TiledImage

class SegFlow:
    def __init__(self, tile_size=512, stride=256, average_weight=0.7, sum_weight=0.3, min_pixels=5):
        """
        Initialize the Segment class with parameters for image tiling.
        
        Parameters:
        - tile_size: Size of the tiles to extract from the image.
        - stride: Stride size for tiling the image.
        - average_weight: When choosing conflicting overlapping segments, how to weight the average
        - sum_weight: When choosing conflicting overlapping segments, how to weight the sum of pixels
        - min_pixels: The minimum number of pixels considered when combining overlaps
        """
        self.tile_size = tile_size
        self.stride = stride
        self.average_weight = 0.7
        self.sum_weight = 0.3
        self.min_pixels = 5

        self.image = None

        self.tiled_image = None


    def load_numpy_arrays(self, nuclear, membrane=None):
        """
        Load the image data from numpy arrays for nuclear and membrane channels.
        
        Parameters:
        - nuclear: Numpy array for the nuclear channel.
        - membrane: Optional numpy array for the membrane channel. If not provided, the nuclear channel will be duplicated.
        """
        if membrane is None:
            membrane = nuclear.copy()
        self.image = np.stack([nuclear, membrane], axis=-1)
        print(f"Loaded numpy arrays with shape: {self.image.shape}")

    def normalize_image(self):
        """
        Normalize each channel of the loaded image separately to have zero mean and unit variance.
        """
        if self.image is not None and len(self.image.shape) == 3:
            normalized_image = np.zeros_like(self.image, dtype=np.float32)
            for channel in range(self.image.shape[-1]):
                channel_data = self.image[:, :, channel]
                channel_mean = np.mean(channel_data)
                channel_std = np.std(channel_data)
                normalized_image[:, :, channel] = (channel_data - channel_mean) / channel_std
            self.image = normalized_image
            print("Normalized image shape:", self.image.shape)

    def extract_raw_tiles(self):
        return TiledImage.from_image(self.image, self.tile_size, self.stride, min_padding = self.stride // 2)
