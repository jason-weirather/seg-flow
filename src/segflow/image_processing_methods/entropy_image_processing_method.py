from .generic_image_processing_method import GenericImageProcessingMethod
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
import cv2
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

class EntropyImageProcessingMethod(GenericImageProcessingMethod):
    def __init__(self, image_mpp, entropy_window_size_px=50, downscale_factor=1):
        """
        Initialize the EntropyImageProcessingMethod class with parameters for the segmentation approach.

        Parameters:
        - image_mpp: Microns per pixel for scaling during segmentation.
        - entropy_window_size_px: Window size for entropy calculation in pixels.
        - downscale_factor: Factor by which to downscale the image for entropy calculation.
        """
        super().__init__(image_mpp)
        self.entropy_window_size_px = entropy_window_size_px
        self.downscale_factor = downscale_factor
        # Initialize min and max values; these will be set during segmentation
        self.min_value = None
        self.max_value = None

    def run_image_processing(self, tiles, batch_size=64):
        """
        Perform entropy-based segmentation on the tiles.

        Parameters:
        - tiles: A batch of image tiles to segment.

        Returns:
        - segmentation_tiles: Numpy array of entropy maps for each tile.
        """
        # Calculate min and max values across all tiles for normalization
        self.min_value = tiles[:, :, :, 0].min()
        self.max_value = tiles[:, :, :, 0].max()
        
        # Use multiprocessing to process tiles
        processed_tiles = self.process_tiles_multiprocessing(tiles)
        return processed_tiles

    def process_tiles_multiprocessing(self, tiles):
        """
        Process tiles using multiprocessing to calculate entropy features.

        Parameters:
        - tiles: Array of image tiles.

        Returns:
        - results: Numpy array of entropy maps for each tile.
        """
        num_workers = cpu_count()
        args = [(idx, tile) for idx, tile in enumerate(tiles)]
        with Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(self.calculate_entropy_wrapper, args),
                    total=len(tiles),
                    desc="Processing tiles"
                )
            )
        return np.array(results)

    def calculate_entropy_wrapper(self, args):
        idx, tile = args
        entropy_map = self.calculate_entropy_features(tile)
        return entropy_map

    def calculate_entropy_features(self, tile):
        """
        Calculates the entropy features for a given tile.

        Parameters:
        - tile: The input image tile.

        Returns:
        - entropy_map: The calculated entropy map for the tile.
        """
        # Extract the first channel (assuming it's the nuclear/DAPI channel)
        dapi_channel = tile[:, :, 0]

        # Downscale if necessary
        if self.downscale_factor > 1:
            new_width = dapi_channel.shape[1] // self.downscale_factor
            new_height = dapi_channel.shape[0] // self.downscale_factor
            dapi_channel = cv2.resize(
                dapi_channel,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )

        # Check if the tile contains information
        if np.all(dapi_channel == 0):
            entropy_map = np.zeros_like(dapi_channel)
        else:
            # Rescale the DAPI channel using the global min and max values
            dapi_channel_rescaled = (dapi_channel - self.min_value) / (self.max_value - self.min_value)
            dapi_channel_rescaled = np.clip(dapi_channel_rescaled, 0, 1)  # Ensure values are in the range [0, 1]

            # Convert to uint8
            dapi_channel_ubyte = img_as_ubyte(dapi_channel_rescaled)

            # Calculate entropy
            entropy_radius = self.entropy_window_size_px // (2 * self.downscale_factor)
            entropy_map = entropy(dapi_channel_ubyte, disk(entropy_radius))

        # Upscale back to original size if downscaled
        if self.downscale_factor > 1:
            entropy_map = cv2.resize(
                entropy_map,
                (tile.shape[1], tile.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        return entropy_map
