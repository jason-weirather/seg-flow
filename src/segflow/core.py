import numpy as np
from tqdm import tqdm
from .continuoussinglechannelimage import ContinuousSingleChannelImage
from .segmentationimage import SegmentationImage
from .tilecombiner import TileCombiner

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

        self.tile_combiner = TileCombiner(tile_size,average_weight,sum_weight,min_pixels)

        self.image = None
        self.image_padded = None
        self.pad_top = None
        self.pad_bottom = None
        self.pad_left = None
        self.pad_right = None
        self.tiles = None
        self.positions = None

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

    def pad_image(self):
        """
        Apply reflection padding to the image to ensure that tiling covers the entire image without leaving out regions.
        """
        if self.image is not None:
            height, width = self.image.shape[:2]
            pad_height_total = self.tile_size + ((self.tile_size - (height - self.tile_size) % self.stride) % self.stride)
            pad_width_total = self.tile_size + ((self.tile_size - (width - self.tile_size) % self.stride) % self.stride)
            
            self.pad_top = pad_height_total // 2
            self.pad_bottom = pad_height_total - self.pad_top
            self.pad_left = pad_width_total // 2
            self.pad_right = pad_width_total - self.pad_left
            
            self.image_padded = np.pad(
                self.image,
                ((self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0)),
                mode='reflect'
            )
            print(f"Padding applied - Top: {self.pad_top}, Bottom: {self.pad_bottom}, Left: {self.pad_left}, Right: {self.pad_right}")
            print(f"Padded image size: height={self.image_padded.shape[0]}, width={self.image_padded.shape[1]}")

    def extract_tiles(self):
        """
        Extract overlapping tiles from the padded image for segmentation.
        
        Returns:
        - tiles: Numpy array of extracted image tiles.
        - positions: List of positions where each tile starts in the original image.
        """
        if self.tiles is not None and self.positions is not None:
            return self.tiles, self.positions
        self.tiles, self.positions = _extract_tiles(self.image_padded, self.tile_size, self.stride)
        return self.tiles, self.positions

    def ingest_continuous_tiles(self, tiles):
        """
        Ingest the tiles and return the full image
        """
        full_image_shape = self.image_padded.shape[:2]
        reconstructed_image = np.zeros(full_image_shape, dtype=tiles[0].dtype)
        weight_matrix = np.zeros(full_image_shape, dtype=np.float32)

        # Create a weighting window to reduce edge effects
        window = np.outer(np.hanning(self.tile_size), np.hanning(self.tile_size))
        window = window / window.max()

        for tile, (row_start, col_start) in zip(tiles, self.positions):
            row_end = row_start + self.tile_size
            col_end = col_start + self.tile_size

            reconstructed_image[row_start:row_end, col_start:col_end] += tile * window
            weight_matrix[row_start:row_end, col_start:col_end] += window

        # Avoid division by zero
        weight_matrix[weight_matrix == 0] = 1

        reconstructed_image /= weight_matrix
        return ContinuousSingleChannelImage(self._crop_padded(reconstructed_image))

    def ingest_segmentation_tiles(self, segmentation_tiles, iou_threshold=0.5):
        """
        Ingest the segmented tiles and recombine them into a full segmentation mask, handling overlaps.
        """
        if self.positions is None:
            raise ValueError("Tiles must be extracted first")

        # Process tiles to get high-confidence central regions 
        # self.tile_size // 8 is 12.5% or 64 px for 512, 4 is 25% or 128px for 512
        segmentation_tiles = self.tile_combiner.high_confidence_tile_filter(segmentation_tiles, self.positions, margin_size_px = self.tile_size // 8)

        # Combine tiles using TileCombiner
        full_segmentation_mask = self.tile_combiner.combine_segmentation_tiles(segmentation_tiles, self.positions, self.image_padded.shape, iou_threshold)

        # Crop the padded segmentation mask
        final_segmentation_mask = self._crop_padded(full_segmentation_mask)

        return SegmentationImage(final_segmentation_mask)

    
    def _crop_padded(self,padded_image_to_crop):
        """
        Remove the padding from the full segmentation mask to obtain the final segmentation mask.
        """
        if self.image_padded is None:
            raise ValueError("Must have padded image first.")
        # Step 11: Remove padding to obtain the final segmentation mask

        # Crop the full segmentation mask to remove the padding
        cropped_segmentation_mask = padded_image_to_crop[self.pad_top:-self.pad_bottom, self.pad_left:-self.pad_right]

        # Handle edge cases where padding amounts are zero
        if self.pad_bottom == 0:
            cropped_segmentation_mask = padded_image_to_crop[self.pad_top:, self.pad_left:-self.pad_right]
        if self.pad_right == 0:
            cropped_segmentation_mask = cropped_segmentation_mask[:, self.pad_left:]

        # Print dimensions of the cropped mask
        print(f"Cropped segmentation mask size: height={cropped_segmentation_mask.shape[0]}, width={cropped_segmentation_mask.shape[1]}")
        return cropped_segmentation_mask

def _extract_tiles(image, tile_size, stride):
    """
    Extract overlapping tiles from the given image.
    
    Parameters:
    - image: Numpy array of the image to extract tiles from.
    - tile_size: Size of each tile.
    - stride: Stride for extracting tiles.
    
    Returns:
    - tiles: Numpy array of extracted tiles.
    - positions: List of positions where each tile starts in the image.
    """
    tiles = []
    positions = []
    height, width = image.shape[:2]
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            # Extract tile
            if len(image.shape) == 3:
                tile = image[y:y+tile_size, x:x+tile_size, :]
            else:
                tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((y, x))
    return np.array(tiles), positions
