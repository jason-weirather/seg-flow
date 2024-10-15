import numpy as np
from scipy.ndimage import gaussian_filter

from ..full_image import ContinuousSingleChannelImage

class TiledImage(np.ndarray):
    """
    TiledImage is assumed to be an image made up of continuous intensity data on 1 or more channels

    Methods of this class may be overwritten to support other image types.
    """
    def __new__(cls, input_image):
        """
        Create a new TiledImage instance by copying an existing TiledImage.

        Parameters:
        - input_image: TiledImage instance to copy.
        
        Returns:
        - TiledImage instance.
        """
        if not isinstance(input_image, TiledImage):
            raise ValueError("input_image must be an instance of TiledImage. Use the class methods to create a new instance from an untiled image or a tiled array.")
        # If input_image is a TiledImage, create a copy
        obj = input_image.view(cls)
        return obj

    @classmethod
    def from_image(cls, input_image, tile_size, stride, min_padding):
        """
        Factory method to create a TiledImage from a fresh untiled image.

        Parameters:
        - input_image: Numpy array of the original image (height, width) or (height, width, m_channels).
        - tile_size: Size of each tile.
        - stride: Stride size for tiling the image.
        - min_padding: Minimum padding to add to ensure complete coverage.
        
        Returns:
        - TiledImage instance.

        Note: This method also sets the padding properties (pad_top, pad_bottom, pad_left, pad_right).
        """
        tiles, positions, pad_top, pad_bottom, pad_left, pad_right = cls._create_tiled_image(input_image, tile_size, stride, min_padding)
        obj = tiles.view(cls)
        obj.pad_top = pad_top
        obj.pad_bottom = pad_bottom
        obj.pad_left = pad_left
        obj.pad_right = pad_right
        obj.padded_shape = (
            input_image.shape[0] + obj.pad_top + obj.pad_bottom,
            input_image.shape[1] + obj.pad_left + obj.pad_right
        )
        obj.positions = positions
        obj.original_shape = input_image.shape
        obj.tile_size = tile_size
        return obj

    @classmethod
    def from_tiled_array(cls, tiled_array, positions, original_shape, pad_top, pad_bottom, pad_left, pad_right):
        """
        Factory method to create a TiledImage from an existing tiled array.

        Parameters:
        - tiled_array: Numpy array representing the tiled image (n_tiles, height, width, m_channels).
        - positions: List of positions where each tile starts in the padded image.
        - original_image_shape: The size (h,w) of the original unpadded image.
        - pad_top: Amount of padding added to the top of the original image.
        - pad_bottom: Amount of padding added to the bottom of the original image.
        - pad_left: Amount of padding added to the left of the original image.
        - pad_right: Amount of padding added to the right of the original image.

        Returns:
        - TiledImage instance.
        """
        obj = tiled_array.view(cls)
        obj.positions = positions
        obj.original_shape = original_shape
        obj.pad_top = pad_top
        obj.pad_bottom = pad_bottom
        obj.pad_left = pad_left
        obj.pad_right = pad_right
        obj.tile_size = tiled_array.shape[1]  # Assuming all tiles have the same height and width

        # Find the furthest extent in both dimensions based on positions
        max_y = obj.original_shape[0] + pad_top + pad_bottom
        max_x = obj.original_shape[1] + pad_left + pad_right

        if obj.stride == 0:
            raise ValueError("Calculated stride is zero. Check the positions provided.")

        # Padded shape is now determined by the max extent of the tiles
        obj.padded_shape = (
            max_y,
            max_x
        )

        return obj


    @property
    def padding(self):
        """
        Return the padding information as a dictionary.
        
        Returns:
        - Dictionary with pad_top, pad_bottom, pad_left, pad_right.
        """
        return {
            'top': self.pad_top,
            'bottom': self.pad_bottom,
            'left': self.pad_left,
            'right': self.pad_right
        }

    @staticmethod
    def _create_tiled_image(input_image, tile_size, stride, min_padding):
        """
        Pad the input image and extract tiles from it.

        Parameters:
        - input_image: Numpy array of the original image.
        - tile_size: Size of each tile.
        - stride: Stride size for tiling the image.
        - min_padding: Minimum padding to add to ensure complete coverage.
        
        Returns:
        - tiles: Numpy array of extracted tiles.
        - positions: List of positions where each tile starts in the padded image.
        - pad_top: Total padding added to the top of the image.
        - pad_bottom: Total padding added to the bottom of the image.
        - pad_left: Total padding added to the left of the image.
        - pad_right: Total padding added to the right of the image.
        """
        # Step 1: Add minimum padding as a margin all around the image
        if input_image.ndim == 3:
            padding = ((min_padding, min_padding), (min_padding, min_padding), (0, 0))
        else:
            padding = ((min_padding, min_padding), (min_padding, min_padding))
        image_padded = np.pad(input_image, padding, mode='reflect')

        # Step 2: Calculate additional padding to ensure tiling fits properly
        height, width = image_padded.shape[:2]
        pad_height_total = (tile_size - (height - tile_size) % stride) % stride
        pad_width_total = (tile_size - (width - tile_size) % stride) % stride

        # Step 3: Split the additional padding as evenly as possible
        pad_top_extra = pad_height_total // 2
        pad_bottom_extra = pad_height_total - pad_top_extra
        pad_left_extra = pad_width_total // 2
        pad_right_extra = pad_width_total - pad_left_extra

        # Step 4: Apply the additional padding
        if input_image.ndim == 3:
            padding = ((pad_top_extra, pad_bottom_extra), (pad_left_extra, pad_right_extra), (0, 0))
        else:
            padding = ((pad_top_extra, pad_bottom_extra), (pad_left_extra, pad_right_extra))
        image_padded = np.pad(image_padded, padding, mode='reflect')

        # Total padding
        pad_top = min_padding + pad_top_extra
        pad_bottom = min_padding + pad_bottom_extra
        pad_left = min_padding + pad_left_extra
        pad_right = min_padding + pad_right_extra

        # Step 5: Extract tiles from the padded image
        tiles, positions = TiledImage._extract_tiles(image_padded, tile_size, stride)

        return tiles, positions, pad_top, pad_bottom, pad_left, pad_right

    @staticmethod
    def _extract_tiles(image, tile_size, stride):
        """
        Extract overlapping tiles from the given image.

        Parameters:
        - image: Numpy array of the padded image to extract tiles from.
        - tile_size: Size of each tile.
        - stride: Stride for extracting tiles.
        
        Returns:
        - tiles: Numpy array of extracted tiles.
        - positions: List of positions where each tile starts in the padded image.
        """
        tiles = []
        positions = []
        height, width = image.shape[:2]
        for y in range(0, height - tile_size + 1, stride):
            for x in range(0, width - tile_size + 1, stride):
                # Extract tile
                if image.ndim == 3:
                    tile = image[y:y+tile_size, x:x+tile_size, :]
                else:
                    tile = image[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
                positions.append((y, x))
        return np.array(tiles), positions

    def __array_finalize__(self, obj):
        if obj is None: return
        self.tile_size = getattr(obj, 'tile_size', None)
        self.stride = getattr(obj, 'stride', None)
        self.min_padding = getattr(obj, 'min_padding', None)
        self.original_shape = getattr(obj, 'original_shape', None)
        self.padded_shape = getattr(obj, 'padded_shape', None)
        self.positions = getattr(obj, 'positions', None)
        self.pad_top = getattr(obj, 'pad_top', None)
        self.pad_bottom = getattr(obj, 'pad_bottom', None)
        self.pad_left = getattr(obj, 'pad_left', None)
        self.pad_right = getattr(obj, 'pad_right', None)

    def slice_tiles(self, n_tiles):
        """
        Slice the tiled image by the number of tiles and update positions accordingly.

        Parameters:
        - n_tiles: Number of tiles to slice.
        
        Returns:
        - Sliced TiledImage instance.
        """
        sliced_tiles = self[:n_tiles]
        sliced_positions = self.positions[:n_tiles]
        obj = sliced_tiles.view(TiledImage)
        obj.positions = sliced_positions
        obj.pad_top = self.pad_top
        obj.pad_bottom = self.pad_bottom
        obj.pad_left = self.pad_left
        obj.pad_right = self.pad_right
        obj.tile_size = self.tile_size
        obj.stride = self.stride
        obj.min_padding = self.min_padding
        obj.original_shape = self.original_shape
        obj.padded_shape = self.padded_shape
        return obj

    def combine_tiles2(self, crop=True):
        """
        Combine the overlapping tiles to create a complete image by averaging overlapping regions.

        Parameters:
        - crop: Boolean, if True (default), crops the image to the original size, otherwise returns the full padded image.

        Returns:
        - Numpy array representing the reconstructed image.
        """
        if self.positions is None:
            raise ValueError("Positions are not set. Make sure the TiledImage was created properly with the correct positions.")

        # Determine if the image is single-channel or multi-channel based on the tile shape
        is_single_channel = self.shape[-1] == 1 if self.ndim == 4 else True

        # Initialize the reconstructed image and weight matrix
        if is_single_channel:
            reconstructed_image = np.zeros((self.padded_shape[0], self.padded_shape[1]), dtype=np.float32)
            weight_matrix = np.zeros((self.padded_shape[0], self.padded_shape[1]), dtype=np.float32)
        else:
            reconstructed_image = np.zeros((self.padded_shape[0], self.padded_shape[1], self.shape[-1]), dtype=np.float32)
            weight_matrix = np.zeros((self.padded_shape[0], self.padded_shape[1], 1), dtype=np.float32)

        # Iterate through each tile and add it to the reconstructed image with appropriate weights
        for tile, (y, x) in zip(self, self.positions):
            tile_height, tile_width = tile.shape[0:2]
            
            if is_single_channel:
                reconstructed_image[y:y+tile_height, x:x+tile_width] += tile  # Single channel, no need to squeeze
                weight_matrix[y:y+tile_height, x:x+tile_width] += 1
            else:
                reconstructed_image[y:y+tile_height, x:x+tile_width, :] += tile
                weight_matrix[y:y+tile_height, x:x+tile_width, :] += 1

        # Avoid division by zero
        weight_matrix[weight_matrix == 0] = 1

        # Average the overlapping areas
        if is_single_channel:
            reconstructed_image /= weight_matrix
        else:
            reconstructed_image /= weight_matrix

        # Crop the padded area to return the original image size if crop is True
        if crop:
            reconstructed_image = reconstructed_image[
                self.pad_top:self.padded_shape[0] - self.pad_bottom,
                self.pad_left:self.padded_shape[1] - self.pad_right
            ]

        return ContinuousSingleChannelImage(reconstructed_image)


    def reform_image_overwrite(self, crop=True):
        """
        Reform the image using overwriting reconstruction without any averaging or other operations.

        Parameters:
        - crop: Boolean, if True (default), crops the image to the original size, otherwise returns the full padded image.

        Returns:
        - Numpy array representing the reconstructed image.
        """
        return ContinuousSingleChannelImage(self._combine_tiles_overwrite(crop))

    def combine_tiles(self, crop=True, method="average"):
        """
        Combine the overlapping tiles to create a complete image by averaging overlapping regions or overwriting.

        Parameters:
        - crop: Boolean, if True (default), crops the image to the original size, otherwise returns the full padded image.
        - method: String, combining method - "average" (default) or "overwrite".

        Returns:
        - Numpy array representing the reconstructed image.
        """
        if self.positions is None:
            raise ValueError("Positions are not set. Make sure the TiledImage was created properly with the correct positions.")

        if method == "average":
            return ContinuousSingleChannelImage(self._combine_tiles_average(crop))
        elif method == "overwrite":
            return ContinuousSingleChannelImage(self._combine_tiles_overwrite(crop))
        elif method == "gaussian_blending":
            return ContinuousSingleChannelImage(self._combine_tiles_gaussian_blending(crop))
        else:
            raise ValueError(f"Invalid method '{method}'. Choose from 'average' or 'overwrite'.")

    def _combine_tiles_average(self, crop):
        """
        Combine tiles by averaging overlapping regions.
        """
        # Determine if the image is single-channel or multi-channel based on the tile shape
        is_single_channel = self.shape[-1] == 1 if self.ndim == 4 else True

        # Initialize the reconstructed image and weight matrix
        if is_single_channel:
            reconstructed_image = np.zeros((self.padded_shape[0], self.padded_shape[1]), dtype=np.float32)
            weight_matrix = np.zeros((self.padded_shape[0], self.padded_shape[1]), dtype=np.float32)
        else:
            reconstructed_image = np.zeros((self.padded_shape[0], self.padded_shape[1], self.shape[-1]), dtype=np.float32)
            weight_matrix = np.zeros((self.padded_shape[0], self.padded_shape[1], 1), dtype=np.float32)

        # Iterate through each tile and add it to the reconstructed image with appropriate weights
        for tile, (y, x) in zip(self, self.positions):
            tile_height, tile_width = tile.shape[0:2]
            
            if is_single_channel:
                reconstructed_image[y:y+tile_height, x:x+tile_width] += tile  # Single channel, no need to squeeze
                weight_matrix[y:y+tile_height, x:x+tile_width] += 1
            else:
                reconstructed_image[y:y+tile_height, x:x+tile_width, :] += tile
                weight_matrix[y:y+tile_height, x:x+tile_width, :] += 1

        # Avoid division by zero
        weight_matrix[weight_matrix == 0] = 1

        # Average the overlapping areas
        if is_single_channel:
            reconstructed_image /= weight_matrix
        else:
            reconstructed_image /= weight_matrix

        # Crop the padded area to return the original image size if crop is True
        if crop:
            reconstructed_image = reconstructed_image[
                self.pad_top:self.padded_shape[0] - self.pad_bottom,
                self.pad_left:self.padded_shape[1] - self.pad_right
            ]

        return reconstructed_image

    def _combine_tiles_overwrite(self, crop):
        """
        Combine tiles by overwriting values without averaging.
        """
        # Determine if the image is single-channel or multi-channel based on the tile shape
        is_single_channel = self.shape[-1] == 1 if self.ndim == 4 else True

        # Initialize the reconstructed image
        if is_single_channel:
            reconstructed_image = np.zeros((self.padded_shape[0], self.padded_shape[1]), dtype=self.dtype)
        else:
            reconstructed_image = np.zeros((self.padded_shape[0], self.padded_shape[1], self.shape[-1]), dtype=self.dtype)

        # Iterate through each tile and overwrite it in the reconstructed image
        for tile, (y, x) in zip(self, self.positions):
            tile_height, tile_width = tile.shape[0:2]
            
            if is_single_channel:
                reconstructed_image[y:y+tile_height, x:x+tile_width] = tile  # Overwrite
            else:
                reconstructed_image[y:y+tile_height, x:x+tile_width, :] = tile

        # Crop the padded area to return the original image size if crop is True
        if crop:
            reconstructed_image = reconstructed_image[
                self.pad_top:self.padded_shape[0] - self.pad_bottom,
                self.pad_left:self.padded_shape[1] - self.pad_right
            ]

        return reconstructed_image


    def _combine_tiles_gaussian_blending(self, crop, sigma=10):
        """
        Combine tiles by applying Gaussian blending to overlapping regions.
        
        Parameters:
            crop (bool): If True, crop the padded area to return the original image size.
            sigma (float): Standard deviation for Gaussian kernel to control blending. Default is 10.
        """
        # Determine if the image is single-channel or multi-channel based on the tile shape
        is_single_channel = self.shape[-1] == 1 if self.ndim == 4 else True

        # Initialize the reconstructed image and weight matrix
        if is_single_channel:
            reconstructed_image = np.zeros((self.padded_shape[0], self.padded_shape[1]), dtype=np.float32)
            weight_matrix = np.zeros((self.padded_shape[0], self.padded_shape[1]), dtype=np.float32)
        else:
            reconstructed_image = np.zeros((self.padded_shape[0], self.padded_shape[1], self.shape[-1]), dtype=np.float32)
            weight_matrix = np.zeros((self.padded_shape[0], self.padded_shape[1], 1), dtype=np.float32)

        # Create a Gaussian weight mask for each tile
        tile_height, tile_width = self.shape[1:3]
        gaussian_weights = np.ones((tile_height, tile_width), dtype=np.float32)
        gaussian_weights = gaussian_filter(gaussian_weights, sigma=sigma)

        # Normalize the weights to make sure the sum is 1 within each tile
        gaussian_weights /= gaussian_weights.max()

        # If multi-channel, expand the Gaussian weights to match the number of channels
        if not is_single_channel:
            gaussian_weights = np.expand_dims(gaussian_weights, axis=-1)

        # Iterate through each tile and add it to the reconstructed image with Gaussian weights
        for tile, (y, x) in zip(self, self.positions):
            tile_height, tile_width = tile.shape[0:2]
            
            if is_single_channel:
                reconstructed_image[y:y+tile_height, x:x+tile_width] += tile * gaussian_weights
                weight_matrix[y:y+tile_height, x:x+tile_width] += gaussian_weights
            else:
                reconstructed_image[y:y+tile_height, x:x+tile_width, :] += tile * gaussian_weights
                weight_matrix[y:y+tile_height, x:x+tile_width, :] += gaussian_weights

        # Avoid division by zero
        weight_matrix[weight_matrix == 0] = 1

        # Average the overlapping areas with the weighted sum
        if is_single_channel:
            reconstructed_image /= weight_matrix
        else:
            reconstructed_image /= weight_matrix

        # Crop the padded area to return the original image size if crop is True
        if crop:
            reconstructed_image = reconstructed_image[
                self.pad_top:self.padded_shape[0] - self.pad_bottom,
                self.pad_left:self.padded_shape[1] - self.pad_right
            ]

        return reconstructed_image

