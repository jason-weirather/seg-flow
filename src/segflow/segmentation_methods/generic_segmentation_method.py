from ..tiled_image import TiledImage, SegmentationTiledImage
import numpy as np

class GenericSegmentationMethod:
    def __init__(self, image_mpp):
        """
        Initialize the GenericSegmentationMethod class with a microns per pixel measurement.

        Parameters:
        - image_mpp: Microns per pixel for scaling during segmentation.
        """
        self.image_mpp = image_mpp

    def run_segmentation(self, tiles, **kwargs):
        if not isinstance(tiles, TiledImage):
            raise ValueError("Tiles going for segmentation must be of type TiledImage")

        # Run the actual segmentation
        generated_tiles = self._run_segmentation(tiles, **kwargs)

        # Standardize the shape to (n_tiles, height, width)
        if generated_tiles.ndim == 4 and generated_tiles.shape[-1] == 1:
            # Squeeze out the unnecessary last dimension if it is (n_tiles, height, width, 1)
            generated_tiles = np.squeeze(generated_tiles, axis=-1)

        # Ensure dtype is integer or boolean
        if not np.issubdtype(generated_tiles.dtype, np.integer) and not np.issubdtype(generated_tiles.dtype, np.bool_):
            raise ValueError("Generated tiles must be of integer or boolean data type")

        # Return the SegmentationTiledImage
        return SegmentationTiledImage.from_tiled_array(
            generated_tiles,
            tiles.positions,
            tiles.original_shape,
            tiles.pad_top,
            tiles.pad_bottom,
            tiles.pad_left,
            tiles.pad_right
        )

    def _run_segmentation(self, tiles, **kwargs):
        """
        Perform segmentation on TiledImage.
        This method should be implemented by subclasses.

        Parameters:
        - tiles: A TiledImage class input.
        - kwargs: Additional parameters for segmentation.

        Returns:
        - labels: Segmented labels for the given tile.
        """
        raise NotImplementedError("Subclasses should implement this method.")

