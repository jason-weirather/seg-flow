from ..tiled_image import TiledImage

class GenericImageProcessingMethod:
    def __init__(self, image_mpp):
        """
        Initialize the GenericImageProcessingMethod class with a microns per pixel measurement.
        
        Parameters:
        - image_mpp: Microns per pixel for scaling during segmentation.
        """
        self.image_mpp = image_mpp

    def run_image_processing(self, tiles, **kwargs):
        if not isinstance(tiles,TiledImage):
            raise ValueErorr("Input tiles must be of type TiledImage")
        generated_tiles = self._run_image_processing(tiles, **kwargs)
        return TiledImage.from_tiled_array(
            generated_tiles,
            tiles.positions,
            tiles.pad_top,
            tiles.pad_bottom,
            tiles.pad_left,
            tiles.pad_right
        )
    def _run_image_processing(self, tiles, **kwargs):
        """
        Perform an image process on tiles.
        This method should be implemented by subclasses.
        
        Parameters:
        - tile: A single image tile to segment.
        - kwargs: Additional parameters for segmentation.
        
        Returns:
        - labels: Processed images for the given tiles.
        """
        raise NotImplementedError("Subclasses should implement this method.")
