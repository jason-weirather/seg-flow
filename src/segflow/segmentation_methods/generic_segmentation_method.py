from ..tiled_image import TiledImage, SegmentationTiledImage

class GenericSegmentationMethod:
    def __init__(self, image_mpp):
        """
        Initialize the GenericSegmentationMethod class with a microns per pixel measurement.
        
        Parameters:
        - image_mpp: Microns per pixel for scaling during segmentation.
        """
        self.image_mpp = image_mpp

    def run_segmentation(self, tiles, **kwargs):
        if not isinstance(tiles,TiledImage):
            raise ValueError("Tiles going for segmentation must be of type TiledImage")
        generated_tiles = self._run_segmentation(tiles,**kwargs)
        return SegmentationTiledImage.from_tiled_array(
            generated_tiles,
            tiles.positions,
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
