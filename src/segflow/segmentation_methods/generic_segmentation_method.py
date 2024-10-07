class GenericSegmentationMethod:
    def __init__(self, image_mpp):
        """
        Initialize the GenericSegmentationMethod class with a microns per pixel measurement.
        
        Parameters:
        - image_mpp: Microns per pixel for scaling during segmentation.
        """
        self.image_mpp = image_mpp

    def run_segmentation(self, tile, **kwargs):
        """
        Perform segmentation on a single tile.
        This method should be implemented by subclasses.
        
        Parameters:
        - tile: A single image tile to segment.
        - kwargs: Additional parameters for segmentation.
        
        Returns:
        - labels: Segmented labels for the given tile.
        """
        raise NotImplementedError("Subclasses should implement this method.")
