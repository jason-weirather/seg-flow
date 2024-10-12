import numpy as np
from skimage import filters, morphology, measure, segmentation, exposure, feature, util
from scipy import ndimage as ndi


from .generic_segmentation_method import GenericSegmentationMethod

class SKWSegmentationMethod(GenericSegmentationMethod):
    def __init__(self, image_mpp, sigma=1, window_size=None, min_size=None, log_sigma=2, min_distance=None):
        """
        Initialize the SKWSegmentationMethod class with parameters for the segmentation approach.
        
        Parameters:
        - image_mpp: Microns per pixel for scaling during segmentation.
        - sigma: Standard deviation for Gaussian smoothing.
        - window_size: Window size for Sauvola thresholding.
        - min_size: Minimum size for small object removal.
        - log_sigma: Standard deviation for Laplacian of Gaussian filter.
        - min_distance: Minimum distance for peak local maxima.
        """
        super().__init__(image_mpp)
        self.sigma = sigma
        # Set default values based on image_mpp if not provided
        self.window_size = window_size if window_size is not None else int(25 * (image_mpp / 0.28))
        self.min_size = min_size if min_size is not None else int(100 * (image_mpp / 0.28)**2)
        self.log_sigma = log_sigma
        self.min_distance = min_distance if min_distance is not None else int(5 * (image_mpp / 0.28))

    def _run_segmentation(self, tiles, batch_size=64):
        """
        Perform SKWS (Scikit-image Kernel [Gaussian, LoG, Sobel] Watershed) Segmentation on a batch of tiles.
        
        Parameters:
        - tiles: A batch of image tiles to segment.
        - batch_size: Number of tiles to process in each batch.
        
        Returns:
        - segmentation_tiles: Numpy array of segmented labels for each tile with shape (n_tiles, tile_height, tile_width, 1).
        """
        # Extract the DAPI channel (channel 0)
        segmentation_tiles = []
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i+batch_size]
            batch_segmentation = [self._segment_single_tile(tile) for tile in batch_tiles]
            segmentation_tiles.extend(batch_segmentation)
            print(f"Processed batch {i // batch_size + 1}/{(len(tiles) - 1) // batch_size + 1}")
        segmentation_tiles = np.array(segmentation_tiles)[..., np.newaxis].astype(np.uint32)
        return segmentation_tiles

    def _segment_single_tile(self, tile):
        """
        Perform segmentation on a single tile.
        
        Parameters:
        - tile: A single image tile to segment.
        
        Returns:
        - labels: Segmented labels for the given tile.
        """
        # Extract the DAPI channel (channel 0)
        dapi_image = tile[:, :, 0]
        
        # Convert image to float in [0, 1]
        dapi_image = util.img_as_float(dapi_image)
        
        # Preprocessing: Apply Gaussian smoothing to reduce noise
        smoothed = filters.gaussian(dapi_image, sigma=self.sigma)
        
        # Rescale smoothed image to [0, 1]
        smoothed = exposure.rescale_intensity(smoothed, in_range='image', out_range=(0, 1))
        
        # Contrast enhancement using histogram equalization
        equalized = exposure.equalize_adapthist(smoothed)
        
        # Adaptive Thresholding: Use Sauvola's method to create a binary image
        threshold_sauvola = filters.threshold_sauvola(equalized, window_size=self.window_size)
        binary = equalized > threshold_sauvola
        
        # Morphological operations to remove small objects and fill holes
        cleaned = morphology.remove_small_objects(binary, min_size=self.min_size)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=self.min_size)
        
        # Compute the distance transform
        distance = ndi.distance_transform_edt(cleaned)
        
        # Use the Laplacian of Gaussian (LoG) to find nuclei centers
        log_image = -filters.laplace(filters.gaussian(equalized, sigma=self.log_sigma))
        coordinates = feature.peak_local_max(log_image, indices=False, min_distance=self.min_distance, labels=cleaned)
        markers = measure.label(coordinates)
        
        # Watershed segmentation using the gradient magnitude
        gradient = filters.sobel(equalized)
        labels = segmentation.watershed(gradient, markers, mask=cleaned)
        
        # Remove small labeled objects (noise)
        labels = morphology.remove_small_objects(labels, min_size=self.min_size)
        
        return labels

