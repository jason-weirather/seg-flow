import numpy as np
from skimage.filters import threshold_otsu
from .segmentationimage import SegmentationImage

class ContinuousSingleChannelImage(np.ndarray):
    def __new__(cls, input_array):
        """
        Create a new ContinuousSingleChannelImage instance, inheriting from numpy.ndarray.

        Parameters:
        - input_array: numpy array, the input data for the single channel.
        """
        if not isinstance(input_array, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if input_array.ndim != 2:
            raise ValueError("Input must be a 2D numpy array")

        # Cast the input_array to our new class (ContinuousSingleChannelImage)
        obj = np.asarray(input_array).view(cls)
        return obj

    def __init__(self, input_array):
        """
        This method is used for initializing extra attributes or logic. 
        Since numpy array creation happens in __new__, __init__ is mostly used for
        initializing any non-array attributes.
        """
        pass  # We are not adding any extra attributes at this point

    def apply_threshold(self, threshold_value):
        """
        Apply a threshold to the channel data.

        Parameters:
        - threshold_value: float, the value to threshold the data. Values above the threshold
          will be set to 1, values below will be set to 0.

        Returns:
        - thresholded_data: numpy array, the binary thresholded version of the input data.
        """
        return SegmentationImage(self >= threshold_value).astype(np.uint8)

    def calculate_otsu_threshold(self):
        """
        Method 1: Otsu's Thresholding.
        Calculate Otsu's threshold from the current channel data.
        
        Returns:
        - float, the calculated Otsu threshold value.
        """
        return threshold_otsu(self.flatten())

    def calculate_percentile_threshold(self, percentile=90):
        """
        Method 2: Percentile-Based Thresholding.

        Parameters:
        - percentile: float, the percentile to use for thresholding.

        Returns:
        - float, the calculated percentile threshold.
        """
        return np.percentile(self.flatten(), percentile)

    def calculate_z_score_threshold(self, k=1.0):
        """
        Method 3: Z-Score Thresholding.

        Parameters:
        - k: float, the Z-score factor. The threshold is calculated as mean + k * standard deviation.

        Returns:
        - float, the calculated Z-score threshold.
        """
        flattened_data = self.flatten()
        mean_val = np.mean(flattened_data)
        std_val = np.std(flattened_data)
        return mean_val + k * std_val

    def determine_threshold(self, method="otsu"):
        """
        Determine threshold using one of the methods.

        Parameters:
        - texture_masks: numpy array, the texture masks to calculate the threshold from.
        - method: str, method to use for threshold calculation. Choose from "otsu", "percentile", or "zscore".
        
        Returns:
        - threshold: float, the calculated threshold value.
        """
        if method == "otsu":
            threshold = self.calculate_otsu_threshold()
        elif method == "percentile":
            threshold = self.calculate_percentile_threshold(percentile=90)
        elif method == "zscore":
            threshold = self.calculate_z_score_threshold(k=1.0)
        else:
            raise ValueError("Invalid method specified. Choose from 'otsu', 'percentile', or 'zscore'.")

        return threshold
