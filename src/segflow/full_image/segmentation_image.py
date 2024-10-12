import numpy as np
from skimage.segmentation import expand_labels
from skimage.morphology import binary_dilation, binary_erosion, disk
from scipy.ndimage import binary_erosion as nd_binary_erosion

class SegmentationImage(np.ndarray):
    def __new__(cls, input_array):
        """
        Create a new SegmentationImage instance, inheriting from numpy.ndarray.

        Parameters:
        - input_array: numpy array, the input data for the single channel.
        """
        if not isinstance(input_array, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if input_array.ndim != 2:
            raise ValueError("Input must be a 2D numpy array")
        if not (np.issubdtype(input_array.dtype, np.integer) or input_array.dtype == np.bool_):
            raise TypeError("Input array must have integer or boolean data type")

        # Cast the input_array to our new class (SegmentationImage)
        obj = np.asarray(input_array).view(cls)
        return obj

    def __init__(self, input_array):
        """
        Initialize any extra attributes or logic.
        Since numpy array creation happens in __new__, __init__ is mostly used for
        initializing any non-array attributes.
        """
        pass  # We are not adding any extra attributes at this point

    def randomize_segmentation(self, seed=1):
        """
        Randomize cell labels in the segmentation mask for better visualization.

        Parameters:
        - seed: Random seed for reproducibility.
        
        Returns:
        - SegmentationImage instance with randomized labels.
        """
        # Identify unique non-zero labels
        unique_labels = np.unique(self)
        non_zero_labels = unique_labels[unique_labels > 0]

        # Create a random permutation of the non-zero labels
        rng = np.random.RandomState(seed=seed)
        randomized_labels = rng.permutation(len(non_zero_labels)) + 1  # Start labels from 1

        # Create a mapping that retains zero (background)
        label_mapping = np.zeros(unique_labels.max() + 1, dtype=np.int32)
        label_mapping[non_zero_labels] = randomized_labels

        # Apply the mapping to the segmentation image
        segmentation_remapped = label_mapping[self]
        
        return self.__class__(segmentation_remapped)

    def dilate_segmentation(self, dilation_pixels=1):
        """
        Dilate each non-zero label in the segmentation image by a specified number of pixels.

        Parameters:
        - dilation_pixels: Number of pixels to dilate each label.

        Returns:
        - SegmentationImage instance with dilated labels.
        """
        if dilation_pixels < 1:
            raise ValueError("dilation_pixels must be at least 1")

        if self.dtype == np.bool_ or np.array_equal(np.unique(self), [0, 1]):
            # Binary segmentation
            selem = disk(dilation_pixels)
            dilated = binary_dilation(self, selem)
            return self.__class__(dilated.astype(self.dtype))
        else:
            # Labeled segmentation
            # Use expand_labels from skimage.segmentation
            dilated = expand_labels(self, distance=dilation_pixels)
            return self.__class__(dilated.astype(self.dtype))

    def erode_segmentation(self, erosion_pixels=1):
        """
        Erode each non-zero label in the segmentation image by a specified number of pixels.

        Parameters:
        - erosion_pixels: Number of pixels to erode each label.

        Returns:
        - SegmentationImage instance with eroded labels.
        """
        if erosion_pixels < 1:
            raise ValueError("erosion_pixels must be at least 1")

        if self.dtype == np.bool_ or np.array_equal(np.unique(self), [0, 1]):
            # Binary segmentation
            selem = disk(erosion_pixels)
            eroded = binary_erosion(self, selem)
            return self.__class__(eroded.astype(self.dtype))
        else:
            # Labeled segmentation
            # Erode each label individually to prevent label merging
            labels = np.unique(self)
            labels = labels[labels != 0]  # Exclude background
            eroded = np.zeros_like(self)
            selem = disk(erosion_pixels)
            for label in labels:
                mask = self == label
                eroded_mask = nd_binary_erosion(mask, structure=selem)
                eroded[eroded_mask] = label
            return self.__class__(eroded.astype(self.dtype))

