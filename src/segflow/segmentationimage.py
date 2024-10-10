import numpy as np

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

        # Cast the input_array to our new class (SegmentationImage)
        obj = np.asarray(input_array).view(cls)
        return obj

    def __init__(self, input_array):
        """
        This method is used for initializing extra attributes or logic. 
        Since numpy array creation happens in __new__, __init__ is mostly used for
        initializing any non-array attributes.
        """
        pass  # We are not adding any extra attributes at this point

    def randomize_segmentation(self, seed=1):
        """
        Randomize cell labels in the segmentation mask for better visualization.

        Parameters:
        - segmentation_image: numpy array, the segmentation mask image.
        - seed: Random seed for reproducibility.
        
        Returns:
        - segmentation_image: numpy array, the randomized segmentation mask.
        """
        
        # Identify unique non-zero labels
        unique_labels = np.unique(self)
        non_zero_labels = unique_labels[unique_labels > 0]

        # Create a random permutation of the non-zero labels
        randomized_labels = np.random.RandomState(seed=seed).permutation(len(non_zero_labels)) + 1  # Start labels from 1

        # Create a mapping that retains zero (background)
        label_mapping = np.zeros(unique_labels.max() + 1, dtype=np.int32)
        label_mapping[non_zero_labels] = randomized_labels

        # Apply the mapping to the segmentation image
        segmentation_remapped = label_mapping[self]
        
        return self.__class__(segmentation_remapped)
