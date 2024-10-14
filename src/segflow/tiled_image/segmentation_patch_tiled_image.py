from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes

from tqdm import tqdm
from copy import deepcopy, copy
import numpy as np
import sys

from .segmentation_tiled_image import SegmentationTiledImage
from ..full_image import SegmentationImage

class SegmentationPatchTiledImage(SegmentationTiledImage):
    """
    SegmentationPathTiledImage is a specialized version of SegmentationTiledImage for handling
    segmented images where each tile represents a patch representative of a cell.
    """
    def __new__(cls, input_image):
        """
        Create a new SegmentationPatchTiledImage instance by copying
        an existing SegmentationTiledImage.
        """
        # Ensure the input image is a SegmentationTiledImage or TiledImage
        if not isinstance(input_image, SegmentationPatchTiledImage):
            raise ValueError("input_image must be an instance SegmentationPatchTiledImage.")
        
        # Create the new instance as a view of the input_image
        obj = input_image.view(cls)
        obj.patch_descriptions = input_image.patch_descriptions
        return obj

    @classmethod
    def from_image(cls, input_image, bbox_size):
        """
        Factory method to create a SegmentationPatchTiledImage from a fresh SegmentationImage image.

        Parameters:
        - input_image: Numpy array of the segmented image (height, width).
        
        Returns:
        - SegmentationPatchTiledImage instance.
        """        
        # Ensure the input image is a SegmentationImage
        if not isinstance(input_image, SegmentationImage):
            raise ValueError("input_image must be an instance of SegmentationImage.")

        saved_bbox_size = input_image.bbox_size
        input_image.bbox_size = bbox_size
        centroids = input_image.centroids
        centroid_list = deepcopy([x for x in centroids.values()])
        if saved_bbox_size is not None:
            input_image.bbox_size = saved_bbox_size

        bbox_height, bbox_width = bbox_size
        n_patches = len(centroids.keys())

        # Initialize an array for patches
        patches_array = np.zeros((n_patches, bbox_height, bbox_width), dtype=input_image.dtype)
        image_height, image_width = input_image.shape

        for idx, (label_value, centroid_info) in tqdm(enumerate(centroids.items()),desc="Building patches",total=len(centroids)):
            y_min, x_min = centroid_info['bbox_position']

            # Calculate bounding box coordinates
            y_max = min(y_min + bbox_height, image_height)
            x_max = min(x_min + bbox_width, image_width)

            # Extract the patch
            patch = input_image[y_min:y_max, x_min:x_max]

            patches_array[idx,:,:] = patch


        # Call the base class method to create the tiled image
        obj = super(SegmentationPatchTiledImage, cls).from_tiled_array(
            np.array(patches_array, dtype=input_image.dtype),
            positions = [x['bbox_position'] for x in centroids.values()],
            original_shape = input_image.shape,
            pad_top = 0,
            pad_bottom = 0,
            pad_left = 0,
            pad_right = 0
        )
        obj.patch_descriptions = centroid_list
        # Add any segmentation-specific initialization here, if needed
        return obj
    
    @classmethod
    def from_tiled_array(cls, tiled_array, positions, original_shape, centroid_list):
        """
        Factory method to create a SegmentationPatchTiledImage from a numpy array, positions, and padding parameters.

        Parameters:
        - tiled_array: A 3D numpy array of shape (n_patches, height, width), where each patch represents a cell or region.
        - positions: A list of (y_min, x_min) tuples indicating the top-left corner of each patch in the original image.
        - original_shape: Tuple (height, width) representing the shape of the full original image.
        - pad_top, pad_bottom, pad_left, pad_right: Padding values to apply to the image.

        Returns:
        - SegmentationPatchTiledImage instance.
        """

        # Initialize an empty array for the full image
        n_patches = len(centroid_list)
        full_image = np.zeros((tiled_array.shape[0], tiled_array.shape[1], tiled_array.shape[2]), dtype=tiled_array.dtype)

        obj = super(SegmentationPatchTiledImage, cls).from_tiled_array(tiled_array, positions, original_shape, 0, 0, 0, 0)
        obj.patch_descriptions = deepcopy(centroid_list)
        
        # Add any segmentation-specific initialization here, if needed
        return obj

    def combine_tiles(self, iou_threshold=0.5, crop=True):
        """
        Ingest the segmented tiles and recombine them into a full segmentation mask, handling overlaps.

        Parameters:
        - iou_threshold: Threshold for IoU to decide if segments represent the same cell.
        - crop: Whether to crop the final segmentation mask to the original image size.
        
        Returns:
        - Segmentation mask of the full image.
        """
        raise NotImplementedError("This feature is not implemented yet.")

    def combine_tiles(self, method="all_labels"):
        """
        Ingest the segmented tiles and recombine them into a full segmentation mask, handling overlaps.

        Parameters:
        - method: Choice of methods for combining the images. Default is 'all_labels'.
        
        Returns:
        - SegmentationImage: A combined image from all patches.
        """
        # Initialize an empty output image with the same shape as the original image
        output_image = np.zeros(self.original_shape, dtype=self.dtype)

        # Iterate over each patch and its corresponding description
        for i, patch_description in enumerate(self.patch_descriptions):
            patch = self[i]
            # Extract bounding box position from patch_description
            y_min, x_min = patch_description['bbox_position']

            # Get the dimensions of the patch
            patch_height, patch_width = patch.shape

            # Calculate the area in the output image to update (y_max, x_max)
            y_max = y_min + patch_height
            x_max = x_min + patch_width

            # Ensure that the patch does not exceed the output image dimensions
            if y_max > output_image.shape[0] or x_max > output_image.shape[1]:
                raise ValueError(f"Patch at {i} exceeds bounds of the output image")

            # Create a mask for patch > 0 (treat patch == 0 as transparent)
            patch_mask = patch > 0

            # Update the output_image at the corresponding location using patch_mask
            output_image_region = output_image[y_min:y_max, x_min:x_max]
            
            # Ensure shapes match before applying the mask
            if output_image_region.shape != patch.shape:
                raise ValueError(f"Shape mismatch at patch {i}: "
                                 f"patch shape {patch.shape} vs region shape {output_image_region.shape}")

            # Apply the mask to update only the non-zero pixels
            output_image_region[patch_mask] = patch[patch_mask]

            # Now place the updated region back into the full image
            output_image[y_min:y_max, x_min:x_max] = output_image_region

        # Return the combined image as a SegmentationImage
        return SegmentationImage(output_image)


    def isolate_center_labels(self):
        """
        For each patch, remove all labels except for the label the patch is centered upon.
        
        Returns:
        - A new SegmentationPatchTiledImage where only the center label remains in each patch.
        """
        # Initialize an empty array for the modified patches
        n_patches = len(self.patch_descriptions)
        bbox_height, bbox_width = self.shape[1:3]  # Assumes patches are of shape (n_patches, height, width)
        
        patches_array = np.zeros((n_patches, bbox_height, bbox_width), dtype=self.dtype)

        for i, centroid in tqdm(enumerate(self.patch_descriptions),desc="Isolating cells",total=len(self.patch_descriptions)):
            region_label = centroid['region_label']  # The label that should be kept in the current patch

            # Get the current patch
            patch = self[i]

            # Zero out all labels in the patch except the region_label
            patches_array[i, :, :] = np.where(patch == region_label, patch, 0)
        
        # Create a new SegmentationPatchTiledImage from the modified patches
        new_instance = self.__class__.from_tiled_array(
            np.array(patches_array, dtype=self.dtype),
            positions=[x['bbox_position'] for x in self.patch_descriptions],
            original_shape=self.original_shape,  # Shape of the full image, excluding patches dimension
            centroid_list=deepcopy(self.patch_descriptions)
        )
        return new_instance

    
    def remove_disjointed_pixels(self):
        """
        Filter through the segments and for each label on each segment,
        if a label is disjointed remove the smallest labels, keeping only the largest connected component.

        Will be much faster if isolate_center_labels() is run first
        """
        total_pixels_removed = 0  # Counter for total removed pixels

        # Iterate through each patch in the image
        for i in tqdm(range(self.shape[0]), desc="Remove disjointed pixels", total=self.shape[0]):
            patch = self[i]  # Get the current patch (256x256)

            # Get the unique labels in the patch (ignoring label 0 for background)
            unique_labels = np.unique(patch)
            unique_labels = unique_labels[unique_labels > 0]  # Exclude background

            # Iterate over each unique label and process the connected components
            for label_value in unique_labels:
                # Create a binary mask where the current label is 1 and everything else is 0
                label_mask = patch == label_value

                # Label the connected components for this specific label
                labeled_components, num_components = label(label_mask, return_num=True, connectivity=2)

                if num_components > 1:  # Only process if there are multiple connected components
                    # Measure the size of each connected component
                    component_sizes = [(region.label, region.area) for region in regionprops(labeled_components)]

                    # Find the label of the largest connected component
                    largest_component_label = max(component_sizes, key=lambda x: x[1])[0]

                    # Count the pixels to be removed (smaller components)
                    pixels_to_remove = np.sum(labeled_components != largest_component_label)
                    total_pixels_removed += pixels_to_remove

                    # Zero out all pixels except for the largest connected component
                    patch[labeled_components != largest_component_label] = 0

            # Update the patch with the new version where disjointed components are removed
            self[i] = patch

        # Output the total number of removed pixels to stderr
        print(f"Total pixels removed: {total_pixels_removed}", file=sys.stderr)

        return self

    def find_patches_with_small_labels(self, min_area_px):
        """
        Identify labels in each patch that have fewer pixels than the min_area_px threshold.

        Parameters:
        - min_area_px: Minimum number of pixels required for a label to be considered significant.

        Returns:
        - small_labels (set): A set of label values that have fewer pixels than the min_area_px threshold.
        """
        small_labels = set()  # Set to store small labels

        # Iterate through each patch in the image
        for i in tqdm(range(self.shape[0]), desc="Identifying small labels", total=self.shape[0]):
            patch = self[i]  # Get the current patch (256x256)

            # Get the unique labels in the patch (ignoring label 0 for background)
            unique_labels = np.unique(patch)
            unique_labels = unique_labels[unique_labels > 0]  # Exclude background

            # Iterate over each unique label and check its total pixel count
            for label_value in unique_labels:
                # Count the total number of pixels for the current label
                label_pixel_count = np.sum(patch == label_value)

                # If the label's pixel count is smaller than the threshold, add to the list
                if label_pixel_count < min_area_px:
                    small_labels.add(label_value)

        return small_labels


    def drop_labels(self, labels_to_drop):
        """
        Drop or modify patches based on the given list of region_label integers.
        
        For each patch, if the region_label of the patch is in the labels_to_drop list, 
        omit the patch. For all other patches, set any pixel matching a label in labels_to_drop to zero.

        Parameters:
        - labels_to_drop (list): List of region_label integers to be dropped or zeroed out.

        Updates:
        - self: Removes the specified patches and updates self.patch_descriptions and self.positions accordingly.
        """
        new_patches = []
        new_patch_descriptions = []
        new_positions = []

        # Iterate over each patch, description, and position
        for i, (patch, description, position) in enumerate(zip(self, self.patch_descriptions, self.positions)):
            region_label = description['region_label']

            if region_label in labels_to_drop:
                # Omit the patch if its region_label is in the labels_to_drop list
                continue

            # Set any pixel that matches a label in labels_to_drop to zero
            patch[np.isin(patch, labels_to_drop)] = 0

            # Append the updated patch, description, and position to the new lists
            new_patches.append(patch)
            new_patch_descriptions.append(description)
            new_positions.append(position)

        # If no patches remain, raise an error
        if len(new_patches) == 0:
            raise ValueError("All patches were removed.")

        # Convert list of new_patches to a numpy array
        new_patches_array = np.array(new_patches)

        # Rebuild the SegmentationPatchTiledImage using the from_tiled_array class method
        return self.__class__.from_tiled_array(
            new_patches_array,
            new_positions,
            self.original_shape,
            new_patch_descriptions
        )

    def find_patches_with_missing_labels(self):
        """
        Traverse patches and find any patches where the 'region_label' is missing.
        
        Returns:
        - missing_labels (list): A list of 'region_label' values that are missing from their respective patches.
        """
        missing_labels = []  # List to store labels that are missing from their patches

        # Iterate over each patch, description, and position
        for i, (patch, description) in enumerate(zip(self, self.patch_descriptions)):
            region_label = description['region_label']

            # Check if the region_label is present in the patch
            if not np.any(patch == region_label):
                # If the region_label is missing, save it to the missing_labels list
                missing_labels.append(region_label)

        return missing_labels


    def find_patches_with_circumscribed_labels(self):
        """
        Identify labels that are circumscribed by the main 'region_label' in each patch.
        
        For each patch, we flood-fill the region_label and then check which other labels overlap
        with this filled mask. Any overlapping label is considered circumscribed.
        

        You should not run isolate_center_labels() before running this, because this requires having all the patch labels present.

        Returns:
        - circumscribed_labels (set): A set of integers corresponding to labels that are circumscribed.
        """
        circumscribed_labels = set()  # Store labels that are circumscribed

        # Iterate through each patch
        for i in tqdm(range(self.shape[0]), desc="Finding circumscribed labels", total=self.shape[0]):
            patch = self[i]  # Get the current patch (256x256)
            region_label = self.patch_descriptions[i]['region_label']  # The main label for this patch

            # Create binary mask for the region_label and flood fill any internal holes
            region_mask = (patch == region_label)
            filled_region_mask = binary_fill_holes(region_mask)

            # Get all unique labels in the patch, excluding the background (label 0) and the region_label
            unique_labels = np.unique(patch)
            unique_labels = unique_labels[(unique_labels > 0) & (unique_labels != region_label)]

            # Check each label to see if it overlaps with the filled region mask
            for label_value in unique_labels:
                # Create binary mask for the current label
                label_mask = (patch == label_value)

                # Check if there is any overlap between the label mask and the filled region mask
                if np.any(label_mask & filled_region_mask):
                    # If there is overlap, the label is circumscribed
                    circumscribed_labels.add(label_value)

        # Return the set of circumscribed labels
        return circumscribed_labels            

