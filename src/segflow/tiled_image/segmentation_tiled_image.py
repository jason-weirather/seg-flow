from .tiled_image import TiledImage
import numpy as np
from skimage.measure import label
from tqdm import tqdm

from ..full_image import SegmentationImage

class SegmentationTiledImage(TiledImage):
    """
    SegmentationTiledImage is a specialized version of TiledImage for handling
    segmented images where each tile represents categorical or labeled data (e.g., tissue regions).
    """
    def __new__(cls, input_image):
        """
        Create a new SegmentationTiledImage instance by copying an existing TiledImage.
        """
        # Ensure the input image is a SegmentationTiledImage or TiledImage
        if not isinstance(input_image, TiledImage):
            raise ValueError("input_image must be an instance of TiledImage or SegmentationTiledImage.")
        
        # Create the new instance as a view of the input_image
        obj = input_image.view(cls)
        return obj

    @classmethod
    def from_image(cls, input_image, tile_size, stride, min_padding):
        """
        Factory method to create a SegmentationTiledImage from a fresh untiled image.

        Parameters:
        - input_image: Numpy array of the segmented image (height, width) or (height, width, labels).
        - tile_size: Size of each tile.
        - stride: Stride size for tiling the image.
        - min_padding: Minimum padding to add to ensure complete coverage.
        
        Returns:
        - SegmentationTiledImage instance.
        """
        # Call the base class method to create the tiled image
        obj = super(SegmentationTiledImage, cls).from_image(input_image, tile_size, stride, min_padding)
        # Add any segmentation-specific initialization here, if needed
        return obj

    @classmethod
    def from_tiled_array(cls, tiled_array, positions, pad_top, pad_bottom, pad_left, pad_right):
        """
        Factory method to create a SegmentationTiledImage from a numpy array, positions, and padding parameters

        Returns:
        - SegmentationTiledImage instance.
        """
        # Call the base class method to create the tiled image
        obj = super(SegmentationTiledImage, cls).from_tiled_array(tiled_array, positions, pad_top, pad_bottom, pad_left, pad_right)
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
        # Initialize the segmentation mask with the padded shape
        full_segmentation_mask = np.zeros(self.padded_shape, dtype=np.int32)
        max_global_label = 0

        for idx, (tile_segmentation, (y, x)) in tqdm(enumerate(zip(self, self.positions)), desc="Processing tiles", total=len(self.positions)):
            if len(tile_segmentation.shape) == 3:
                tile_segmentation = tile_segmentation.squeeze(axis=-1)
            y_end = y + self.tile_size
            x_end = x + self.tile_size

            # Extract the region from the full segmentation mask that corresponds to the tile
            full_mask_region = full_segmentation_mask[y:y_end, x:x_end]

            # Find unique labels in the tile
            tile_labels = np.unique(tile_segmentation)
            tile_labels = tile_labels[tile_labels != 0]

            for tile_label in tile_labels:
                tile_mask = tile_segmentation == tile_label
                overlap_mask = full_mask_region > 0

                # Find the labels in the overlap region
                overlapping_labels = np.unique(full_mask_region[tile_mask & overlap_mask])
                overlapping_labels = overlapping_labels[overlapping_labels != 0]

                if overlapping_labels.size == 0:
                    # If no overlap, assign a new global label
                    max_global_label += 1
                    full_mask_region[tile_mask] = max_global_label
                else:
                    # If overlap, calculate IoU with existing labels
                    ious = []
                    for overlap_label in overlapping_labels:
                        full_mask_label_mask = full_mask_region == overlap_label
                        iou = self._calculate_iou(tile_mask, full_mask_label_mask)
                        ious.append((iou, overlap_label))

                    # Get the best IoU match
                    best_iou, best_overlap_label = max(ious, key=lambda x: x[0])

                    if best_iou >= iou_threshold:
                        # Merge labels if IoU is above the threshold
                        full_mask_region[tile_mask] = best_overlap_label
                    else:
                        # Otherwise, assign a new label
                        max_global_label += 1
                        full_mask_region[tile_mask] = max_global_label

            # Place the updated region back into the full segmentation mask
            full_segmentation_mask[y:y_end, x:x_end] = full_mask_region

        # Optional: Relabel connected components
        full_segmentation_mask = label(full_segmentation_mask)

        # Crop to the original image size if required
        if crop:
            full_segmentation_mask = full_segmentation_mask[
                self.pad_top:self.padded_shape[0] - self.pad_bottom,
                self.pad_left:self.padded_shape[1] - self.pad_right
            ]

        return SegmentationImage(full_segmentation_mask)


    def _calculate_iou(self, mask1, mask2):
        """
        Calculate the Intersection-over-Union (IoU) of two boolean masks.

        Parameters:
        - mask1: First boolean mask.
        - mask2: Second boolean mask.

        Returns:
        - IoU value.
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union

    def high_confidence_tile_filter(self, margin_size_px):
        """
        Process segmentation tiles to extract high-confidence central regions and adjust labels to ensure uniqueness.
        
        Parameters:
        - segmentation_tiles: Numpy array of segmented tiles.
        - positions: List of positions where each tile starts in the original image.
        
        Returns:
        - Numpy array of high-confidence segmented tiles.
        """

        segmentation_tiles = self
        positions = self.positions

        confidence_segmentation_tiles = []
        max_label = 0  # Initialize max_label for label uniqueness

        for idx, (tile_segmentation, (y, x)) in tqdm(enumerate(zip(segmentation_tiles, positions)),desc="Removing edge cells",total=len(positions)):
            # Extract the high-confidence central region of the tile
            y_start = margin_size_px
            y_end = self.tile_size - margin_size_px
            x_start = margin_size_px
            x_end = self.tile_size - margin_size_px

            high_confidence_region = tile_segmentation[y_start:y_end, x_start:x_end]
            labels_in_high_confidence = np.unique(high_confidence_region)
            labels_in_high_confidence = labels_in_high_confidence[labels_in_high_confidence != 0]

            # Zero out labels not present in the high-confidence region
            mask = np.isin(tile_segmentation, labels_in_high_confidence)
            tile_segmentation_cleaned = np.where(mask, tile_segmentation, 0)

            # Adjust labels to ensure uniqueness across all tiles
            labels_to_adjust = np.unique(tile_segmentation_cleaned)
            labels_to_adjust = labels_to_adjust[labels_to_adjust != 0]

            # Create a mapping to adjust labels
            label_mapping = {label: label + max_label for label in labels_to_adjust}
            tile_segmentation_adjusted = np.zeros_like(tile_segmentation_cleaned)
    
            for old_label, new_label in label_mapping.items():
                tile_segmentation_adjusted[tile_segmentation_cleaned == old_label] = new_label

            # Update max_label for the next tile
            if labels_to_adjust.size > 0:
                max_label = tile_segmentation_adjusted.max()

            confidence_segmentation_tiles.append(tile_segmentation_adjusted)

        confidence_segmentation_tiles = np.array(confidence_segmentation_tiles)
        return SegmentationTiledImage.from_tiled_array(confidence_segmentation_tiles, self.positions, self.pad_top, self.pad_bottom, self.pad_left, self.pad_right)
