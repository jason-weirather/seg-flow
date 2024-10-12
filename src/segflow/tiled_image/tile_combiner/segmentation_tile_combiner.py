import numpy as np
from tqdm import tqdm
from skimage.measure import label

class SegmentationTileCombiner(TileCombiner):
    def __init__(self, tile_size=512, average_weight=0.7, sum_weight=0.3, min_pixels=5):
        self.tile_size = tile_size
        self.average_weight = average_weight
        self.sum_weight = sum_weight
        self.min_pixels = min_pixels

    def combine_segmentation_tiles(self, segmentation_tiles, positions, image_shape, iou_threshold=0.5):
        """
        Ingest the segmented tiles and recombine them into a full segmentation mask, handling overlaps.

        Parameters:
        - segmentation_tiles: Numpy array of segmented tiles.
        - positions: List of positions where each tile starts in the original image.
        - image_shape: Shape of the original padded image.
        - iou_threshold: Threshold for IoU to decide if segments represent the same cell.
        
        Returns:
        - Segmentation mask of the full image.
        """
        full_segmentation_mask = np.zeros(image_shape[:2], dtype=np.int32)
        max_global_label = 0

        for idx, (tile_segmentation, (y, x)) in tqdm(enumerate(zip(segmentation_tiles, positions)), desc="Processing tiles", total=len(positions)):
            tile_segmentation = tile_segmentation.squeeze(axis=-1)
            y_end = y + self.tile_size
            x_end = x + self.tile_size

            full_mask_region = full_segmentation_mask[y:y_end, x:x_end]

            tile_labels = np.unique(tile_segmentation)
            tile_labels = tile_labels[tile_labels != 0]

            for tile_label in tile_labels:
                tile_mask = tile_segmentation == tile_label
                overlap_mask = full_mask_region > 0

                overlapping_labels = np.unique(full_mask_region[tile_mask & overlap_mask])
                overlapping_labels = overlapping_labels[overlapping_labels != 0]

                if overlapping_labels.size == 0:
                    max_global_label += 1
                    full_mask_region[tile_mask] = max_global_label
                else:
                    ious = []
                    for overlap_label in overlapping_labels:
                        full_mask_label_mask = full_mask_region == overlap_label
                        iou = self._calculate_iou(tile_mask, full_mask_label_mask)
                        ious.append((iou, overlap_label))

                    best_iou, best_overlap_label = max(ious, key=lambda x: x[0])

                    if best_iou >= iou_threshold:
                        # Merge labels
                        full_mask_region[tile_mask] = best_overlap_label
                    else:
                        max_global_label += 1
                        full_mask_region[tile_mask] = max_global_label

            full_segmentation_mask[y:y_end, x:x_end] = full_mask_region

        # Optional: Relabel connected components
        full_segmentation_mask = label(full_segmentation_mask)
        return full_segmentation_mask

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

    def high_confidence_tile_filter(self, segmentation_tiles, positions, margin_size_px = None):
        """
        Process segmentation tiles to extract high-confidence central regions and adjust labels to ensure uniqueness.
        
        Parameters:
        - segmentation_tiles: Numpy array of segmented tiles.
        - positions: List of positions where each tile starts in the original image.
        
        Returns:
        - Numpy array of high-confidence segmented tiles.
        """
        if margin_size_px is None:
            margin_size_px = self.tile_size // 8  # For 12.5% margin
        confidence_segmentation_tiles = []
        max_label = 0  # Initialize max_label for label uniqueness

        for idx, (tile_segmentation, (y, x)) in enumerate(zip(segmentation_tiles, positions)):
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
        return confidence_segmentation_tiles
