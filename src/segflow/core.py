import numpy as np
from tqdm import tqdm
from .continuoussinglechannelimage import ContinuousSingleChannelImage
from .segmentationimage import SegmentationImage

class SegFlow:
    def __init__(self, tile_size=512, stride=256, average_weight=0.7, sum_weight=0.3, min_pixels=5):
        """
        Initialize the Segment class with parameters for image tiling.
        
        Parameters:
        - tile_size: Size of the tiles to extract from the image.
        - stride: Stride size for tiling the image.
        - average_weight: When choosing conflicting overlapping segments, how to weight the average
        - sum_weight: When choosing conflicting overlapping segments, how to weight the sum of pixels
        - min_pixels: The minimum number of pixels considered when combining overlaps
        """
        self.tile_size = tile_size
        self.stride = stride
        self.average_weight = 0.7
        self.sum_weight = 0.3
        self.min_pixels = 5

        self.image = None
        self.image_padded = None
        self.pad_top = None
        self.pad_bottom = None
        self.pad_left = None
        self.pad_right = None
        self.tiles = None
        self.positions = None

    def load_numpy_arrays(self, nuclear, membrane=None):
        """
        Load the image data from numpy arrays for nuclear and membrane channels.
        
        Parameters:
        - nuclear: Numpy array for the nuclear channel.
        - membrane: Optional numpy array for the membrane channel. If not provided, the nuclear channel will be duplicated.
        """
        if membrane is None:
            membrane = nuclear.copy()
        self.image = np.stack([nuclear, membrane], axis=-1)
        print(f"Loaded numpy arrays with shape: {self.image.shape}")

    def normalize_image(self):
        """
        Normalize each channel of the loaded image separately to have zero mean and unit variance.
        """
        if self.image is not None and len(self.image.shape) == 3:
            normalized_image = np.zeros_like(self.image, dtype=np.float32)
            for channel in range(self.image.shape[-1]):
                channel_data = self.image[:, :, channel]
                channel_mean = np.mean(channel_data)
                channel_std = np.std(channel_data)
                normalized_image[:, :, channel] = (channel_data - channel_mean) / channel_std
            self.image = normalized_image
            print("Normalized image shape:", self.image.shape)

    def pad_image(self):
        """
        Apply reflection padding to the image to ensure that tiling covers the entire image without leaving out regions.
        """
        if self.image is not None:
            height, width = self.image.shape[:2]
            pad_height_total = self.tile_size + ((self.tile_size - (height - self.tile_size) % self.stride) % self.stride)
            pad_width_total = self.tile_size + ((self.tile_size - (width - self.tile_size) % self.stride) % self.stride)
            
            self.pad_top = pad_height_total // 2
            self.pad_bottom = pad_height_total - self.pad_top
            self.pad_left = pad_width_total // 2
            self.pad_right = pad_width_total - self.pad_left
            
            self.image_padded = np.pad(
                self.image,
                ((self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0)),
                mode='reflect'
            )
            print(f"Padding applied - Top: {self.pad_top}, Bottom: {self.pad_bottom}, Left: {self.pad_left}, Right: {self.pad_right}")
            print(f"Padded image size: height={self.image_padded.shape[0]}, width={self.image_padded.shape[1]}")

    def extract_tiles(self):
        """
        Extract overlapping tiles from the padded image for segmentation.
        
        Returns:
        - tiles: Numpy array of extracted image tiles.
        - positions: List of positions where each tile starts in the original image.
        """
        if self.tiles is not None and self.positions is not None:
            return self.tiles, self.positions
        self.tiles, self.positions = _extract_tiles(self.image_padded, self.tile_size, self.stride)
        return self.tiles, self.positions

    def combine_continuous_tiles(self, tiles):
        """
        Ingest the tiles and return the full image
        """
        full_image_shape = self.image_padded.shape[:2]
        reconstructed_image = np.zeros(full_image_shape, dtype=tiles[0].dtype)
        weight_matrix = np.zeros(full_image_shape, dtype=np.float32)

        # Create a weighting window to reduce edge effects
        window = np.outer(np.hanning(self.tile_size), np.hanning(self.tile_size))
        window = window / window.max()

        for tile, (row_start, col_start) in zip(tiles, self.positions):
            row_end = row_start + self.tile_size
            col_end = col_start + self.tile_size

            reconstructed_image[row_start:row_end, col_start:col_end] += tile * window
            weight_matrix[row_start:row_end, col_start:col_end] += window

        # Avoid division by zero
        weight_matrix[weight_matrix == 0] = 1

        reconstructed_image /= weight_matrix
        return ContinuousSingleChannelImage(self._crop_padded(reconstructed_image))

    def ingest_tile_segmentation(self, segmentation_tiles, iou_threshold=0.5):
        """
        Ingest the segmented tiles and recombine them into a full segmentation mask, handling overlaps.

        Parameters:
        - segmentation_tiles: Numpy array of segmented tiles.
        - iou_threshold: Threshold for IoU to decide if segments represent the same cell.
        """
        if self.positions is None:
            raise ValueError("Tiles must be extracted first")

        # Process tiles to get high-confidence central regions
        segmentation_tiles = self._calculate_high_confidence_tiles(segmentation_tiles)

        full_segmentation_mask = np.zeros(self.image_padded.shape[:2], dtype=np.int32)
        max_global_label = 0

        for idx, (tile_segmentation, (y, x)) in tqdm(enumerate(zip(segmentation_tiles, self.positions)),desc="Processing tiles",total=len(self.positions)):
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

            #print(f"Processed tile {idx + 1}/{len(segmentation_tiles)}", end='\r')
        #print("\nRecombination complete.")

        # Optional: Relabel connected components
        from skimage.measure import label

        full_segmentation_mask = label(full_segmentation_mask)

        # Crop the padded segmentation mask
        final_segmentation_mask = self._crop_padded(full_segmentation_mask)

        return SegmentationImage(final_segmentation_mask)


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


    def _calculate_high_confidence_tiles(self, segmentation_tiles):
        """
        Process segmentation tiles to extract high-confidence central regions and adjust labels to ensure uniqueness.
        
        Parameters:
        - segmentation_tiles: Numpy array of segmented tiles.
        
        Returns:
        - Numpy array of high-confidence segmented tiles.
        """
        # Step 7: Process segmentation tiles to focus on high-confidence regions

        # Define the margin size (e.g., 12.5% of tile_size)
        margin = self.tile_size // 8  # For 12.5% margin, 512 // 8 = 64 pixels

        confidence_segmentation_tiles = []
        max_label = 0  # Initialize max_label for label uniqueness

        for idx, (tile_segmentation, (y, x)) in enumerate(zip(segmentation_tiles, self.positions)):
            # Extract the high-confidence central region of the tile
            y_start = margin
            y_end = self.tile_size - margin
            x_start = margin
            x_end = self.tile_size - margin

            # Get labels present in the high-confidence region
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
    
    def _calculate_overlaps(self, confidence_segmentation_tiles):
        """
        Generate a full overlap mask to quantify overlaps between tiles.
        
        Parameters:
        - confidence_segmentation_tiles: Numpy array of high-confidence segmented tiles.
        
        Returns:
        - Numpy array representing the overlap mask.
        """
        # Step 8: Generate the full overlap mask to quantify overlaps between tiles

        # Initialize the full overlap mask
        full_overlap_mask = np.zeros(self.image_padded.shape[:2], dtype=np.int32)

        for idx, (tile_segmentation, (y, x)) in enumerate(zip(confidence_segmentation_tiles, self.positions)):
            # Define the region corresponding to the current tile
            y_end = y + self.tile_size
            x_end = x + self.tile_size
    
            # Extract the corresponding region from the full overlap mask
            overlap_mask_region = full_overlap_mask[y:y_end, x:x_end]
    
            # Ensure dimensions match
            if tile_segmentation.ndim == 3 and tile_segmentation.shape[-1] == 1:
                tile_segmentation = tile_segmentation.squeeze(-1)

            # Create a mask of non-zero labels in the tile
            non_zero_mask = tile_segmentation > 0

            # Identify overlapping regions
            overlap_mask = non_zero_mask & (overlap_mask_region > 0)

            # Increment overlap counts
            full_overlap_mask[y:y_end, x:x_end][overlap_mask] += 1
            full_overlap_mask[y:y_end, x:x_end][~overlap_mask & non_zero_mask] = 1

            print(f"Processed tile {idx + 1}/{len(confidence_segmentation_tiles)}", end='\r')

        print("\nFull overlap mask generated.")
        return full_overlap_mask

    def _calculate_tile_overlaps(self, full_overlap_mask, confidence_segmentation_tiles):
        """
        Calculate index coverage for each cell in each tile based on overlap information.
        
        Parameters:
        - full_overlap_mask: Numpy array representing the full overlap mask.
        - confidence_segmentation_tiles: Numpy array of high-confidence segmented tiles.
        
        Returns:
        - List of dictionaries with coverage scores for each tile.
        """
        # Step 9: Calculate index coverage for each cell in each tile
        # Extract overlap mask tiles corresponding to segmentation tiles
        overlap_mask_tiles, _ = _extract_tiles(full_overlap_mask, self.tile_size, self.stride)
        
        # Constants for scoring
        w1 = self.average_weight # Weight for average overlap
        w2 = self.sum_weight # Weight for sum overlap
        min_pixels = self.min_pixels # Minimum pixels for a cell to be considered

        index_coverage_tiles = []

        for tile_segmentation, overlap_tile in zip(confidence_segmentation_tiles, overlap_mask_tiles):
            # Ensure tile_segmentation is 2D
            if tile_segmentation.ndim > 2:
                tile_segmentation = tile_segmentation.squeeze()

            # Dictionary to store scores for each cell label
            tile_index_coverage = {}

            # Find unique cell labels
            unique_labels = np.unique(tile_segmentation)
            unique_labels = unique_labels[unique_labels != 0]

            for label in unique_labels:
                # Mask for the current cell
                label_mask = (tile_segmentation == label)

                # Get overlap values for this cell
                overlap_values = overlap_tile[label_mask]
                num_pixels = np.sum(label_mask)

                # Consider cells with sufficient size
                if num_pixels >= min_pixels:
                    # Calculate average and sum of overlap
                    avg_overlap = np.mean(overlap_values)
                    sum_overlap = np.sum(overlap_values)

                    # Combine metrics into a score
                    score = w1 * avg_overlap + w2 * sum_overlap

                    # Store the score
                    tile_index_coverage[label] = score

            # Append the index coverage dictionary for the tile
            index_coverage_tiles.append(tile_index_coverage)
        print("Index coverage (with combined score) for each tile calculated.")
        return index_coverage_tiles


    def _crop_padded(self,padded_image_to_crop):
        """
        Remove the padding from the full segmentation mask to obtain the final segmentation mask.
        """
        if self.image_padded is None:
            raise ValueError("Must have padded image first.")
        # Step 11: Remove padding to obtain the final segmentation mask

        # Crop the full segmentation mask to remove the padding
        cropped_segmentation_mask = padded_image_to_crop[self.pad_top:-self.pad_bottom, self.pad_left:-self.pad_right]

        # Handle edge cases where padding amounts are zero
        if self.pad_bottom == 0:
            cropped_segmentation_mask = padded_image_to_crop[self.pad_top:, self.pad_left:-self.pad_right]
        if self.pad_right == 0:
            cropped_segmentation_mask = cropped_segmentation_mask[:, self.pad_left:]

        # Print dimensions of the cropped mask
        print(f"Cropped segmentation mask size: height={cropped_segmentation_mask.shape[0]}, width={cropped_segmentation_mask.shape[1]}")
        return cropped_segmentation_mask

def _extract_tiles(image, tile_size, stride):
    """
    Extract overlapping tiles from the given image.
    
    Parameters:
    - image: Numpy array of the image to extract tiles from.
    - tile_size: Size of each tile.
    - stride: Stride for extracting tiles.
    
    Returns:
    - tiles: Numpy array of extracted tiles.
    - positions: List of positions where each tile starts in the image.
    """
    tiles = []
    positions = []
    height, width = image.shape[:2]
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            # Extract tile
            if len(image.shape) == 3:
                tile = image[y:y+tile_size, x:x+tile_size, :]
            else:
                tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((y, x))
    return np.array(tiles), positions
