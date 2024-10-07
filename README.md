# SegFlow: Segmentation Workflow Tool

## Overview

**SegFlow** is a tool for segmenting image tiles extracted from OME-TIFF files. It supports tiling, segmentation, and visualization workflows with specific methods to normalize and randomize the labels of segmented images.

## Installation

To install SegFlow, run:

```bash
pip install segflow
```

## Quickstart Example

Below is an example demonstrating how to load an OME-TIFF file, segment it, and visualize the results.

### Step 1: Load OME-TIFF File

```python
from segflow import OMETiffHelper, SegFlow
import matplotlib.pyplot as plt

# Define the path to your OME-TIFF file
ome_tiff_path = 'image.ome.tiff'

# Initialize the SegFlow object with a tile size and stride
segmenter = SegFlow(tile_size=512, stride=256)

# Load the OME-TIFF file using OMETiffHelper
with OMETiffHelper(ome_tiff_path) as ome:
    print(ome)  # Prints details about the OME-TIFF file
    # Load channel data (example: nuclear channel is Channel 0)
    segmenter.load_numpy_arrays(nuclear=ome.get_channel_data(0))

# Visualize the loaded image
plt.figure(figsize=(8, 8))
plt.imshow(segmenter.image[:, :, 0], cmap="gray")
plt.title('Nuclear Channel Visualization')
plt.axis('off')
plt.show()
```

#### Expected Output

```
OME-TIFF File: image.ome.tiff
Image Dimensions: (25, 2500, 2500)
Axes: CYX
Data Type: uint16
Pixel Size X: 0.28 None
Pixel Size Y: 0.28 None
Number of Channels: 25
  Channel 1: DAPI (ID: Channel:0)
  ...
```

### Step 2: Image Preprocessing

```python
# Normalize and pad the image
segmenter.normalize_image()
segmenter.pad_image()

# Extract tiles from the image for processing
tiles, _ = segmenter.extract_tiles()
```

### Step 3: Running the Segmentation

```python
from segflow.segmentation_methods import SKWSegmentationMethod

# Define image resolution and batch size
image_mpp = 0.28  # Microns per pixel
batch_size = 64   # Number of tiles to process in each batch

# Initialize segmentation method
skws = SKWSegmentationMethod(image_mpp=image_mpp)

# Run segmentation on the image tiles
segmentation_tiles = skws.run_segmentation(tiles, batch_size=batch_size)

# Ingest the segmentation results
segmenter.ingest_tile_segmentation(segmentation_tiles)

# Crop the segmentation to remove padding
segmenter.crop_segmentation_padded()
```

### Step 4: Visualization of Segmentation

```python
# Visualize the segmented image
plt.figure(figsize=(8, 8))
plt.imshow(segmenter.segmentation)
plt.title('Segmentation with Original Labels')
plt.axis('off')
plt.show()
```

### Step 5: Randomizing Segmentation Labels

```python
# Randomize the segmentation labels for visualization
segmenter.randomize_segmentation()

# Visualize the remapped segmentation
plt.figure(figsize=(8, 8))
plt.imshow(segmenter.segmentation)
plt.title('Segmentation with Randomized Labels')
plt.axis('off')
plt.show()
```

---

## Inputs and Parameters

- **Numpy arrays** or **OME-TIFF Path**: Nuclear/Membrane masks or Paths to the OME-TIFF file (`ome_tiff_path`).
- **Tile Size**: Size of the image tiles to process (`tile_size=512`).
- **Stride**: Step size for moving the tile extraction window (`stride=256`).
- **Image Resolution (mpp)**: Resolution of the image in microns per pixel (`image_mpp=0.28`).
- **Batch Size**: Number of image tiles to process simultaneously (`batch_size=64`).

## Output

- **Segmentation Results**: Visualizations of the segmented regions with original and randomized labels.
