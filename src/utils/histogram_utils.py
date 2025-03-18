import cv2
import numpy as np
from PIL import Image

# Fixes a colormatch bug
def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)


def RGBA2LAB(image):
    """
    Converts a PIL image to LAB color space, ignoring fully transparent pixels.

    Args:
        image (PIL.Image.Image): Input image in RGBA format.

    Returns:
        np.ndarray: LAB representation of non-transparent pixels with shape (-1, 1, 3),
                    or an empty array if all pixels are transparent.
    """
    # Ensure the image is in RGBA mode.
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # Convert PIL image to NumPy array (shape: H x W x 4)
    img_array = np.array(image)

    # Create a mask for non-transparent pixels (alpha channel != 0)
    mask = img_array[..., 3] != 0

    # If all pixels are transparent, return an empty signature
    if not np.any(mask):
        return np.empty((0, 4), dtype=np.float32)

    # Use mask to filter to only array values that are non-transparent
    # Use array slicing to remove the Alpha values from the array, leaving just RGB
    # (PIL images in RGBA are ordered as R, G, B, A)
    rgb_pixels = img_array[mask][:, :3]

    # cv2.cvtColor expects an image with shape (H, W, 3), so reshape to (-1, 1, 3)
    rgb_pixels = rgb_pixels.reshape(-1, 1, 3)

    # Convert the RGB pixels to LAB.
    lab_pixels = cv2.cvtColor(rgb_pixels, cv2.COLOR_RGB2LAB).reshape(-1, 3)

    # Reshape lab_pixels to mimic an image (each pixel as one row)
    lab_image = lab_pixels.reshape(-1, 1, 3)

    return lab_image

def extract_weighted_color_histogram(image, bins=8):
    """
    Extracts a weighted LAB colour histogram signature for use in EMD.
    Only non-transparent pixels (alpha > 0) are considered.

    Args:
        image (PIL.Image.Image): Input RGBA image.
        bins (int): Number of bins per channel.

    Returns:
        np.ndarray: Signature array of shape (n, 4) where the first column is the
                    normalised weight and the next three columns are the LAB bin centers.
    """
    # Convert the RGBA image to lab image format
    lab_image = RGBA2LAB(image)

    # Check if lab_image is empty (no non-transparent pixels)
    if lab_image.size == 0:
        return np.empty((0, 4), dtype=np.float32) # Return empty signature

    # Compute the 3D histogram over LAB space.
    hist = cv2.calcHist([lab_image], channels=[0, 1, 2], mask=None,
                        histSize=[bins, bins, bins], ranges=[0, 256, 0, 256, 0, 256])


    total = hist.sum()
    # If histogram is empty, return empty array
    if total == 0:
        return np.empty((0, 4), dtype=np.float32)
    # Normalise the histogram
    hist_norm = hist / total

    # Create bin centers for each channel.
    # Example: [  0,  32,  64,  96, 128, 160, 192, 224, 256]
    # Bin 1: 0 → 32
    bin_edges = np.linspace(0, 256, bins + 1) # Create each edge for each bin, creating 9 evenly spaced points

    # Get center for each of these bins
    # Example: Bin 1: Center = (0 + 32) / 2 = 16
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Create a grid of points in 3D space
    mesh_grid = np.meshgrid(bin_centers, bin_centers, bin_centers, indexing='ij')

    # Build a grid of LAB bin centers.
    grid = np.stack(mesh_grid, axis=-1) # Combines 3 sets of bins into stack representing L,A,B
    grid = grid.reshape(-1, 3) # Flatten grid to list of LAB coordinates


    # Flatten the normalised histogram into a 1D list of bin frequencies.
    hist_flat = hist_norm.flatten()

    # Keep only bins with nonzero weight.
    nonzero = hist_flat > 0
    weights = hist_flat[nonzero]
    coords = grid[nonzero]

    # Create the signature for EMD: first column is weight, next three are LAB coordinates.
    signature = np.column_stack((weights, coords)).astype(np.float32) # Creates a single matrix
    return signature


def compare_emd(segment1, segment2):
    """
    Computes Earth Mover's Distance (EMD) between two colour histograms.
    Uses the weighted LAB colour histogram signature computed from non-transparent pixels.

    Args:
        segment1 (ProcessedSegment): First segment with an attribute 'image' (PIL.Image.Image).
        segment2 (ProcessedSegment): Second segment.

    Returns:
        float: EMD similarity score (lower is more similar).
    """

    sig1 = extract_weighted_color_histogram(segment1.image)
    sig2 = extract_weighted_color_histogram(segment2.image)

    # If either signature is empty, return infinity (max distance).
    if sig1.shape[0] == 0 or sig2.shape[0] == 0:
        return float("inf")

    # Compute the Earth Mover's Distance.
    emd_value = cv2.EMD(sig1, sig2, cv2.DIST_L2)[0]
    return emd_value

def apply_clahe(image):
    """
    Applies CLAHE (Adaptive Histogram Equalisation) to the L channel of a LAB image.

    Args:
        image (PIL.Image.Image): Input BGR image.

    Returns:
        PIL.Image.Image: Image with equalised brightness while preserving colours.
    """
    image = np.array(image) # Convert to BGR array

    # Convert image to LAB colour space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Enhances contrast locally by dividing the image into small regions.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) # cliplimit prevents over-brightening

    lab[:, :, 0] = clahe.apply(lab[:, :, 0])  # Apply CLAHE to L channel
    bgr_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) # Convert back to BGR

    return Image.fromarray(bgr_image) # Convert back to PIL.Image
