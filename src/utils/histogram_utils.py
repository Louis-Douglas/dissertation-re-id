import PIL
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)


def image_to_lab_average(image):
    """
    Converts an image to the LAB colour space and computes the average L, A, B values.

    Args:
        image (PIL.Image.Image): Input RGB(A) image.

    Returns:
        tuple: (L, A, B) average values of the image.
    """
    # Ensure image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert to NumPy array
    img_array = np.array(image)

    # Convert to LAB color space
    lab_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

    # Compute average LAB values
    l_mean = np.mean(lab_image[:, :, 0])
    a_mean = np.mean(lab_image[:, :, 1])
    b_mean = np.mean(lab_image[:, :, 2])

    return (l_mean, a_mean, b_mean)

# Comparison to EMD method
def compare_delta_e(image1, image2):
    """
    Computes Delta-E (CIEDE2000) colour difference between two images.

    Args:
        image1 (PIL.Image.Image): First image.
        image2 (PIL.Image.Image): Second image.

    Returns:
        float: Delta-E (CIEDE2000) colour difference (lower = more similar).
    """
    # Compute average LAB colour for both images
    lab1 = image_to_lab_average(image1)
    lab2 = image_to_lab_average(image2)

    # Convert to colormath LabColor objects
    color1 = LabColor(lab1[0], lab1[1], lab1[2])
    color2 = LabColor(lab2[0], lab2[1], lab2[2])

    # Compute Delta-E (CIEDE2000)
    delta_e = delta_e_cie2000(color1, color2)

    return delta_e


def RGBA2LAB(image):
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

def extract_weighted_color_histogram(image, bins=32):
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
    bin_edges = np.linspace(0, 256, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Build a grid of LAB bin centers.
    grid = np.stack(np.meshgrid(bin_centers, bin_centers, bin_centers, indexing='ij'), axis=-1)
    grid = grid.reshape(-1, 3)

    # Flatten the normalised histogram.
    hist_flat = hist_norm.flatten()

    # Keep only bins with nonzero weight.
    nonzero = hist_flat > 0
    weights = hist_flat[nonzero]
    coords = grid[nonzero]

    # Create the signature for EMD: first column is weight, next three are LAB coordinates.
    signature = np.column_stack((weights, coords)).astype(np.float32)
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

# Older EMD method that doesn't ignore transparent pixels, could be used for comparison
# def extract_weighted_color_histogram(image, bins=8):
#     """
#     Extracts a weighted colour histogram for use in EMD.
#
#     Args:
#         image (PIL.Image.Image): Input RGBA image.
#         bins (int): Number of bins per channel.
#
#     Returns:
#         tuple: Feature vector of colour histogram & corresponding weights.
#     """
#     # Convert BGR to LAB (better perceptual accuracy)
#     lab = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2LAB)
#
#     # Compute 3D histogram
#     hist = cv2.calcHist([lab], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
#
#     # Normalise and flatten
#     hist = cv2.normalize(hist, hist).flatten()
#
#     # Create corresponding weights
#     weights = hist / hist.sum()
#
#     return hist, weights
#
#
#
# def compare_emd(segment1, segment2):
#     """
#     Computes Earth Mover's Distance (EMD) between two colour histograms.
#
#     Args:
#         segment1 (ProcessedSegment): First segment.
#         segment2 (ProcessedSegment): Second segment.
#
#     Returns:
#         float: EMD similarity score (lower is better).
#     """
#     hist1, weights1 = extract_weighted_color_histogram(segment1.image)
#     hist2, weights2 = extract_weighted_color_histogram(segment2.image)
#
#
#     # Convert histograms into signature format for EMD
#     sig1 = np.column_stack((weights1, hist1))
#     sig2 = np.column_stack((weights2, hist2))
#
#     # Check for empty histograms
#     if np.sum(sig1[:, 0]) == 0 or np.sum(sig2[:, 0]) == 0:
#         return float("inf")  # Max distance if either histogram is empty
#
#     # Compute EMD (lower = more similar)
#     return cv2.EMD(sig1.astype(np.float32), sig2.astype(np.float32), cv2.DIST_L2)[0]

    # hist1, weights1 = extract_lab_ab_histogram(segment1.image, segment1.yolo_mask)
    # hist2, weights2 = extract_lab_ab_histogram(segment2.image, segment2.yolo_mask)

# Modified from: https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html &
# https://pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
def calculate_histogram(image):
    """
    Computes the colour histograms for an image across the Blue, Green, and Red channels.

    Args:
        image (numpy.ndarray): The input image in BGR format (as loaded by OpenCV).

    Returns:
        list of numpy.ndarray: A list containing three histograms, one for each colour channel (B, G, R).
    """
    histograms = []
    colours = ("b", "g", "r")
    for channel, colour in enumerate(colours):
        hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
        histograms.append(hist)
    return histograms

def visualise_histogram(histograms, title="Histogram"):
    """
    Visualises the inputted histograms using matplotlib.

    Args:
        histograms (list of numpy arrays): List of histograms that each contain three channels (B, G, R).
        title (str): Title of the histogram plot.
    """
    plt.figure(figsize=(8, 6))

    # Plot each histogram with its corresponding colour
    for hist, color in zip(histograms, ('b', 'g', 'r')):
        plt.plot(hist, color=color, label=f"{color.upper()} channel")
        plt.xlim([0, 256])  # Intensity range for 8-bit images

    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.show()


def save_histogram(histograms, save_path="histogram_plot.png", title="Histogram"):
    """
    Saves a visualised histogram plot as an image.

    Args:
        histograms (list of numpy arrays): List of histograms that each contain three channels (B, G, R).
        save_path (str): Path to save the histogram plot.
        title (str): Title of the histogram plot.
    """
    plt.figure(figsize=(8, 6))

    # Plot each histogram with its corresponding color
    for hist, color in zip(histograms, ('b', 'g', 'r')):
        plt.plot(hist, color=color, label=f"{color.upper()} channel")
        plt.xlim([0, 256])  # Intensity range for 8-bit images

    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free memory

    print(f"Histogram saved to {save_path}")


def normalise_histogram(hist):
    """
    Normalises a histogram using L1 normalisation.
    This is required for comparing histograms especially when image size differs.

    Args:
        hist (numpy.ndarray): Input histogram.

    Returns:
        numpy.ndarray: Normalised histogram.
    """

    # Ensure all values in the histogram are stored as float32 (required for normalisation)
    hist = hist.astype(np.float32)


    hist = cv2.normalize(hist, None, alpha=0, beta=1, norm_type=cv2.NORM_L1)
    return hist

# Modified from: https://www.geeksforgeeks.org/normalize-an-image-in-opencv-python/
# Todo: handle converting from PIL image to array and back inside this function
def normalise_image(image):
    """
    Normalises the brightness of an image by converting it to the HSV colour space,
    applying min-max normalisation to the brightness channel, and converting it back to BGR.

    Args:
        image (numpy.ndarray): The input image in BGR format (as loaded by OpenCV).

    Returns:
        numpy.ndarray: The normalised image in BGR format.
    """
    # Convert the image to HSV colour space to obtain the brightness channel
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Normalise the brightness channel
    hsv_image[:, :, 2] = cv2.normalize(hsv_image[:, :, 2], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Equalise the brightness channel
    # hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])

    # Convert back to BGR colour space
    normalised_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return normalised_image


def equalise_image(image):
    """
    Equalise the brightness of an image

    Args:
        image (numpy.ndarray): The input image in BGR format (as loaded by OpenCV).

    Returns:
        numpy.ndarray: The normalised image in BGR format.
    """
    # Convert the image to HSV colour space to obtain the brightness channel
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Equalise the brightness channel
    hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])

    # Convert back to BGR colour space
    eq_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return eq_image


def apply_clahe(image):
    """
    Applies CLAHE (Adaptive Histogram Equalisation) to the L channel of a LAB image.

    Args:
        image (numpy.ndarray): Input BGR image.

    Returns:
        numpy.ndarray: Image with equalised brightness while preserving colours.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    lab[:, :, 0] = clahe.apply(lab[:, :, 0])  # Apply CLAHE to L channel

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # Convert back to BGR
