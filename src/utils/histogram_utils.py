import cv2
import numpy as np
import matplotlib.pyplot as plt

def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)

def extract_lab_ab_histogram(image, mask, bins=8):
    """
    Extracts a weighted LAB histogram using only A and B channels to ignore brightness variations.

    Args:
        image (PIL.Image.Image): Input RGBA image.
        mask (numpy.ndarray): Binary mask indicating valid pixels.
        bins (int): Number of bins per channel.

    Returns:
        tuple: Histogram of A/B channels & corresponding weights.
    """
    # Convert PIL Image to NumPy array
    img_arr = np.array(image, dtype=np.uint8)  # Shape: (H, W, 4) (RGBA)

    # Extract RGB channels and alpha channel
    rgb_image = img_arr[:, :, :3]  # Shape: (H, W, 3)
    alpha_channel = img_arr[:, :, 3]  # Shape: (H, W) - Alpha values

    # Create a binary mask for non-transparent pixels (alpha > 0)
    mask = alpha_channel > 0
    img_arr = np.array(rgb_image, dtype=np.uint8)  # Convert to RGB
    masked_rgb = img_arr[mask]  # Extract only non-transparent pixels

    if masked_rgb.size == 0:
        return np.zeros((bins ** 2,)), np.zeros((bins ** 2,))  # Return empty

    lab_pixels = cv2.cvtColor(masked_rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)

    # Extract only A/B channels, ignore L
    ab_pixels = lab_pixels[:, 1:3]

    # Compute 2D histogram for A/B channels
    hist = cv2.calcHist([ab_pixels.astype(np.float32)], [0, 1], None, [bins, bins], [0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    weights = hist / hist.sum()

    return hist, weights


def extract_weighted_color_histogram(image, bins=8):
    """
    Extracts a weighted colour histogram for use in EMD.

    Args:
        image (PIL.Image.Image): Input BGR image.
        bins (int): Number of bins per channel.

    Returns:
        tuple: Feature vector of colour histogram & corresponding weights.
    """
    # Convert BGR to LAB (better perceptual accuracy)
    lab = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2LAB)

    # Compute 3D histogram
    hist = cv2.calcHist([lab], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])

    # Normalize and flatten
    hist = cv2.normalize(hist, hist).flatten()

    # Create corresponding weights
    weights = hist / hist.sum()

    return hist, weights

def compare_emd(segment1, segment2):
    """
    Computes Earth Mover's Distance (EMD) between two colour histograms.

    Args:
        segment1 (ProcessedSegment): First segment.
        segment2 (ProcessedSegment): Second segment.

    Returns:
        float: EMD similarity score (lower is better).
    """
    # hist1, weights1 = extract_weighted_color_histogram(segment1.image)
    # hist2, weights2 = extract_weighted_color_histogram(segment2.image)
    hist1, weights1 = extract_lab_ab_histogram(segment1.image, segment1.yolo_mask)
    hist2, weights2 = extract_lab_ab_histogram(segment2.image, segment2.yolo_mask)

    # Convert histograms into signature format for EMD
    sig1 = np.column_stack((weights1, hist1))
    sig2 = np.column_stack((weights2, hist2))

    # Check for empty histograms
    if np.sum(sig1[:, 0]) == 0 or np.sum(sig2[:, 0]) == 0:
        return float("inf")  # Max distance if either histogram is empty

    # Compute EMD (lower = more similar)
    return cv2.EMD(sig1.astype(np.float32), sig2.astype(np.float32), cv2.DIST_L2)[0]


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
    Applies CLAHE (Adaptive Histogram Equalization) to the L channel of a LAB image.

    Args:
        image (numpy.ndarray): Input BGR image.

    Returns:
        numpy.ndarray: Image with equalized brightness while preserving colours.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    lab[:, :, 0] = clahe.apply(lab[:, :, 0])  # Apply CLAHE to L channel

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # Convert back to BGR
