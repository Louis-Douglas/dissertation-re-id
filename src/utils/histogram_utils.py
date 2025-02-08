import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Modified from: https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
def compare_histograms(hist1, hist2):
    """
    Compares two color histograms using the Bhattacharyya distance.

    Args:
        hist1 (list of numpy.ndarray): The first histogram, a list of three histograms (one for each B, G, R channel).
        hist2 (list of numpy.ndarray): The second histogram, structured the same way as hist1.

    Returns:
        float: The mean Bhattacharyya distance between the corresponding colour channels of the two histograms.
               Lower values indicate higher similarity.

    Example:
        >>> hist1 = calculate_histogram(image1)
        >>> hist2 = calculate_histogram(image2)
        >>> similarity_score = compare_histograms(hist1, hist2)
        >>> print(f"Similarity Score: {similarity_score}")
    """
    similarities = []
    for h1, h2 in zip(hist1, hist2):
        similarity = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
        similarities.append(similarity)
    return np.mean(similarities)

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

    # Convert back to BGR colour space
    normalised_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return normalised_image
