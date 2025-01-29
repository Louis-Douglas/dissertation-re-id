import cv2
import numpy as np
from torchvision.ops import box_iou, distance_box_iou
import matplotlib.pyplot as plt

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

def image_split(image):
    """
    Splits an image into four equal horizontal sections.

    Args:
        image (PIL.Image.Image): The input image to be split.

    Returns:
        dict: A dictionary where the keys (1 to 4) represent section numbers,
              and the values are the corresponding cropped image sections (PIL.Image.Image).
    """
    width, height = image.size
    section_height = height // 4
    image_sections = {}

    for i in range(4):
        top = i * section_height  # Get the top y value of the current section
        bottom = (i + 1) * section_height  # Get the bottom y value of the current section
        cropped_section = image.crop((0, top, width, bottom))  # Crop to those y values
        image_sections[i + 1] = cropped_section

    return image_sections

# TODO: Setup unit test for if objects are connected and test against these thresholds to help fine tune
def is_connected(box1, box2, iou_threshold=0.1, distance_threshold=0.1):
    """
    Determines whether two bounding boxes are considered connected based on
    Intersection over Union (IoU) and center distance.

    Args:
        box1 (Tensor array): The first bounding box in the format (x1, y1, x2, y2).
        box2 (Tensor array): The second bounding box in the same format as box1.
        iou_threshold (float, optional): The minimum IoU required for boxes to be considered connected.
                                         Defaults to 0.1.
        distance_threshold (float, optional): The maximum center distance allowed for boxes to be considered
                                              connected. Defaults to 0.1.

    Returns:
        bool: True if the boxes are considered connected, False otherwise.
    """
    # Get intersection over union of the two boxes
    iou = box_iou(box1, box2)

    # Calculate center distance between two boxes
    distance = distance_box_iou(box1, box2)

    # Check IoU and distance thresholds
    return iou > iou_threshold or distance < distance_threshold

def visualise_histogram(histograms, title="Histogram"):
    """
    Visualises the inputted histograms using matplotlib.

    Args:
        histograms (list of numpy arrays): List of histograms that each contain three channels (B, G, R).
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
    plt.show()
