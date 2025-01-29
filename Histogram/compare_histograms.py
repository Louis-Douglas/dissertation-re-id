# https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
# https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
# https://pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/

import cv2
import matplotlib.pyplot as plt
from utils import calculate_histogram, compare_histograms

# Load the two images
image1_path = "compare-images/1.png"
image2_path = "compare-images/2.png"

# Open images in BGR format
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Calculate histograms
hist1 = calculate_histogram(image1)
hist2 = calculate_histogram(image2)

# Compare histograms and get similarity score
similarity = compare_histograms(hist1, hist2)
# 0 = Perfectly similar, 1 = No similarity
print(f"Histogram similarity: {similarity:.2f}")

# Plot histograms for both images (Use for debugging)
colors = ("b", "g", "r")
plt.figure(figsize=(12, 6))
for i, color in enumerate(colors):
    plt.subplot(2, 1, 1)
    plt.plot(hist1[i], color=color)
    plt.title("Image 1 Color Histogram")
    plt.xlim([0, 256])

    plt.subplot(2, 1, 2)
    plt.plot(hist2[i], color=color)
    plt.title("Image 2 Color Histogram")
    plt.xlim([0, 256])

plt.tight_layout()
plt.show()