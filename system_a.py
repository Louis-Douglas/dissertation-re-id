from Histogram.histogram_visualisation import visualise_histogram
from Histogram.utils import calculate_histogram
import os
import cv2

# Function to get all images in a directory
def get_images(input_dir = "images"):
    images = []
    for file in os.listdir(input_dir):
        # Construct file path and load image
        file_path = os.path.join(input_dir, file)
        file_extension = os.path.splitext(file)[1]
        if file_extension == ".jpg":
            image = cv2.imread(file_path)
            images.append(image)
    return images

def main():
    images = get_images()
    for image in images:
        hist = calculate_histogram(image)
        visualise_histogram(hist)

# Get all images in market dataset
# Split the images into the chunks (image_split.py)
# Calculate histograms for all chunks of all images (group the chunk histograms)
# Take a histogram, compare to all other histograms, most similar one put into a dictionary,
# Generate mean histogram values from all images in bucket, for future comparisons


if __name__ == '__main__':
    main()