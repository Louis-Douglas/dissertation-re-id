from Histogram.utils import calculate_histogram, image_split, compare_histograms, visualise_histogram
import os
import cv2
from PIL import Image
import numpy as np

# Function to get all images in a directory
def get_images(input_dir = "images"):
    images = {}
    for file in os.listdir(input_dir):
        # Construct file path and load image
        file_path = os.path.join(input_dir, file)
        file_extension = os.path.splitext(file)[1]
        if file_extension == ".jpg":
            image = Image.open(file_path)
            # image = cv2.imread(file_path) # (B G R) Image
            images[file.title()] = image
    return images


def calculate_buckets(image_group_list):
    buckets = {}
    bucket_id = 1  # Arbitrary ID for buckets

    while image_group_list:
        # print(image_group_list)
        # Take the first image group from the list
        current_image_group = image_group_list.popitem()
        print(f"Popped off: {current_image_group[0]}")
        # print(f"Popped off: {current_image_group[1][2]}")

        # Initialise variables to track the most similar match
        best_match = None
        best_match_index = None
        best_similarity = float("inf")

        # Compare the current image histogram with all remaining images
        for image_group_title, image_group in image_group_list.items():

            print(f"Checking {current_image_group[0]} against {image_group_title}")
            total_similarity = 0.0

            # Loop through each section of the image, comparing its similarity to the current image groups section
            for section_number, section in image_group.items():
                compare_other = current_image_group[1][section_number]
                similarity_score = compare_histograms(section, compare_other)
                total_similarity += similarity_score
                print(f"Section {section_number} similarity: {similarity_score}")
            total_similarity /= len(image_group)
            if total_similarity < best_similarity:
                best_similarity = total_similarity
                best_match = image_group
                best_match_index = image_group_title
            print("--------")

        image_group_list.pop(best_match_index)
        print(f"Best match for {current_image_group[0]} - {best_match_index}")


def main():
    images = get_images()

    # Todo: Figure out the right data structure to hold the 4 sections and all histograms
    sectioned_images = {}

    for title, image in images.items():
        image_sections = image_split(image)
        histogram_sections = {}
        for number, section in image_sections.items():
            # Convert PIL image to NumPy Array image
            section_bgr = cv2.cvtColor(np.asarray(section), cv2.COLOR_RGB2BGR)
            hist = calculate_histogram(section_bgr)
            # visualise_histogram(hist, title=f"Image {title} - Section {number}")
            histogram_sections[number] = hist
        sectioned_images[title] = histogram_sections


    # print(sectioned_images)
    # bucketed_images = calculate_buckets(sectioned_images)
    calculate_buckets(sectioned_images)



# Get all images in market dataset -
# Split the images into the chunks (image_split.py) -
# Calculate histograms for all chunks of all images (group the chunk histograms) -
# Take a histogram, compare to all other histograms, most similar one put into a dictionary -
# Generate mean histogram values from all images in bucket, for future comparisons


if __name__ == '__main__':
    main()