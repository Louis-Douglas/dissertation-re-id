import numpy as np
from Histogram.utils import calculate_histogram, compare_histograms
import cv2

class Bucket:
    def __init__(self, bucket_id):
        """
        Initialises a Bucket object that stores multiple ProcessedImage objects.

        Args:
            bucket_id (int): Unique identifier for the bucket.
        """

        self.bucket_id = bucket_id
        self.images = []  # List of ProcessedImage objects
        self.average_histograms = {}  # Dictionary storing average histograms for sections

    def add_image(self, processed_image):
        """
        Adds a ProcessedImage to the bucket and updates the average histogram.

        Args:
            processed_image (ProcessedImage): The processed image to be added.
        """
        self.images.append(processed_image)
        self.update_average_histogram()

    def update_average_histogram(self):
        """
        Computes the average histogram for each section of all images in the bucket.
        """
        if not self.images:
            return

        # Todo: Refactor this DRY
        if len(self.images) == 1:
            # histogram_sections = {}
            for section_number, section in self.images[0].image_sections.items():
                # Convert PIL image to NumPy Array image
                section_bgr = cv2.cvtColor(np.asarray(section), cv2.COLOR_RGB2BGR)
                hist = calculate_histogram(section_bgr)
                # visualise_histogram(hist, title=f"Image {self.images[0].image_name} - Section {section_number}")
                # histogram_sections[section_number] = hist
                # self.average_histograms[section_number] = histogram_sections
                self.average_histograms[section_number] = hist
            return

        # Dictionary to collect histograms for each section number across all images.
        section_histograms = {}
        for image in self.images:
            for section_number, section in image.image_sections.items():
                # Convert PIL image to NumPy Array image
                section_bgr = cv2.cvtColor(np.asarray(section), cv2.COLOR_RGB2BGR)
                hist = calculate_histogram(section_bgr)
                if section_number not in section_histograms:
                    section_histograms[section_number] = []
                section_histograms[section_number].append(hist)

        # Compute the average histogram for each section using np.mean.
        for section_number, hist_list in section_histograms.items():
            average_section_hist = np.mean(hist_list, axis=0)
            # print("av: ", average_section_hist)
            # visualise_histogram(average_section_hist, title=f"Images averaged Section {section_number}")
            self.average_histograms[section_number] = average_section_hist

    def compare_with_bucket(self, other_bucket):
        """
        Compares this bucket with another bucket using histogram comparison.

        Args:
            other_bucket (Bucket): Another bucket to compare histograms with.

        Returns:
            float: Average similarity score across all sections.
        """
        if not self.average_histograms or not other_bucket.average_histograms:
            return float('inf')  # If no histograms, return a high value (no similarity)

        similarities = []
        for section in self.average_histograms.keys():
            if section in other_bucket.average_histograms:
                similarity_score = compare_histograms(
                    self.average_histograms[section],
                    other_bucket.average_histograms[section]
                )
                similarities.append(similarity_score)

        if similarities:
            return np.mean(similarities)
        else:
            return float('inf')


    def merge_bucket(self, other_bucket):
        """
        Merges another bucket into this bucket.
        This function adds all ProcessedImage objects from the other bucket into this one
        and updates the average histograms accordingly.

        Args:
            other_bucket (Bucket): The bucket to merge into this one.
        """
        # Extend the list of images with images from the other bucket.
        self.images.extend(other_bucket.images)
        # Recompute the average histograms to reflect the merged images.
        self.update_average_histogram()
