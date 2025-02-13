import numpy as np
import cv2
from src.utils.histogram_utils import calculate_histogram, compare_histograms


class Bucket:
    def __init__(self, bucket_id):
        """
        Initialises a Bucket object that stores multiple ProcessedImage objects.
        Maintains incremental histogram sums and counts to compute averages efficiently.
        """
        self.bucket_id = bucket_id
        self.images = []  # List of ProcessedImage objects
        self.average_histograms = {}  # Dictionary: segment key -> average histogram
        self.histogram_sums = {}  # Dictionary: segment key -> cumulative histogram sum
        self.histogram_counts = {}  # Dictionary: segment key -> number of images contributing

    def add_image(self, processed_image):
        """
        Adds a ProcessedImage to the bucket and updates the average histogram incrementally.

        Args:
            processed_image (ProcessedImage): The processed image to be added.
        """
        self.images.append(processed_image)
        self._update_histograms_for_image(processed_image)

    def _update_histograms_for_image(self, processed_image):
        """
        Incrementally updates the histogram sums, counts, and average for each image segment
        contained in a ProcessedImage.

        Args:
            processed_image (ProcessedImage): The processed image whose segments are to be added.
        """

        if not processed_image.image_segments:
            return

        # Loop over each segment provided in image_segments
        for segment_key, segment_image in processed_image.image_segments.items():
            # Convert the PIL image to a NumPy array (BGR format for OpenCV)
            segment_bgr = cv2.cvtColor(np.asarray(segment_image), cv2.COLOR_RGB2BGR)
            hist = calculate_histogram(segment_bgr)

            if segment_key in self.histogram_sums:
                self.histogram_sums[segment_key] += hist
                self.histogram_counts[segment_key] += 1
            else:
                # Use a copy to avoid modifying the original histogram later
                self.histogram_sums[segment_key] = hist.copy()
                self.histogram_counts[segment_key] = 1

            # Update the average histogram for this segment
            segment_channels_avg = []
            for channel in self.histogram_sums[segment_key]:
                segment_channels_avg.append(channel / self.histogram_counts[segment_key])

            self.average_histograms[segment_key] = segment_channels_avg

    def compare_with_bucket(self, other_bucket):
        """
        Compares this bucket with another bucket by comparing the average histograms
        for each common image segment.

        Args:
            other_bucket (Bucket): Another bucket to compare with.

        Returns:
            float: The mean similarity score across all matching segments.
                   Returns float('inf') if no valid histograms exist.
        """
        if not self.average_histograms or not other_bucket.average_histograms:
            return float('inf')  # High value when one bucket has no histogram data

        similarities = []
        # missing_segments = []
        for segment_key in self.average_histograms.keys():
            if segment_key in other_bucket.average_histograms:
                similarity_score = compare_histograms(
                    self.average_histograms[segment_key],
                    other_bucket.average_histograms[segment_key]
                )
                similarities.append(similarity_score)
            else:
                # missing_segments.append(segment_key)
                print(f"Missing segmented object {segment_key}")
                # return float('inf')

        if similarities:
            return np.mean(similarities)
        else:
            return float('inf')


    def merge_bucket(self, other_bucket):
        """
        Merges another bucket into this one by extending the list of images and merging
        the histogram data (sums and counts) so that the average histograms reflect the union.

        Args:
            other_bucket (Bucket): The bucket whose images and histograms are to be merged.
        """
        # Extend images list
        self.images.extend(other_bucket.images)

        # Merge histogram sums and counts
        for segment_key, other_sum in other_bucket.histogram_sums.items():
            other_count = other_bucket.histogram_counts[segment_key]
            if segment_key in self.histogram_sums:
                self.histogram_sums[segment_key] += other_sum
                self.histogram_counts[segment_key] += other_count
            else:
                self.histogram_sums[segment_key] = other_sum.copy()
                self.histogram_counts[segment_key] = other_count

            # Update the average histogram for the merged segment
            segment_channels_avg = []
            for channel in self.histogram_sums[segment_key]:
                segment_channels_avg.append(channel / self.histogram_counts[segment_key])

            self.average_histograms[segment_key] = segment_channels_avg

    def __str__(self):
        """
        Returns a string representation of the bucket, listing the image file names.
        """
        if not self.images:
            return f"Bucket {self.bucket_id} is empty."
        image_titles = [image.image_name for image in self.images]
        return f"Bucket {self.bucket_id} contains images:\n- " + "\n- ".join(image_titles)

    def __repr__(self):
        return self.__str__()
