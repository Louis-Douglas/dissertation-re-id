import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.utils.histogram_utils import compare_emd
from src.utils.file_ops import ensure_directory_exists, save_comparison_image
from src.core.processed_segment import ProcessedSegment
from typing import List
from memory_profiler import profile

class ProcessedImage:
    def __init__(self, image_path, original_image, processed_segments: List[ProcessedSegment]):
        """
        Initialises a ProcessedImage object.

        Args:
            image_path (str): Path to the image file.
            original_image (PIL.Image.image): Original image.
            List[ProcessedSegment]: A list of segments extracted from the subject.
        """
        self.image_path = image_path
        self.image_name = os.path.splitext(os.path.basename(image_path))[0]
        self.original_image = original_image  # Store the extracted person image
        self.processed_segments = processed_segments  # List of segment objects

    def sort_processed_segments(self, segments: List[ProcessedSegment]):
        """
        Initialises a ProcessedImage object.

        Args:
            List[ProcessedSegment]: A list of segments extracted from the subject.

        Returns:
            Dict[str, ProcessedSegment]: A dictionary of class to segment.
        """
        segments_by_class = {}
        for segment in segments:
            class_name = segment.class_name
            if class_name not in segments_by_class:
                segments_by_class[class_name] = []  # Initialise as list

            # Append segmented image to the list
            segments_by_class[class_name].append(segment)
        return segments_by_class

    @profile
    def compare_processed_image(self, other_image, penalty=10.0, save_path="../logs/comparison_results",
                                save_logs=False):
        """
        Compares this ProcessedImage with another using a Hungarian matching scheme over object images.

        Additionally:
          - Saves images being compared into directories.
          - Logs similarity scores for matched objects.

        Args:
            other_image (ProcessedImage): The other image to compare with.
            penalty (float): Cost penalty per unmatched object.
            save_path (str): Directory to save compared images.
            save_logs (bool): Whether to save logs.

        Returns:
            float: Overall similarity score.
        """
        segment_similarities = []

        # Initialise log messages.
        log_messages = []
        if save_logs:
            # Create comparison directory
            image_comparison_dir = os.path.join(save_path, f"compare_{self.image_name}_vs_{other_image.image_name}")
            ensure_directory_exists(image_comparison_dir)
            # Append intro
            log_messages.append(f"Comparison between {self.image_name} and {other_image.image_name}")
            log_messages.append("=" * 50)

        sorted_segments1 = self.sort_processed_segments(self.processed_segments)
        sorted_segments2 = self.sort_processed_segments(other_image.processed_segments)

        for class_name in sorted_segments1.keys():
            if class_name in sorted_segments2.keys():
                segment_list1 = sorted_segments1[class_name]
                segment_list2 = sorted_segments2[class_name]
                m, n = len(segment_list1), len(segment_list2)
                cost_matrix = np.zeros((m, n))

                for i, segment1 in enumerate(segment_list1):
                    for j, segment2 in enumerate(segment_list2):
                        cost_matrix[i, j] = compare_emd(segment1, segment2)

                # Solve using Hungarian Algorithm for minimum cost matches
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                total_cost = cost_matrix[row_ind, col_ind].sum()

                # Apply penalty for unmatched objects
                unmatched = abs(m - n)
                total_cost += penalty * unmatched
                avg_cost = total_cost / len(row_ind)
                segment_similarities.append(avg_cost)

                if save_logs:
                    # Create class-specific directory in the comparison folder.
                    class_dir = os.path.join(image_comparison_dir, class_name)
                    ensure_directory_exists(class_dir)

                for i, j in zip(row_ind, col_ind):
                    similarity_score = round(float(cost_matrix[i, j]), 4)
                    if save_logs:
                        log_messages.append(
                            f"Matched {class_name} {i} ({self.image_name}) with {class_name} {j} "
                            f"({other_image.image_name}): Similarity = {similarity_score}"
                        )
                        # Save matched segmented images only if logging is enabled.
                        save_comparison_image(
                            segment_list1[i].image,
                            segment_list2[j].image,
                            class_name,
                            os.path.join(class_dir, f"{self.image_name}_{i}_vs_{other_image.image_name}_{j}_.png"),
                            similarity_score,
                            self.image_name,
                            other_image.image_name
                        )
                if save_logs:
                    log_messages.append("")

        # Compute final similarity score and log final message.
        if segment_similarities:
            overall_similarity = np.mean(segment_similarities)
            if save_logs:
                log_messages.append(f"\nOverall Similarity: {overall_similarity:.4f}")
        else:
            overall_similarity = float('inf')
            if save_logs:
                log_messages.append("\nNo matching segments found. Similarity set to inf.")

        # If logging is enabled, write all log messages to file in one go.
        if save_logs:
            log_file = os.path.join(image_comparison_dir, "similarity_scores.txt")
            with open(log_file, "w") as log:
                log.write("\n".join(log_messages))

        return overall_similarity
