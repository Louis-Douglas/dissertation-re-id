import torchreid
import torch
import glob
import os
import time
import numpy as np
from memory_profiler import memory_usage
from src.utils.evaluation_utils import evaluate_rank_map_per_query
from src.utils.file_ops import split_gallery_evenly

def process_subset(i, subset, dataset_dir, extractor):
    # Count total images in this subset
    total_images = len(subset)
    print(f"Increment {i}: {total_images} gallery images")

    query_image_paths = sorted(glob.glob(os.path.join(dataset_dir, "query/*/*.png")))
    gallery_image_paths = subset

    # Extract features
    query_features = extractor(query_image_paths)
    gallery_features = extractor(gallery_image_paths)

    print(f"Extracted {query_features.shape[0]} query features")
    print(f"Extracted {gallery_features.shape[0]} gallery features")

    # Normalise features
    query_features = torch.nn.functional.normalize(query_features, dim=1)
    gallery_features = torch.nn.functional.normalize(gallery_features, dim=1)

    # Compute similarity
    similarity_matrix = torch.mm(query_features, gallery_features.t())  # Cosine similarity

    # Convert to numpy
    similarity_matrix = similarity_matrix.cpu().numpy()

    print(similarity_matrix)

    # Evaluate Re-ID Performance
    rank1_array, rank5_array, mAP_array = evaluate_rank_map_per_query(similarity_matrix, query_image_paths,
                                                                      gallery_image_paths)
    print(f"Overall Rank-1 Accuracy: {np.mean(rank1_array) * 100:.2f}%\n")
    print(f"Overall Rank-5 Accuracy: {np.mean(rank5_array) * 100:.2f}%\n")
    print(f"Overall mAP: {np.mean(mAP_array) * 100:.2f}%\n\n")


def main():
    extractor = torchreid.utils.FeatureExtractor(
        model_name='resnet50',
        model_path='../Training/resnet50.pth', # The path to the downloaded model
        device="cpu" # Change if you have a compatible GPU
    )

    # dataset_dir = "../datasets/Ethical-filtered-cropped"
    dataset_dir = "../datasets/Ethical-filtered"

    gallery_dir = os.path.join(dataset_dir, "gallery")

    # Define increments
    increments = [1, 2, 3, 4, 5] # images per person

    gallery_subsets = split_gallery_evenly(gallery_dir, increments)

    # List to store data results: (increment number, total gallery images, elapsed time, memory usage)
    results = []

    # Process each gallery subset and record the processing time
    for i, subset in enumerate(gallery_subsets, start=1):
        start_time = time.perf_counter()
        # memory_usage will run process_subset in a separate process and sample memory usage.
        mem_usage = memory_usage(
            (process_subset, (i, subset, dataset_dir, extractor)),
            interval=0.1,
            timeout=None
        )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        total_images = len(subset)
        max_memory = max(mem_usage)  # in MB
        results.append((i, total_images, elapsed_time, max_memory))

    # Output timing and memory results as a formatted table.
    print("\nTiming and Memory Usage Results:")
    print("{:<12} {:<20} {:<20} {:<20}".format("Increment", "Gallery Images", "Time (s)", "Max Memory (MB)"))
    for inc, num_img, t, mem in results:
        print("{:<12} {:<20} {:<20.2f} {:<20.2f}".format(inc, num_img, t, mem))

if __name__ == "__main__":
    main()