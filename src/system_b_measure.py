import glob
import numpy as np
from memory_profiler import memory_usage
import os
import time

from src.utils.file_ops import split_gallery_evenly
from src.utils.evaluation_utils import evaluate_rank_map_per_query
from src.utils.segmentation_utils import get_processed_images
from multiprocessing import Pool
from src.utils.multiprocessing_utils import compute_similarity

def process_subset(i, subset, dataset_dir, save_logs):
    # Count total images in this subset
    total_images = len(subset)
    print(f"Increment {i}: {total_images} gallery images")

    query_image_paths = sorted(glob.glob(os.path.join(dataset_dir, "query/*/*.png")))
    gallery_image_paths = subset

    moda_model_path = "../weights/modanet-seg-30.mlpackage"  # YOLO model trained for clothing segmentation
    coco_model_path = "../weights/yolo11n-seg.mlpackage"  # YOLO model trained for person segmentation

    query_image_files = get_processed_images(query_image_paths, coco_model_path, moda_model_path, save_logs)
    gallery_image_files = get_processed_images(gallery_image_paths, coco_model_path, moda_model_path, save_logs)

    num_query = len(query_image_files)
    num_gallery = len(gallery_image_files)

    # Automatically disable logging if dataset is too large
    if (num_query * num_gallery) > 10000:
        print("Disabling save logs due to large dataset")
        save_logs = False

    # Create distance matrix (query × gallery)
    dist_matrix = np.zeros((num_query, num_gallery), dtype=np.float32)

    # Build a list of tasks: each task is a tuple (i, j, query_img, gallery_img, save_logs)
    tasks = []
    for i, query_img in enumerate(query_image_files):
        for j, gallery_img in enumerate(gallery_image_files):
            tasks.append((i, j, query_img, gallery_img, save_logs))

    # Use a multiprocessing Pool with 8 processes.
    with Pool(processes=8) as pool:
        results = pool.map(compute_similarity, tasks)

    # Fill in the distance matrix from the results.
    for i, j, result in results:
        if result is not None:
            dist_matrix[i, j] = result

    # Convert to numpy
    similarity_matrix = 1 / (1 + dist_matrix)  # Invert to make higher values better

    print(similarity_matrix)

    # Evaluate Re-ID Performance
    rank1_array, rank5_array, mAP_array = evaluate_rank_map_per_query(similarity_matrix, query_image_paths, gallery_image_paths)
    print(f"Overall Rank-1 Accuracy: {np.mean(rank1_array) * 100:.2f}%")
    print(f"Overall Rank-5 Accuracy: {np.mean(rank5_array) * 100:.2f}%")
    print(f"Overall mAP: {np.mean(mAP_array) * 100:.2f}%\n")


def main():
    save_logs = False
    dataset_dir = "../datasets/Ethical-filtered"

    gallery_dir = os.path.join(dataset_dir, "gallery")
    # Define increments
    increments = [1, 2, 3, 4, 5] # images per person
    gallery_subsets = split_gallery_evenly(gallery_dir, increments)

    # List to store results: (increment number, total gallery images, elapsed time, memory usage)
    results = []

    # Process each gallery subset and record the processing time
    for i, subset in enumerate(gallery_subsets, start=1):
        start_time = time.perf_counter()
        # memory_usage will run process_subset in a separate process and sample memory usage.
        mem_usage = memory_usage(
            (process_subset, (i, subset, dataset_dir, save_logs)),
            interval=0.1,
            timeout=None
        )
        # process_subset(i, subset, dataset_dir, save_logs)
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