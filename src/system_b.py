import glob
import numpy as np
import os
from multiprocessing import Pool

from src.utils.evaluation_utils import visualize_reid_results, evaluate_rank_map_per_query
from src.utils.segmentation_utils import get_processed_images
from src.utils.multiprocessing_utils import compute_similarity


def main(dataset_dir, enable_apply_clahe):
    print("Current working directory:", os.getcwd())
    save_logs = False


    query_image_paths = sorted(glob.glob(os.path.join(dataset_dir, "query/*/*.png")))
    gallery_image_paths = sorted(glob.glob(os.path.join(dataset_dir, "gallery/*/*.png")))

    moda_model_path = "../weights/modanet-seg.mlpackage"  # YOLO model trained for clothing segmentation
    coco_model_path = "../weights/yolo11n-seg.mlpackage"  # YOLO model trained for person segmentation


    query_image_files = get_processed_images(query_image_paths, coco_model_path, moda_model_path, save_logs, enable_apply_clahe)
    gallery_image_files = get_processed_images(gallery_image_paths, coco_model_path, moda_model_path, save_logs, enable_apply_clahe)

    num_query = len(query_image_paths)
    num_gallery = len(gallery_image_paths)

    # Automatically disable logging if dataset is too large
    if (num_query * num_gallery) > 1000:
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

    # Evaluate Re-ID Performance
    rank1_array, rank5_array, mAP_array = evaluate_rank_map_per_query(similarity_matrix, query_image_paths,
                                                                      gallery_image_paths)

    # Save results to a text file
    output_dir = "../logs/statistical_test_data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "system_b_ethical_filtered_uncropped_CLAHE_testing.txt")
    with open(output_file, "w") as f:
        f.write("System B - Ethical-filtered-uncropped-CLAHE\n")
        f.write("========================\n\n")
        f.write(f"Overall Rank-1 Accuracy: {np.mean(rank1_array) * 100:.2f}%\n")
        f.write(f"Overall Rank-5 Accuracy: {np.mean(rank5_array) * 100:.2f}%\n")
        f.write(f"Overall mAP: {np.mean(mAP_array) * 100:.2f}%\n\n")
        f.write("Per-Query Metrics:\n")
        f.write("Query\tRank-1\tRank-5\tmAP\n")
        for i in range(len(rank1_array)):
            f.write(f"{i + 1}\t{rank1_array[i]:.3f}\t{rank5_array[i]:.3f}\t{mAP_array[i]:.3f}\n")

        # Save the arrays in a copy-paste friendly format
        f.write("Per-Query Metrics as Python Lists:\n")
        f.write(f"rank1_array = {rank1_array.tolist()}\n")
        f.write(f"rank5_array = {rank5_array.tolist()}\n")
        f.write(f"mAP_array   = {mAP_array.tolist()}\n")

    # Visualise the results
    visualize_reid_results(query_image_paths, gallery_image_paths, similarity_matrix, top_k=5)

    return rank1_array, rank5_array, mAP_array

if __name__ == "__main__":
    dataset_dir = "../datasets/Ethical-filtered"
    enable_apply_clahe = True
    rank1_array, rank5_array, mAP_array = main(dataset_dir, enable_apply_clahe)
    print(f"Rank1 : {np.mean(rank1_array) * 100:.2f}%")
    print(f"Rank5 : {np.mean(rank5_array) * 100:.2f}%")
    print(f"mAP : {np.mean(mAP_array) * 100:.2f}%")