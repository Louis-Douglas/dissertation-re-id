import glob
import numpy as np

from src.utils.validation_utils import evaluate_rank_map, visualize_reid_results
from src.utils.segmentation_utils import get_processed_images

def main():
    save_logs = False
    query_image_paths = sorted(glob.glob("../datasets/system_b/query/*/*.png"))
    gallery_image_paths = sorted(glob.glob("../datasets/system_b/gallery/*/*.png"))

    moda_model_path = "../Training/modanet-seg-30.mlpackage"  # Use a YOLO model trained for clothing segmentation
    coco_model_path = "../Training/yolo11n-seg.mlpackage"  # Use a YOLO model trained for person segmentation

    query_image_files = get_processed_images(query_image_paths, coco_model_path, moda_model_path, save_logs)
    gallery_image_files = get_processed_images(gallery_image_paths, coco_model_path, moda_model_path, save_logs)

    num_query = len(query_image_paths)
    num_gallery = len(gallery_image_paths)

    # Create distance matrix (query × gallery)
    dist_matrix = np.zeros((num_query, num_gallery), dtype=np.float32)

    # Fill in similarity values
    for i, query_img in enumerate(query_image_files):
        for j, gallery_img in enumerate(gallery_image_files):
            dist = query_img.compare_processed_image(gallery_img, save_logs)  # Your similarity function
            dist_matrix[i, j] = dist  # Assign the similarity score

    # Convert to numpy
    similarity_matrix = 1 / (1 + dist_matrix)  # Invert to make higher values better


    print(similarity_matrix)

    # Evaluate Re-ID Performance
    evaluate_rank_map(similarity_matrix, query_image_paths, gallery_image_paths)

    # Visualise the results
    # visualize_reid_results(query_image_paths, gallery_image_paths, similarity_matrix, top_k=5)

if __name__ == "__main__":
    main()