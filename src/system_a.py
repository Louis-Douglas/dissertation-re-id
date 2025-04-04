import torchreid
import torch
import glob
import os
import numpy as np
from src.utils.evaluation_utils import visualize_reid_results, evaluate_rank_map_per_query

def main(dataset_dir):
    extractor = torchreid.utils.FeatureExtractor(
        model_name='resnet50',
        model_path='../Training/resnet50.pth', # The path to the downloaded model
        device="cpu" # Change if you have a compatible GPU
    )

    query_image_paths = sorted(glob.glob(os.path.join(dataset_dir, "query/*/*.png")))
    gallery_image_paths = sorted(glob.glob(os.path.join(dataset_dir, "gallery/*/*.png")))

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
    rank1_array, rank5_array, ap_array = evaluate_rank_map_per_query(similarity_matrix, query_image_paths, gallery_image_paths)

    # Save results to a text file
    output_dir = "../logs/statistical_test_data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "system_a_hypothesis_2_results_delete.txt")
    with open(output_file, "w") as f:
        f.write("System A - Ethical-filtered-uncropped\n")
        f.write("========================\n\n")
        f.write(f"Overall Rank-1 Accuracy: {np.mean(rank1_array) * 100:.2f}%\n")
        f.write(f"Overall Rank-5 Accuracy: {np.mean(rank5_array) * 100:.2f}%\n")
        f.write(f"Overall mAP: {np.mean(ap_array) * 100:.2f}%\n\n")
        f.write("Per-Query Metrics:\n")
        f.write("Query\tRank-1\tRank-5\tmAP\n")
        for i in range(len(rank1_array)):
            f.write(f"{i + 1}\t{rank1_array[i]:.3f}\t{rank5_array[i]:.3f}\t{ap_array[i]:.3f}\n")

        # Save the arrays in a copy-paste friendly format
        f.write("Per-Query Metrics as Python Lists:\n")
        f.write(f"rank1_array = {rank1_array.tolist()}\n")
        f.write(f"rank5_array = {rank5_array.tolist()}\n")
        f.write(f"mAP_array   = {ap_array.tolist()}\n")

    return rank1_array, rank5_array, ap_array

    # Visualise the results
    # visualize_reid_results(query_images, gallery_images, similarity_matrix, top_k=5)

if __name__ == "__main__":
    dataset_dir = "../datasets/Ethical-filtered-cropped"
    rank1_array, rank5_array, ap_array = main(dataset_dir)
    print(f"Rank1 : {np.mean(rank1_array) * 100:.2f}%")
    print(f"Rank5 : {np.mean(rank5_array) * 100:.2f}%")
    print(f"mAP : {np.mean(ap_array) * 100:.2f}%")