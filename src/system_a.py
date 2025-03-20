import torchreid
import torch
import glob
import os
from src.utils.evaluation_utils import evaluate_rank_map, visualize_reid_results

def main():
    extractor = torchreid.utils.FeatureExtractor(
        model_name='resnet50',
        model_path='../Training/resnet50.pth', # The path to the downloaded model
        device="cpu" # Change if you have a compatible GPU
    )

    # dataset_dir = "../datasets/Ethical-filtered-cropped"
    dataset_dir = "../datasets/Ethical-filtered"

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
    evaluate_rank_map(similarity_matrix, query_image_paths, gallery_image_paths)

    # Visualise the results
    # visualize_reid_results(query_images, gallery_images, similarity_matrix, top_k=5)

if __name__ == "__main__":
    main()