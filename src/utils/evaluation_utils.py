import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils.file_ops import load_image

def evaluate_rank_map_per_query(similarity_matrix, query_images, gallery_images):
    """
    Evaluates Re-ID performance.

    For each query image, this function computes an array of:
      - A binary indicator for Rank-1 (1 if the top gallery match is correct, else 0),
      - A binary indicator for Rank-5 (1 if any of the top-5 gallery matches is correct, else 0),
      - The Average Precision (AP) based on the ranks of all correct matches.

    Args:
        similarity_matrix (np.ndarray): A 2D array (num_query x num_gallery) of similarity scores.
        query_images (list): List of query image file paths.
        gallery_images (list): List of gallery image file paths.

    Returns:
        tuple: Three np.ndarray objects corresponding to the per-query Rank-1, Rank-5, and AP values.
    """
    num_query = len(query_images)

    # Extract query person IDs from the parent folder names
    query_pids = []

    for q in query_images:
        # Get the parent directory of the file path
        parent_dir = os.path.dirname(q)

        # Get the name of that directory (assumed to be the person ID)
        pid_str = os.path.basename(parent_dir)

        # Convert to integer and store
        pid = int(pid_str)
        query_pids.append(pid)

    # Convert the list to a NumPy array
    query_pids = np.array(query_pids)

    # Create array of gallery truth person IDs
    gallery_pids = []

    for g in gallery_images:
        parent_dir = os.path.dirname(g)
        pid_str = os.path.basename(parent_dir)
        pid = int(pid_str)
        gallery_pids.append(pid)

    gallery_pids = np.array(gallery_pids)

    # For each query, sort gallery indices by descending similarity score.
    indices = np.argsort(-similarity_matrix, axis=1) # Sorting on query axis

    rank1_list = []
    rank5_list = []
    ap_list = []

    # Loop over query images from 0 to 99
    for i in range(num_query):
        gallery_image_indices = indices[i] # Get sorted gallery image indices for query i
        sorted_gallery_pids = gallery_pids[gallery_image_indices] # Get actual person IDs for sorted gallery image list
        truth_pid = query_pids[i] # Get truth person ID
        correct_matches = (sorted_gallery_pids == truth_pid) # Get array of correct matches

        # Rank-1 indicator: 1 if the top-ranked gallery image is correct
        rank1_list.append(1 if correct_matches[0] else 0)
        # Rank-5 indicator: 1 if any of the top 5 gallery images is correct
        rank5_list.append(1 if np.any(correct_matches[:5]) else 0)

        # Compute Average Precision (AP) for the query
        # Finds positions in the sorted gallery array where correct matches occurred
        relevant = np.where(correct_matches)[0] # Indices of where there's correct matches in the 100 long array
        if len(relevant) > 0:
            precision_at_k = []

            for k, idx in enumerate(relevant):
                # k is the index in the list of relevant correct matches
                # idx is the index at which the k-th relevant item appears in the full ranked list

                # Compute precision at this point
                precision = (k + 1) / (idx + 1) # Number of relevant items / Position in the full ranked list
                precision_at_k.append(precision)

            ap_list.append(np.mean(precision_at_k))
        else:
            ap_list.append(0)

    return np.array(rank1_list), np.array(rank5_list), np.array(ap_list)


def visualize_reid_results(query_images, gallery_images, similarity_matrix, top_k=5):
    """
    Visualises the Re-ID results by showing query images with their top-k matched gallery images.
    """

    num_queries = len(query_images)
    for i in range(num_queries):
        # 1 row, top-k + 1 columns to show query and then top-k gallery matches
        fig, axes = plt.subplots(1, top_k + 1, figsize=(30, 10))

        # Load and display the query image
        query_img = load_image(query_images[i])
        axes[0].imshow(query_img)
        axes[0].set_title("Query")
        axes[0].axis("off")

        # Retrieve similarity scores for query image i
        query_similarities = similarity_matrix[i]  # This is a 1D array of similarity scores

        # Multiply query_similarities by -1 to sort in descending order
        # Run argsort to sort indices from most to least similarity
        sorted_indices = np.argsort(-query_similarities)

        # Select only the top-k gallery indices
        top_indices = sorted_indices[:top_k]

        # Initialise an empty list to store top-k gallery images
        top_gallery_imgs = []

        # Loop through top-k indices and retrieve corresponding gallery image paths
        for gal_id in top_indices:
            gallery_image_path = gallery_images[gal_id]  # Get the gallery image filename
            top_gallery_imgs.append(gallery_image_path)  # Add it to the list

        # Align each top-k image in its appropriate column
        for j, gallery_path in enumerate(top_gallery_imgs):
            gallery_img = load_image(gallery_path)
            # Set the gallery image to the appropriate subplot
            axes[j + 1].imshow(gallery_img)
            axes[j + 1].set_title(f"Match {j+1}")
            axes[j + 1].axis("off") # Remove axis ticks and labels

        plt.show()  # Display the images