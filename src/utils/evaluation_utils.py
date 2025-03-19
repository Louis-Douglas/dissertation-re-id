import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils.file_ops import load_image

def evaluate_rank_map(similarity_matrix, query_images, gallery_images):
    num_query = similarity_matrix.shape[0] # Extract number of query images from rows in matrix

    # Initialise empty lists
    query_pids_list = []
    gallery_pids_list = []

    # Extract person IDs for queries
    for query_image_path in query_images:
        # Get the person ID from the folder name
        pid = int(os.path.basename(os.path.dirname(query_image_path)))
        query_pids_list.append(pid)  # Append result to the list

    # Extract person IDs for gallery
    for gallery_image_path in gallery_images:
        # Get the person ID from the folder name
        pid = int(os.path.basename(os.path.dirname(gallery_image_path)))
        gallery_pids_list.append(pid)  # Append result to the list

    # Convert lists to NumPy arrays
    query_pids = np.array(query_pids_list)
    gallery_pids = np.array(gallery_pids_list)

    # Sort the gallery by similarity scores
    # Reversing the similarity_matrix to then order highest to lowest
    # axis=1 ensures we sort the columns, not the rows
    indices = np.argsort(-similarity_matrix, axis=1)  # Descending order for similarity

    rank1 = 0
    rank5 = 0
    avg_precision = []

    for i in range(num_query):
        # Extract sorted gallery person IDs for this query
        sorted_gallery_pids = gallery_pids[indices[i]]  # Retrieve reordered gallery PIDs
        # Compare with the actual query person ID
        correct_matches = np.array(sorted_gallery_pids == query_pids[i])  # Create an array of booleans

        if correct_matches[0]:  # Rank-1
            rank1 += 1
        if np.any(correct_matches[:5]):  # Rank-5
            rank5 += 1

        # Compute Average Precision (AP)
        relevant = np.where(correct_matches)[0]  # Get correct match positions
        if len(relevant) > 0:
            # Initialise an empty list for precision values
            precision_at_k = []

            # Iterate over relevant matches
            for k, idx in enumerate(relevant):
                precision_value = (k + 1) / (idx + 1)  # Compute precision at this rank
                precision_at_k.append(precision_value)  # Append to list

            avg_precision.append(np.mean(precision_at_k))

    # Compute final scores
    rank1 = rank1 / num_query
    rank5 = rank5 / num_query

    if avg_precision:
        mAP = np.mean(avg_precision)
    else:
        mAP = 0

    print(f"Rank-1 Accuracy: {rank1 * 100:.2f}%")
    print(f"Rank-5 Accuracy: {rank5 * 100:.2f}%")
    print(f"Mean Average Precision (mAP): {mAP * 100:.2f}%")

    return rank1, rank5, mAP

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
        for idx in top_indices:
            gallery_image_path = gallery_images[idx]  # Get the gallery image filename
            top_gallery_imgs.append(gallery_image_path)  # Add it to the list

        # Align each top-k image in its appropriate column
        for j, gallery_path in enumerate(top_gallery_imgs):
            gallery_img = load_image(gallery_path)
            # Set the gallery image to the appropriate subplot
            axes[j + 1].imshow(gallery_img)
            axes[j + 1].set_title(f"Match {j+1}")
            axes[j + 1].axis("off") # Remove axis ticks and labels

        plt.show()  # Display the images