import networkx as nx
import matplotlib.pyplot as plt
from src.utils.file_ops import clear_directory
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from src.utils.segmentation_utils import get_processed_images
import numpy as np
from glob import glob

def group_images_hierarchical(image_list, threshold=50, method='complete'):
    """
    Groups images hierarchically based on their pairwise similarity distances.

    This function calculates a distance matrix from a list of images, performs
    hierarchical clustering, and groups the images into clusters based on a
    specified distance threshold. It also visualises the clustering using a
    dendrogram.

    Args:
        image_list (list): A list of image objects.
        threshold (float, optional): The distance threshold for cutting the
            hierarchical clustering tree.
        method (str, optional): The linkage method for hierarchical clustering..

    Returns:
        list: A list of sets, where each set represents a cluster of image names.
    """
    N = len(image_list)
    # Build an NxN distance matrix
    dist_matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i+1, N):
            # Gives similarity distance score (lower = better)
            dist = image_list[i].compare_processed_image(image_list[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Convert the NxN matrix into condensed form that only contains unique pairwise distances
    condensed = squareform(dist_matrix, checks=False)

    # Perform hierarchical clustering using scipy's linkage function
    Z = linkage(condensed, method=method)

    # Cut the tree at 'threshold' to get cluster labels
    labels = fcluster(Z, t=threshold, criterion='distance')

    # Group images by label
    clusters_dict = {}
    for idx, label in enumerate(labels):
        # Check if label exists in the dictionary and initialise if empty
        if label not in clusters_dict:
            clusters_dict[label] = []
        # Append the image name to the list corresponding to the label
        clusters_dict[label].append(image_list[idx].image_name)

    # Initialise an empty list to store cluster sets
    clusters = []
    # Iterate through each list of image names in clusters_dict
    for images in clusters_dict.values():
        image_set = set(images)  # Convert the list to a set
        clusters.append(image_set)  # Add the set to the final list of clusters

    # Loop and print the clusters for manual validation
    print("Clusters:")
    for cluster in clusters:
        print(cluster)

    # Initialise an empty list for labels
    image_labels = []
    for img in image_list:
        image_labels.append(img.image_name)

    # Visualise the clustering using scipy's dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(Z, labels=np.array(image_labels), color_threshold=threshold)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold = {threshold}")
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Image Names")
    plt.ylabel("Similarity Distance")
    plt.legend()
    plt.show()
    return clusters

def group_images(image_list, merge_threshold=30.5):
    all_comparisons = []
    G = nx.Graph()
    for processed_image in image_list:
        name = processed_image.image_name
        G.add_node(name)

    highest_true_match = 0
    highest_match_str = ""
    lowest_false_match = 100
    lowest_match_str = ""

    # Clear log output directory before running
    log_output_directory = "../logs/comparison_results"
    clear_directory(log_output_directory)

    # Compute pairwise similarities.
    for i in range(len(image_list)):
        for j in range(i + 1, len(image_list)):
            i1 = image_list[i]
            i2 = image_list[j]
            sim = i1.compare_processed_image(i2, 10.0, log_output_directory)
            all_comparisons.append(f"{i1.image_name} & {i2.image_name} with similarity {sim}")
            n1 = i1.image_name.split('_')[0]
            n2 = i2.image_name.split('_')[0]
            # print(f"{n1} & {n2} & {sim}")
            if (n1 == n2) and (sim > highest_true_match):
                highest_true_match = sim
                highest_match_str = f"{i1.image_name} & {i2.image_name}"

            if (n1 != n2) and (sim < lowest_false_match):
                lowest_false_match = sim
                lowest_match_str = f"{i1.image_name} & {i2.image_name}"

            if sim < merge_threshold:
                G.add_edge(i1.image_name, i2.image_name, weight=sim)

    clusters = list(nx.connected_components(G))
    nx.draw(G, with_labels=True)
    plt.show()
    print("Clusters:")
    for cluster in clusters:
        print(cluster)

    for item in all_comparisons:
        print(item)

    print("Highest true match:", highest_true_match)
    print("Highest match:", highest_match_str)
    print("Lowest false match:", lowest_false_match)
    print("Lowest match:", lowest_match_str)


def main():
    """
    Runs the re-identification pipeline using Modanet YOLO model for clothing-based segmentation and
    COCO YOLO model for person extraction.

    Args:
        dataset_dir (str): Path to the dataset directory.
        moda_model_path (str): Path to the YOLO model file.
    """
    # dataset_dir = "../datasets/Gen-test2"
    dataset_dir = "../datasets/Custom-Gen"
    gallery_image_paths = sorted(glob(f"{dataset_dir}/*/*.png") + glob(f"{dataset_dir}/*/*.jpg"))
    moda_model_path = "../Training/modanet-seg-30.pt"  # Use a YOLO model trained for clothing segmentation
    coco_model_path = "../Training/yolo11x-seg.pt"  # Use a YOLO model trained for person segmentation

    processed_images = get_processed_images(gallery_image_paths, coco_model_path, moda_model_path)
    # group_images(processed_images)
    group_images_hierarchical(processed_images)



if __name__ == '__main__':
    main()
