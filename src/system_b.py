from ultralytics import YOLO

from src.utils.segmentation_utils import get_target_person, get_connected_segments
import os
import torch
from PIL import Image
from src.core.processed_image import ProcessedImage
import networkx as nx
import matplotlib.pyplot as plt
from src.utils.file_ops import clear_directory

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


def main(dataset_dir, moda_model_path, coco_model_path):
    """
    Runs the re-identification pipeline using a YOLO model for clothing-based segmentation.

    Args:
        dataset_dir (str): Path to the dataset directory.
        moda_model_path (str): Path to the YOLO model file.
    """
    # Load Modanet trained YOLO model
    moda_model = YOLO(moda_model_path)
    coco_model = YOLO(coco_model_path)

    # Collect all image paths
    image_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(dataset_dir)
        for file in files if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Run YOLO prediction, filtering for person
    coco_results = coco_model.predict(image_paths, classes=[0], device=torch.device("mps"), stream=True, retina_masks=True)

    person_to_box = {}

    for result in coco_results:
        target = get_target_person(result)
        person_to_box[result.path] = target["bounding_box"]

    # Run YOLO prediction, filtering for person
    moda_results = moda_model.predict(image_paths, device=torch.device("mps"), stream=True, retina_masks=True)
    processed_images = []

    # Process results
    for result in moda_results:
        related_person_box = person_to_box[result.path]
        image_name = os.path.basename(result.path)
        print(f"Processing {image_name}")

        # Open the original image
        original_image = Image.open(result.path).convert("RGBA")
        # equalised_image = Image.fromarray(equalise_image(np.array(original_image)))
        # equalised_image = Image.fromarray(apply_clahe(np.array(equalised_image)))
        # normalised_image = Image.fromarray(normalise_image(np.array(equalised_image)))

        # Ensure the result contains masks
        if result.masks is None:
            continue

        print(result.path)

        processed_segments = get_connected_segments(original_image, related_person_box, result)

        # Convert extracted clothing regions into a processed image
        processed_image = ProcessedImage(result.path, original_image, processed_segments)
        processed_images.append(processed_image)

    group_images(processed_images)



if __name__ == '__main__':
    # dataset_dir = "../datasets/Gen-test2"
    dataset_dir = "../datasets/Custom-Gen"
    moda_model_path = "../Training/modanet-seg2.pt"  # Use a YOLO model trained for clothing segmentation
    coco_model_path = "../Training/yolo11x-seg.pt"  # Use a YOLO model trained for person segmentation
    main(dataset_dir, moda_model_path, coco_model_path)
