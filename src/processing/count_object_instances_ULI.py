import torch
import os
import concurrent.futures
from collections import defaultdict
from ultralytics import YOLO


def count_segmented_objects(image_dir, model_path="yolo11x-seg.mlpackage"):
    """
    Runs YOLO segmentation on all images in a directory and sums the total instances of each segmented object class.

    Args:
        image_dir (str): Path to the directory containing images.
        model_path (str): Path to the YOLO model.

    Returns:
        dict: A dictionary with class names as keys and total segmented object counts as values.
    """
    # Load YOLO model for segmentation
    model = YOLO(model_path)
    total_class_counts = defaultdict(int)

    total_images = 0
    # Get count of total number of images to process
    for file in os.listdir(image_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            total_images += 1

    print(f"Processing {total_images} images in {image_dir}...")

    results = model.predict(image_dir, verbose=False, device=torch.device("mps"), stream=True, task="segment")
    class_names = model.names

    processed = 0
    for result in results:
        image_class_counts = defaultdict(int)
        data = result.masks.data if result.masks is not None else []

        for i, mask in enumerate(data):
            cls = result.boxes.data[i][-1]
            class_name = class_names[int(cls)]
            image_class_counts[class_name] += 1

        for class_name, count in image_class_counts.items():
            total_class_counts[class_name] += count

        processed += 1
        if processed % 100 == 0:
            print(f"{image_dir}: Processing... {processed}/{total_images} images")

    return total_class_counts


def process_directories(directories, model_path="yolo11x-seg.mlpackage", max_workers=4):
    """
    Processes multiple directories concurrently using multithreading.

    Args:
        directories (list): List of directory paths to process.
        model_path (str): Path to the YOLO model.
        max_workers (int): Number of threads to use for parallel processing.
    """
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dir = {executor.submit(count_segmented_objects, directory, model_path): directory for directory in
                         directories}

        for future in concurrent.futures.as_completed(future_to_dir):
            directory = future_to_dir[future]
            try:
                results[directory] = future.result()
                print(f"Completed processing for {directory}")
            except Exception as e:
                print(f"Error processing {directory}: {e}")

    return results

directories = []
root_path = "../Datasets/Gen-Test/identity115"
for sub_dir in os.listdir(root_path):
    directories.append(os.path.join(root_path, sub_dir))

# Run segmentation counting in multiple directories concurrently
final_results = process_directories(directories, model_path="../../Training/yolo11x-seg.mlpackage", max_workers=2)

# Print total results
for directory, counts in final_results.items():
    print(f"Total Segmented Objects in {directory}: {counts}")
