import torch
import os
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

    # Dictionary to store the total count of each segmented class
    total_class_counts = defaultdict(int)

    total_images = 0

    # Get count of total number of images to process
    for file in os.listdir(image_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            total_images += 1

    print(f"Processing {total_images} images...")

    # Run YOLO segmentation in streaming mode (batch processing)
    results = model.predict(image_dir, verbose=False, device=torch.device("mps"), stream=True, task="segment")

    # Get class names
    class_names = model.names

    # Track number of processed images
    processed = 0
    for result in results:
        image_class_counts = defaultdict(int)

        # Extract segmentation masks
        if result.masks is not None:
            data = result.masks.data
        else:
            data = []

        for i, mask in enumerate(data):
            cls = result.boxes.data[i][-1]  # Get last element of bounding box data which is Class index
            class_name = class_names[int(cls)]
            image_class_counts[class_name] += 1  # Count segmented objects

        # Add image counts to total counts
        for class_name, count in image_class_counts.items():
            total_class_counts[class_name] += count

        processed += 1
        if processed % 100 == 0:
            print(f"Processing... {processed}/{total_images} images")

    return total_class_counts

# Get the absolute path of the directory where THIS script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the project root (if script is in `src/`, go up one level)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

# Now define paths relative to the project root
MODEL_PATH = os.path.join(PROJECT_ROOT, "Training", "yolo11x-seg.pt")
IMAGE_DIR = os.path.join(PROJECT_ROOT, "datasets", "Market-1501-v15.09.15", "bounding_box_train")

# Run segmentation counting
total_counts = count_segmented_objects(IMAGE_DIR, MODEL_PATH)
print("Total Segmented Objects:", total_counts)
