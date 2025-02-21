import torch
import os
import cv2
import numpy as np
from collections import defaultdict

from ultralytics import YOLO
from src.utils.file_ops import save_dominant_colors_to_csv


def get_dominant_color_hex(image, mask):
    """
    Computes the average colour of the region in 'image' where 'mask' is True and returns it as a hex string.
    Assumes the image is in BGR format (as is typical with OpenCV and ultralytics).
    """
    if mask.sum() == 0:
        return None
    # Extract only the pixels corresponding to the object
    object_pixels = image[mask]
    # Compute the average colour (remains in BGR)
    avg_color = np.mean(object_pixels, axis=0)
    # Convert to integers and switch from BGR to RGB
    avg_color = [int(x) for x in avg_color[::-1]]
    # Format to hex code
    hex_code = '#{:02x}{:02x}{:02x}'.format(*avg_color)
    return hex_code


def get_dominant_colors(image_dir, model_path="yolo11x-seg.pt"):
    """
    Runs YOLO segmentation on all images in the provided directory and creates a dictionary mapping each
    object class to a list of dominant color hex codes from each segmented instance.
    """
    # Load YOLO segmentation model
    model = YOLO(model_path)

    # Dictionary: class name -> list of hex codes
    dominant_colors = defaultdict(list)

    # Count total images to process
    total_images = sum(1 for file in os.listdir(image_dir) if file.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"Processing {total_images} images...")

    # Run YOLO segmentation in streaming mode for efficiency
    results = model.predict(image_dir, verbose=False, device=torch.device("mps"), stream=True, task="segment")

    processed = 0
    for result in results:
        # Retrieve the original image
        image = result.orig_img
        masks = result.masks.data if result.masks is not None else []

        for i, mask in enumerate(masks):
            cls = result.boxes.data[i][-1] # Get last element of bounding box data which is Class index
            class_name = model.names[int(cls)]

            # Convert the mask tensor to a boolean numpy array
            mask_np = mask.cpu().numpy().astype(bool)

            # If the mask size doesn't match the image, resize it (using nearest neighbor interpolation)
            if mask_np.shape[:2] != image.shape[:2]:
                mask_np = cv2.resize(mask_np.astype(np.uint8), (image.shape[1], image.shape[0]),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)

            # Compute the dominant colour hex code for the segmented region
            hex_code = get_dominant_color_hex(image, mask_np)
            if hex_code:
                dominant_colors[class_name].append(hex_code)

        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed}/{total_images} images")

    return dominant_colors


if __name__ == "__main__":

    # Get the absolute path of the directory where THIS script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Define the project root (if script is in `src/`, go up one level)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

    # Now define paths relative to the project root
    MODEL_PATH = os.path.join(PROJECT_ROOT, "Training", "modanet-seg2.pt")
    IMAGE_DIR = os.path.join(PROJECT_ROOT, "datasets", "Market-1501-v15.09.15", "bounding_box_train")
    CSV_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "frequency_tables", "dominant_colors.csv")

    print(f"Model path: {MODEL_PATH}")
    print(f"Image directory: {IMAGE_DIR}")

    # Check if paths exist
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    colors_dict = get_dominant_colors(IMAGE_DIR, MODEL_PATH)

    save_dominant_colors_to_csv(colors_dict, CSV_OUTPUT_PATH)

    print("Dominant Colors by Class:")
    for cls, hex_codes in colors_dict.items():
        print(f"{cls}: {hex_codes}")
