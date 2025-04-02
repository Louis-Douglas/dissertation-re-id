from torchvision.ops import box_iou, distance_box_iou
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO
import os
from memory_profiler import profile

from src.utils.histogram_utils import apply_clahe
from src.core.processed_segment import ProcessedSegment
from src.core.processed_image import ProcessedImage
from typing import List

def get_target_person(result):
    """
    Finds the largest segmented person mask in a YOLO result.

    Args:
        result (ultralytics.engine.results.Results): A single YOLO segmentation result.

    Returns:
        dict or None: Dictionary containing:
            - 'mask': The largest person's mask (torch.Tensor).
            - 'bounding_box': Bounding box of the largest mask.
            - 'area': Pixel area of the mask.
        Returns None if no persons are found.
    """
    masks = result.masks  # Get segmentation masks
    boxes = result.boxes  # Bounding boxes for detected objects
    class_names = result.names  # Get class names

    largest_mask = None
    largest_area = 0
    largest_box = None

    if masks is None:
        return None  # No masks found

    for i, mask in enumerate(masks.data):
        class_id = int(boxes[i].cls[0].item())  # Class ID of the detection

        if class_names[class_id] == "person":
            # Compute the area of the mask
            mask_area = torch.count_nonzero(mask).item()

            # Check if this is the largest person mask found so far
            if mask_area > largest_area:
                largest_area = mask_area
                largest_mask = mask
                largest_box = boxes[i].xyxy[0].tolist()

    if largest_mask is None:
        return None  # No person masks found

    return {
        "mask": largest_mask,
        "bounding_box": largest_box,
        "area": largest_area
    }

def extract_segmented_object(mask, original_image):
    """
    Extracts a segmented object from a YOLO mask, applies transparency, and crops the result.

    Args:
        mask (torch.Tensor): The segmentation mask for the detected object.
        original_image (PIL.Image.Image): The original input image (RGBA).

    Returns:
        PIL.Image.Image: Cropped and transparent-segmented PIL.Image of object.
    """
    # Convert mask to NumPy array (binary mask)
    mask_array = mask.cpu().numpy().astype(np.uint8) * 255  # Convert to 0-255 scale

    # Create a black and white PIL image from the mask array
    mask_image = Image.fromarray(mask_array, mode="L")

    # Create a blank RGBA image with a transparent background
    cropped_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))

    # Ensure mask dimensions match the original image dimensions
    mask_image = mask_image.resize(original_image.size)

    # Paste the original image onto the blank image, using the mask as an alpha channel
    cropped_image.paste(original_image, mask=mask_image)

    # Crop the result to the bounding box of the mask
    bbox = mask_image.getbbox()
    if bbox:
        cropped_image = cropped_image.crop(bbox)

    return cropped_image


def is_connected(box1, box2, iou_threshold=0):
    """
    Determines whether two bounding boxes are considered connected based on
    Intersection over Union (IoU) and center distance.

    Args:
        box1 (Tensor array): The first bounding box in the format (x1, y1, x2, y2).
        box2 (Tensor array): The second bounding box in the same format as box1.
        iou_threshold (float, optional): The minimum IoU required for boxes to be considered connected.
                                         Defaults to 0.

    Returns:
        bool: True if the boxes are considered connected, False otherwise.
    """
    # Get intersection over union of the two boxes
    iou = box_iou(box1, box2)

    # Return true if iou is above threshold
    return iou > iou_threshold

def get_connected_segments(original_image, person_box, result, iou_threshold=0.0):
    """
    Extracts objects (e.g., clothing, accessories) connected to a detected person
    using bounding boxes and segmentation masks.

    Args:
        original_image (PIL.Image.Image): The source image.
        person_box (list or tensor): Bounding box [x1, y1, x2, y2] of the detected person.
        result (ultralytics.YOLO.Result): YOLO detection result with bounding boxes, class names, and masks.
        iou_threshold (float, optional): IoU threshold for object-person association. Default is 0.1.

    Returns:
         List[ProcessedSegment]: List of processed segments.
    """
    boxes = result.boxes  # Bounding boxes for detected objects
    class_names = result.names  # Class names from model

    tensor_person_box = torch.tensor([person_box], dtype=torch.float32)  # Convert to tensor

    processed_segments = []
    # connected_objects = {}  # Store connected object details

    for i, box in enumerate(boxes):
        class_id = int(box.cls[0].item())  # Class ID of the detection
        class_name = class_names[class_id]  # Class name
        bbox = box.xyxy[0].tolist()  # Bounding box coordinates [x1, y1, x2, y2]

        if class_name == "person":
            continue  # Skip the person itself

        # Get bounding box of current object
        tensor_object_box = torch.tensor([bbox], dtype=torch.float32)

        # Check if object is connected to the person
        if is_connected(tensor_person_box, tensor_object_box, iou_threshold):
            masks = result.masks.data  # Get mask tensors
            mask = masks[i]  # Extract the mask for the detected object
            extracted_object_image = extract_segmented_object(mask, original_image)

            processed_segment = ProcessedSegment(extracted_object_image, class_id, class_name, box, mask)
            processed_segments.append(processed_segment)
    return processed_segments


def process_single_image(image_path, coco_model, moda_model, save_logs, enable_apply_clahe):
    """
    Processes a single image and returns a ProcessedImage object.

    Args:
        image_path (str): Path to the image.
        coco_model (YOLO): YOLO model for person detection.
        moda_model (YOLO): YOLO model for clothing detection.
        save_logs (bool): Whether to save logs.
        enable_apply_clahe (bool): Whether to apply clahe.

    Returns:
        ProcessedImage or None: Processed image object if processing succeeds, otherwise None.
    """
    try:
        # Run YOLO inference (single image at a time, avoid streaming)
        coco_result = coco_model.predict(image_path, classes=[0], device="cpu", retina_masks=True, verbose=False)[0]

        # Get person bounding box and save detection image
        target = get_target_person(coco_result)
        if not target:
            print(f"No person detected in {image_path}")
            return None

        person_box = target["bounding_box"]

        if save_logs:
            # Log inference results for each person
            file_name = os.path.splitext(os.path.basename(image_path))[0]
            coco_result.save(os.path.join("../logs/inference_results", f"{file_name}_person.png"))

        # Run Modanet prediction (single image at a time)
        moda_result = moda_model.predict(image_path, device="cpu", retina_masks=True, verbose=False)[0]

        # Open image safely with context manager (auto-closes image)
        with Image.open(image_path).convert("RGBA") as original_image:

            # Optional for hypothesis testing
            if enable_apply_clahe:
                clahe_image = apply_clahe(original_image)
            else:
                clahe_image = original_image

            # Ensure the result contains masks before processing
            if moda_result.masks is None:
                return None

            # Get connected segments and process clothing items
            processed_segments = get_connected_segments(clahe_image, person_box, moda_result)

            # Create a processed image object and return it
            processed_image = ProcessedImage(image_path, original_image, processed_segments)
            return processed_image

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# @profile
def get_processed_images(image_paths, coco_model_path, moda_model_path, save_logs=False, enable_apply_clahe=True):
    """
    Processes images using YOLO for person detection and clothing segmentation.

    Args:
        image_paths (list): List of image file paths.
        coco_model_path (str): Path to the YOLO model for person detection.
        moda_model_path (str): Path to the YOLO model for clothing detection.
        save_logs (bool): Whether to save logs.
        enable_apply_clahe (bool): Whether to apply clahe.

    Returns:
        list: List of processed images.
    """
    # Load models once
    moda_model = YOLO(moda_model_path)
    coco_model = YOLO(coco_model_path)

    processed_images = []
    for image_path in image_paths:
        processed_image = process_single_image(image_path, coco_model, moda_model, save_logs, enable_apply_clahe)
        if processed_image is not None:
            processed_images.append(processed_image)
    return processed_images