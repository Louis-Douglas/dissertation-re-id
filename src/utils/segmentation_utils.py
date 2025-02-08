from torchvision.ops import box_iou, distance_box_iou
from PIL import Image
import numpy as np
import torch
import cv2

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


# TODO: Setup unit test for if objects are connected and test against these thresholds to help fine tune
def is_connected(box1, box2, iou_threshold=0.1, distance_threshold=0.1):
    """
    Determines whether two bounding boxes are considered connected based on
    Intersection over Union (IoU) and center distance.

    Args:
        box1 (Tensor array): The first bounding box in the format (x1, y1, x2, y2).
        box2 (Tensor array): The second bounding box in the same format as box1.
        iou_threshold (float, optional): The minimum IoU required for boxes to be considered connected.
                                         Defaults to 0.1.
        distance_threshold (float, optional): The maximum center distance allowed for boxes to be considered
                                              connected. Defaults to 0.1.

    Returns:
        bool: True if the boxes are considered connected, False otherwise.
    """
    # Get intersection over union of the two boxes
    iou = box_iou(box1, box2)

    # Calculate center distance between two boxes
    distance = distance_box_iou(box1, box2)

    # Check IoU and distance thresholds
    return iou > iou_threshold or distance < distance_threshold

def get_connected_objects(person_box, result, iou_threshold=0.1, distance_threshold=0.1):
    """
    Identifies all objects that are 'connected' to a person in a YOLO segmentation result.

    Args:
        person_box (Tensor array): The bounding box of the person.
        result (ultralytics.engine.results.Results): A single YOLO result.
        iou_threshold (float, optional): The minimum IoU required for an object to be connected to a person.
        distance_threshold (float, optional): The maximum Distance IoU allowed for connection.

    Returns:
        dict or None: A dictionary containing:
            - 'person_box': Bounding box of the detected person.
            - 'connected_objects': List of connected objects with class names and bounding boxes.
          Returns None if multiple or no people are detected.
    """
    boxes = result.boxes  # Bounding boxes for detected objects
    class_names = result.names  # Class names from model

    tensor_person_box = torch.tensor(person_box, dtype=torch.float32)  # Convert to tensor

    connected_objects = []  # Store connected object details

    for box in boxes:
        class_id = int(box.cls[0].item())  # Class ID of the detection
        class_name = class_names[class_id]  # Class name

        if class_name == "person":
            continue  # Skip the person itself

        # Get bounding box of current object
        tensor_object_box = torch.tensor([box.xyxy[0].tolist()], dtype=torch.float32)

        # Check if object is connected to the person
        if is_connected(tensor_person_box, tensor_object_box, iou_threshold, distance_threshold):
            connected_objects.append({
                "class": class_name,
                "bounding_box": box.xyxy[0].tolist(),
                "confidence": box.conf[0].item()
            })

    return {
        "person_box": person_box,
        "connected_objects": connected_objects
    }