import os
import pytest
from PIL import Image, ImageChops
from pyasn1_modules.rfc7292 import bagtypes
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import torch
from src.utils.segmentation_utils import get_target_person, extract_segmented_object, get_connected_segments
from torchvision.ops import box_iou

# ----- get target person function -----

def load_image(image_path):
    """Loads an image using OpenCV."""
    return cv2.imread(image_path)


def draw_bounding_box(image, box, color, label="Detected"):
    """Draws a bounding box on an image."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def test_get_target_person():
    """Tests getting the target person by comparing YOLO result output with ground truth."""
    main_image_dir = "images/target_person"
    input_image_path = os.path.join(main_image_dir, "input/test_image.png")
    output_image_path = os.path.join(main_image_dir, "output")
    ground_truth_bbox = [203.9024,  40.6577, 389.4435, 634.5903]
    model_path = "../yolo11x-seg.pt"

    # Load YOLO model
    model = YOLO(model_path)

    # Load image
    image = load_image(input_image_path)

    # Run YOLO inference
    results = model(image)[0]  # First result

    # Run get_target_person function
    detected = get_target_person(results)

    if detected is None:
        print("No person detected!")
        return

    detected_bbox = detected["bounding_box"]

    # Compute IoU
    detected_bbox = torch.tensor([detected_bbox], dtype=torch.float32)
    ground_truth_bbox = torch.tensor([ground_truth_bbox], dtype=torch.float32)
    iou = box_iou(detected_bbox, ground_truth_bbox).item()  # Convert tensor to scalar

    # Draw ground truth and detected bounding box
    draw_bounding_box(image, ground_truth_bbox.squeeze().tolist(), (0, 255, 0), "Ground Truth")
    draw_bounding_box(image, detected_bbox.squeeze().tolist(), (255, 0, 0), "Detected")

    # Compare visually
    bbox_compare_output_path = os.path.join(output_image_path, "output_comparison.png")
    cv2.imwrite(bbox_compare_output_path, image)

    # Visually verify there's other people in the shot
    inference_output_path = os.path.join(output_image_path, "inference.png")
    results.save(inference_output_path)

    # Debug display image
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title(f"IoU: {iou:.2f}")
    # plt.show()

    # Assert IoU threshold (e.g., IoU should be >= 0.5 for a match)
    assert iou >= 0.5, f"Detected person bounding box IoU ({iou:.2f}) is too low!"

    print(f"Test passed! IoU: {iou:.2f}")


# ----- Test person extraction -----

# Paths to test images
TEST_PERSON_EXTRACTION_IMAGES = "images/person_extraction"
TRUTH_PERSON_EXTRACTION_IMAGES = "images/person_extraction/truths"

# Helper function to compare images
def images_are_similar(img1, img2, tolerance=5):
    """Compares two images and returns True if they are similar within a given tolerance."""
    diff = ImageChops.difference(img1, img2).convert("L")
    # Compute the difference between images
    diff_bbox = diff.getbbox()
    diff_max_extrema = max(diff.getextrema())

    # Check if images are similar
    return diff_bbox is None or diff_max_extrema <= tolerance


@pytest.mark.parametrize("image_name", ["image_segmentation_1.png", "image_segmentation_2.png"])
def test_person_extraction(image_name):
    """Test that object extraction segments and crops the image correctly on a person."""

    # Load the test image
    test_image_path = os.path.join(TEST_PERSON_EXTRACTION_IMAGES, "input", image_name)
    assert os.path.exists(test_image_path), f"Test image {test_image_path} not found"

    # Load pretrained YOLO model
    model = YOLO("../yolo11x-seg.pt")

    # Run prediction on test image patch, filtering to only person class, (mps enabled for MACOS)
    results = model.predict(test_image_path, classes=[0], device=torch.device("mps"), stream=True, retina_masks=True)

    # Process results
    for result in results:

        # Load the original image
        original_image = Image.open(test_image_path).convert("RGBA")
        person_data = get_target_person(result)

        # Extract person by providing segmentation mask and original image
        extracted_image = extract_segmented_object(person_data["mask"], original_image)
        extracted_image.save(os.path.join(TEST_PERSON_EXTRACTION_IMAGES, "output", f"{image_name}"))

        # Uncomment to save extracted truth
        # extracted_image.save(os.path.join(TEST_PERSON_EXTRACTION_IMAGES, "truths", image_name))

        # Load the expected ground truth image
        ground_truth_path = os.path.join(TRUTH_PERSON_EXTRACTION_IMAGES, f"truth_{image_name}")
        assert os.path.exists(ground_truth_path), f"Ground truth image {ground_truth_path} not found"
        ground_truth_image = Image.open(ground_truth_path).convert("RGBA")

        # Validate extracted image against ground truth
        assert images_are_similar(extracted_image, ground_truth_image), "Extracted image does not match ground truth"


# ----- Test getting segments connected to person -----
def test_get_connected_segments():
    """Tests extracting objects connected to a person and separating non-connected items."""

    main_image_dir = "images/connected_segments"
    input_dir = os.path.join(main_image_dir, "input")
    output_dir = os.path.join(main_image_dir, "output")

    # Directories for saving the cropped outputs
    connected_dir = os.path.join(output_dir, "connected_items")
    non_connected_dir = os.path.join(output_dir, "non_connected_items")
    os.makedirs(connected_dir, exist_ok=True)
    os.makedirs(non_connected_dir, exist_ok=True)

    # Load the YOLO models
    modanet_model_path = "../modanet-seg.pt"
    moda_model = YOLO(modanet_model_path)

    coco_model_path = "../yolo11x-seg.pt"
    coco_model = YOLO(coco_model_path)

    image_name = "test_image_1"
    input_image_path = os.path.join(input_dir, f"{image_name}.png")

    # Load image using both OpenCV and PIL as needed
    image_cv = cv2.imread(input_image_path)
    image_pil = Image.open(input_image_path).convert("RGBA")

    # Run YOLO inference
    coco_results = coco_model(image_cv)[0]
    moda_results = moda_model(image_cv)[0]

    # Get the target person bounding box using the coco model results
    target_person = get_target_person(coco_results)
    if target_person is None:
        raise AssertionError(f"No person detected in {image_name}!")
    person_bbox = target_person["bounding_box"]

    # Get connected segments from the PIL image, person bounding box, and moda model results
    connected_segments = get_connected_segments(image_pil, person_bbox, moda_results)

    # Separate objects into connected and non-connected lists
    connected_items = []
    non_connected_items = []
    for obj in moda_results.boxes:
        obj_class_id = int(obj.cls[0].item())
        obj_class_name = moda_results.names[obj_class_id]
        obj_bbox = obj.xyxy[0].tolist()  # [x1, y1, x2, y2]

        person_bbox_draw = torch.tensor([person_bbox], dtype=torch.float32)
        obj_bbox_draw = torch.tensor([obj_bbox], dtype=torch.float32)

        draw_image = image_cv.copy()
        draw_bounding_box(draw_image, person_bbox_draw.squeeze().tolist(), (0, 255, 0), "Person")
        draw_bounding_box(draw_image, obj_bbox_draw.squeeze().tolist(), (255, 0, 0), f"{obj_class_name} - {box_iou(person_bbox_draw, obj_bbox_draw).item():.4f}")
        # Compare visually
        bbox_compare_output_path = os.path.join(output_dir, f"output_comparison_{obj_class_name}.png")
        cv2.imwrite(bbox_compare_output_path, draw_image)

        # Check if this object is in one of the connected segments
        is_connected = False  # Initialise as False
        for seg in connected_segments:
            if seg.yolo_box.xyxy[0].tolist() == obj_bbox:
                is_connected = True  # If any segment matches the bbox, mark as connected
                break  # Exit loop early since we found a match

        if is_connected:
            connected_items.append((obj_class_name, obj_bbox))
        else:
            non_connected_items.append((obj_class_name, obj_bbox))

    # Crop and save the connected items
    for i, (class_name, bbox) in enumerate(connected_items):
        x1, y1, x2, y2 = map(int, bbox)
        cropped = image_cv[y1:y2, x1:x2]
        connected_img_path = os.path.join(connected_dir, f"{class_name}_connected_{i}_{image_name}.png")
        cv2.imwrite(connected_img_path, cropped)
        print(f"Saved connected item: {class_name} to {connected_img_path} with bbox: {bbox}")

    # Crop and save the non-connected items
    for i, (class_name, bbox) in enumerate(non_connected_items):
        x1, y1, x2, y2 = map(int, bbox)
        cropped = image_cv[y1:y2, x1:x2]
        non_connected_img_path = os.path.join(non_connected_dir, f"{class_name}_non_connected_{i}_{image_name}.png")
        cv2.imwrite(non_connected_img_path, cropped)
        print(f"Saved non-connected item: {class_name} to {non_connected_img_path} with bbox: {bbox}")

    # Save modanet results for visual verification
    inference_output_path = os.path.join(output_dir, f"moda_inference_{image_name}.png")
    moda_results.save(inference_output_path)

    # Expected truth bounding boxes for connected and non-connected items.
    truth_connected = [
        ("pants", [155.6253662109375, 343.0889892578125, 341.788818359375, 583.724365234375]),
        ("footwear", [141.73104858398438, 557.96826171875, 221.24917602539062, 611.35986328125]),
        ("footwear", [311.79132080078125, 572.38720703125, 398.71673583984375, 613.7119140625]),
        ("bag", [157.1688232421875, 203.13552856445312, 217.32672119140625, 341.4526672363281]),
        ("outer", [185.982666015625, 170.40908813476562, 284.4220275878906, 384.4436950683594]),
        ("bag", [25.147689819335938, 464.5473327636719, 161.8079071044922, 604.21533203125]),
        ("top", [242.1937713623047, 163.2430419921875, 313.47381591796875, 354.104248046875]),
        ("headwear", [240.5162353515625, 90.83261108398438, 334.72723388671875, 166.4210205078125]),
        # ("outer", [564.1175537109375, 293.03289794921875, 607.0076904296875, 353.1461181640625]),
        ("bag", [242.06231689453125, 177.67501831054688, 275.65740966796875, 276.5303955078125]),
        ("top", [242.95523071289062, 343.4818420410156, 277.9419860839844, 367.9126892089844]),
        ("top", [253.90457153320312, 343.52691650390625, 277.3778381347656, 368.85626220703125]),  # Example: a "table" with these coordinates.
    ]
    truth_non_connected = [
        ("outer", [564.1175537109375, 293.03289794921875, 607.0076904296875, 353.1461181640625]),
        ("outer", [563.9481811523438, 292.8067932128906, 609.5321655273438, 388.1569519042969]),
    ]

    # Check if the results match the truth.
    assert connected_items == truth_connected, (
        f"Connected items do not match truth:\nDetected: {connected_items}\nExpected: {truth_connected}"
    )
    assert non_connected_items == truth_non_connected, (
        f"Non-connected items do not match truth:\nDetected: {non_connected_items}\nExpected: {truth_non_connected}"
    )
