import os
import pytest
from PIL import Image, ImageChops
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import torch
from src.utils.segmentation_utils import get_target_person, extract_segmented_object
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

    # Display image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"IoU: {iou:.2f}")
    plt.show()

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