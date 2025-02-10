import os
import pytest
from PIL import Image, ImageChops
from ultralytics import YOLO
import torch

from src.utils.segmentation_utils import get_target_person, extract_segmented_object

# Paths to test images
TEST_IMAGE_DIR = "tests/test_images/input"
GROUND_TRUTH_DIR = "tests/test_images/truths"


# Helper function to compare images
def images_are_similar(img1, img2, tolerance=5):
    """Compares two images and returns True if they are similar within a given tolerance."""
    diff = ImageChops.difference(img1, img2).convert("L")
    # Compute the difference between images
    diff_bbox = diff.getbbox()
    diff_max_extrema = max(diff.getextrema())

    # Check if images are similar
    return diff_bbox is None or diff_max_extrema <= tolerance


@pytest.mark.parametrize("image_name", ["image_segmentation_1.png", "image_segmentation_2.png"])  # Add your test image names
def test_person_extraction(image_name):
    """Test that object extraction segments and crops the image correctly on a person."""

    # Load the test image
    test_image_path = os.path.join(TEST_IMAGE_DIR, image_name)
    assert os.path.exists(test_image_path), f"Test image {test_image_path} not found"

    # Load pretrained YOLO model
    model = YOLO("tests/yolo11x-seg.pt")

    # Run prediction on test image patch, filtering to only person class, (mps enabled for MACOS)
    results = model.predict(test_image_path, classes=[0], device=torch.device("mps"), stream=True)

    # Process results
    for result in results:

        # Load the original image
        original_image = Image.open(test_image_path).convert("RGBA")

        # Extract person by providing segmentation mask and original image
        extracted_image = extract_segmented_object(result.masks.data[0], original_image)

        # Uncomment to save extracted truth
        # extracted_image.save(os.path.join("tests/test_images", image_name))

        # Load the expected ground truth image
        ground_truth_path = os.path.join(GROUND_TRUTH_DIR, f"truth_{image_name}")
        assert os.path.exists(ground_truth_path), f"Ground truth image {ground_truth_path} not found"
        ground_truth_image = Image.open(ground_truth_path).convert("RGBA")

        # Validate extracted image against ground truth
        assert images_are_similar(extracted_image, ground_truth_image), "Extracted image does not match ground truth"

# Todo: write get target person (need images with people in the background, current dataset doesn't support this)