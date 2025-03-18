import pytest
import numpy as np
import cv2
from PIL import Image
import os

from src.utils.histogram_utils import extract_weighted_color_histogram, compare_emd, apply_clahe, RGBA2LAB

# ----- Test RGBA2LAB -----
def test_RGBA2LAB_single_pixel():
    """
    Test that a single opaque red pixel in an RGBA image is correctly converted to LAB.
    """
    # Create a 1x1 RGBA image with a red pixel (255, 0, 0, 255).
    red = (255, 0, 0, 255)
    image = Image.new("RGBA", (1, 1), red)

    # Compute expected LAB value using OpenCV conversion.
    rgb_array = np.array([[[255, 0, 0]]], dtype=np.uint8)
    expected_lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)

    result = RGBA2LAB(image)
    np.testing.assert_array_equal(result, expected_lab)


def test_RGBA2LAB_from_RGB_mode():
    """
    Test that an image in RGB mode is properly converted to RGBA internally and then processed.
    """
    # Create a 1x1 RGB image with a green pixel.
    green = (0, 255, 0)
    image = Image.new("RGB", (1, 1), green)

    # Expected: Convert green (RGB) to LAB.
    rgb_array = np.array([[[0, 255, 0]]], dtype=np.uint8)
    expected_lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)

    result = RGBA2LAB(image)
    np.testing.assert_array_equal(result, expected_lab)


def test_RGBA2LAB_all_transparent():
    """
    Test that an image with all transparent pixels returns an empty LAB array.
    """
    # Create a 2x2 image with all pixels transparent.
    transparent = (10, 20, 30, 0)  # Colour values are arbitrary because alpha = 0.
    image = Image.new("RGBA", (2, 2), transparent)

    result = RGBA2LAB(image)
    # Should return an empty array with shape (0, 4)
    assert result.shape == (0, 4)


def test_RGBA2LAB_multiple_pixels():
    """
    Test conversion of a multi-pixel RGBA image with different colours.
    """
    # Create a 2x2 RGBA image with different opaque colours.
    arr = np.array([
        [(10, 20, 30, 255), (40, 50, 60, 255)],
        [(70, 80, 90, 255), (100, 110, 120, 255)]
    ], dtype=np.uint8)
    image = Image.fromarray(arr, mode="RGBA")

    # Compute expected LAB values:
    # - Remove alpha channel and reshape RGB pixels to (-1, 1, 3)
    # -1 to determine the correct dimension based on total elements
    # 1 to ensure it keeps a second axis (rows)
    # 3 preserves the 3 colour channels
    rgb_pixels = arr[..., :3].reshape(-1, 1, 3)
    expected_lab = cv2.cvtColor(rgb_pixels, cv2.COLOR_RGB2LAB)

    result = RGBA2LAB(image)
    np.testing.assert_array_equal(result, expected_lab)



# --- Tests for extract_weighted_color_histogram ---

# Directory where test images are stored.
EXTRACT_DATA_DIR = os.path.join(os.path.dirname(__file__), "images/extract_histogram")

def test_extract_histogram_constant():
    """
    Test that a constant image produces a signature with one bin.
    images/extract_histogram/constant.png is a 10x10 fully opaque image with constant colour.
    """
    image_path = os.path.join(EXTRACT_DATA_DIR, "constant.png")
    image = Image.open(image_path).convert("RGBA")

    # Call the function using the real image.
    signature = extract_weighted_color_histogram(image, bins=8)

    # For a constant image, we expect a single nonzero bin.
    # Check that there is exactly one row in the signature and that its weight is 1.
    assert signature.shape[0] == 1, "Expected exactly one bin for a constant image."
    np.testing.assert_allclose(signature[0, 0], 1.0, rtol=1e-5)


def test_extract_histogram_multiple():
    """
    Test that an image with two distinct LAB values produces two bins in the signature.
    images/extract_histogram/multiple.png is a 2x2 fully opaque image with two distinct colours of black and white.
    """
    image_path = os.path.join(EXTRACT_DATA_DIR, "multiple.png")
    image = Image.open(image_path).convert("RGBA")

    signature = extract_weighted_color_histogram(image, bins=2)

    # For bins=2, we expect two nonzero bins.
    assert signature.shape[0] == 2, "Expected two nonzero bins for the multiple-colour image."

    # Additionally, check that each bin's weight is 0.5 (since the image has 4 pixels total).
    np.testing.assert_allclose(np.sort(signature[:, 0]), [0.5, 0.5], rtol=1e-5)


def test_extract_histogram_empty():
    """
    Test that an image with all transparent pixels returns an empty signature.
    """
    image_path = os.path.join(EXTRACT_DATA_DIR, "empty.png")
    image = Image.open(image_path).convert("RGBA")

    signature = extract_weighted_color_histogram(image, bins=8)
    assert signature.shape == (0, 4), "Expected an empty signature for an image with no valid pixels."


# --- Tests for compare_emd using real images ---
# Directory where test images are stored.
EMD_DATA_DIR = os.path.join(os.path.dirname(__file__), "images/compare_emd")

# Define a simple DummySegment class that loads an image from file.
class DummySegment:
    def __init__(self, image_filename):
        image_path = os.path.join(EMD_DATA_DIR, image_filename)
        self.image = Image.open(image_path).convert("RGBA")


def test_compare_emd_both_empty():
    """
    Test that when both segments have an empty signature (using empty.png), compare_emd returns infinity.
    """
    seg_empty1 = DummySegment("empty.png")
    seg_empty2 = DummySegment("empty.png")
    result = compare_emd(seg_empty1, seg_empty2)
    assert result == float("inf"), "Expected infinity when both signatures are empty."


def test_compare_emd_one_empty():
    """
    Test that when one segment has a valid signature and the other is empty, compare_emd returns infinity.
    """
    seg_valid = DummySegment("sig1.png")
    seg_empty = DummySegment("empty.png")
    result = compare_emd(seg_valid, seg_empty)
    assert result == float("inf"), "Expected infinity when one signature is empty."


def test_compare_emd_identical():
    """
    Test that two segments with identical signatures (using same.png) have an EMD of 0.
    """
    seg1 = DummySegment("same.png")
    seg2 = DummySegment("same.png")
    result = compare_emd(seg1, seg2)
    assert np.isclose(result, 0.0, atol=1e-5), f"Expected 0 EMD for identical signatures, got {result}."


def test_compare_emd_different():
    """
    Test that segments producing different signatures (using sig1.png and sig2.png) return a positive EMD.
    The expected value should be based on ground truth computed from these curated images.
    """
    seg1 = DummySegment("sig1.png")
    seg2 = DummySegment("sig2.png")
    result = compare_emd(seg1, seg2)

    expected = 32 # Pre-determined EMD for the two images.
    assert np.isclose(result, expected, atol=1e-5), f"Expected EMD ~{expected}, got {result}."


# ----- Test apply CLAHE -----

def test_output_type_and_size():
    """
    Test that apply_clahe returns a PIL.Image with the same dimensions as the input.
    """
    # Create a constant image (100x100) with a fixed colour.
    color = [100, 150, 200]  # B, G, R values (even though PIL assumes RGB, the function treats the array as BGR)
    img_array = np.full((100, 100, 3), color, dtype=np.uint8)
    image = Image.fromarray(img_array)

    output = apply_clahe(image)

    assert isinstance(output, Image.Image), "Output is not a PIL Image."
    assert output.size == image.size, "Output image size does not match input image size."


def test_constant_image():
    """
    Test that applying CLAHE to a constant image yields an image that is nearly constant.
    """
    # Create a 50x50 constant image (all pixels [128, 128, 128])
    constant_color = [128, 128, 128]
    img_array = np.full((50, 50, 3), constant_color, dtype=np.uint8)
    image = Image.fromarray(img_array)

    output = apply_clahe(image)
    out_array = np.array(output)

    # The output should be nearly constant. Using a tight tolerance.
    # Compare all pixels to the top-left pixel.
    reference = out_array[0, 0]
    assert np.allclose(out_array, reference, atol=1), "Output image is not constant for a constant input."


def test_increases_contrast():
    """
    Test that CLAHE increases the contrast (L channel standard deviation) of a low-contrast image.
    """
    # Create a low-contrast image with slight random noise.
    base_value = 120
    np.random.seed(42)
    noise = np.random.randint(-3, 4, (100, 100, 3), dtype=np.int16)
    img_array = np.clip(base_value + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(img_array)

    # Compute the L channel standard deviation of the original image.
    lab_orig = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
    std_orig = np.std(lab_orig[:, :, 0])

    # Apply CLAHE.
    output = apply_clahe(image)
    out_array = np.array(output)
    lab_out = cv2.cvtColor(out_array, cv2.COLOR_BGR2LAB)
    std_out = np.std(lab_out[:, :, 0])

    assert std_out > std_orig, (
        f"CLAHE did not increase contrast: original L channel std={std_orig}, output L channel std={std_out}"
    )


def test_invalid_input():
    """
    Test that applying CLAHE with an invalid input type raises an exception.
    """
    with pytest.raises(Exception):
        apply_clahe("not an image")
