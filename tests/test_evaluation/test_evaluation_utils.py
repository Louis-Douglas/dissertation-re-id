import numpy as np
import pytest
import matplotlib.pyplot as plt
from PIL import Image

from src.utils.evaluation_utils import visualize_reid_results, evaluate_rank_map_per_query


# Helper function to create a dummy image.
def create_dummy_image(save_path, size=(10, 10), color=(255, 0, 0, 255)):
    """
    Creates and saves a simple RGBA image with a solid color.
    """
    img = Image.new("RGBA", size, color)
    img.save(save_path)


# ----- Tests for evaluate_rank_map_per_query -----

def test_evaluate_rank_map_per_query():
    # Create dummy file paths.
    query_images = [
        "0001/1_1.jpg",
        "0002/2_1.jpg"
    ]
    gallery_images = [
        "0001/0001.jpg",
        "0002/0002.jpg",
        "0001/0001.jpg",
        "0002/0002.jpg"
    ]

    similarity_matrix = np.array([
        [0.9, 0.5, 0.7, 0.3],
        [0.9, 0.8, 0.7, 0.6]
    ])

    # Expected outputs
    expected_rank1 = np.array([1, 0])
    expected_rank5 = np.array([1, 1])
    expected_ap = np.array([1.0, 0.5])

    # Call the function with the test data.
    rank1, rank5, ap = evaluate_rank_map_per_query(similarity_matrix, query_images, gallery_images)

    # Use numpy testing utilities for array comparisons.
    np.testing.assert_array_equal(rank1, expected_rank1)
    np.testing.assert_array_equal(rank5, expected_rank5)
    np.testing.assert_array_almost_equal(ap, expected_ap, decimal=5)

# ----- Tests for visualize_reid_results -----

# Define a dummy load_image function that returns a dummy image array.
def dummy_load_image(path):
    # Returns a simple 10x10 white image.
    return np.ones((10, 10, 3)) * 255


def test_visualize_reid_results(monkeypatch):
    # Override plt.show so that no figures are actually rendered.
    monkeypatch.setattr(plt, "show", lambda: None)
    # Override load_image to avoid file I/O.
    monkeypatch.setattr("src.utils.evaluation_utils.load_image", dummy_load_image)

    # Use dummy file paths so the files don't need to exist on disk.
    query_images = ["0001/0001.png", "0002/0002.png"]
    gallery_images = [
        "0001/1_1.png", "0001/1_2.png",
        "0002/2_1.png", "0002/2_2.png"
    ]

    # Create a dummy similarity matrix.
    similarity_matrix = np.array([
        [0.9, 0.8, 0.2, 0.1],
        [0.1, 0.2, 0.7, 0.6]
    ])

    # Run the visualisation function. If no errors occur, the test passes.
    visualize_reid_results(query_images, gallery_images, similarity_matrix, top_k=2)