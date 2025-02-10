from ultralytics import YOLO
from src.utils.segmentation_utils import extract_segmented_object, get_target_person
import os
import torch
from PIL import Image
from src.core.processed_image import ProcessedImage
from src.core.bucket import Bucket

def get_closest_bucket(bucket_dict, current_bucket):
    """
    Finds the most similar bucket to the given bucket based on histogram comparison.

    Args:
        bucket_dict (dict): Dictionary of buckets to check against.
        current_bucket (Bucket): The bucket to compare against others.

    Returns:
        tuple: The most similar Bucket object and its similarity score.
    """
    # Initialise variables to track the most similar match
    best_match = None
    best_similarity = float("inf")

    # Compare the current bucket with all remaining buckets
    for bucket_id, bucket in bucket_dict.items():

        # print(f"Checking {bucket_id} against {current_bucket.bucket_id}")
        similarity = bucket.compare_with_bucket(current_bucket)

        if similarity < best_similarity:
            best_similarity = similarity
            best_match = bucket
    return best_match, best_similarity


def sort_buckets(bucket_dict):
    """
    Sorts and merges buckets based on similarity.

    Args:
        bucket_dict (dict): Dictionary containing bucket_id mapped to Bucket objects.
    """

    while bucket_dict:
        # Take the first bucket from the dictionary
        current_bucket = bucket_dict.popitem()[1]
        print("----------------------")
        print(f"Popped bucket #{current_bucket.bucket_id}")

        # Get the closest bucket in similarity
        best_match, best_similarity = get_closest_bucket(bucket_dict, current_bucket)

        # Todo: Remove magic number
        if best_similarity < 0.50:
            print(
                f"Best match for {current_bucket.bucket_id} is "
                f"Bucket {best_match.bucket_id} (Similarity: {best_similarity:.2f})")

            print(f"\nCurrent bucket: {current_bucket}")
            print("\nBest match bucket images BEFORE merge:", best_match)


            # Pause and wait for user input before merging
            input("\nPress Enter to merge these buckets...")

            # Merge the two buckets
            best_match.merge_bucket(current_bucket)

            # Show contents after merging
            print("\nBuckets merged! Best match bucket NOW contains:", best_match)

        else:
            print(f"\nNo suitable match found for Bucket {current_bucket.bucket_id}")

        # Pause before continuing to the next step
        input("\nPress Enter to continue to the next bucket...\n")


def main(dataset_dir, model_path):
    # Load pretrained YOLO model
    model = YOLO(model_path)

    # Collect all image paths
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))

    # Run prediction on directory, filtering to only person class
    results = model.predict(image_paths, classes=[0], device=torch.device("mps"), stream=True)

    processed_images = []

    # Process results
    for result in results:
        # Get mask of primary target person for re-identification
        person_data = get_target_person(result)
        image_name = os.path.basename(result.path)
        print(f"Processing {image_name}")

        # Open the original image
        original_image = Image.open(result.path).convert("RGBA")

        # Extract person "object" from the original image, removing background
        extracted_image = extract_segmented_object(person_data["mask"], original_image)

        # Convert the extracted image into a processed image to give us sections
        processed_image = ProcessedImage(result.path, extracted_image)
        processed_images.append(processed_image)

    buckets = {}
    for i, processed_image in enumerate(processed_images):
        bucket = Bucket(i)
        bucket.add_image(processed_image)
        buckets[i] = bucket

    sort_buckets(buckets)


if __name__ == '__main__':
    dataset_dir = "../datasets/Gen-test"
    model_path = "../Training/yolo11x-seg.pt"  # Use a segmentation-trained YOLO model
    main(dataset_dir, model_path)