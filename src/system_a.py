from ultralytics import YOLO
from src.utils.segmentation_utils import extract_segmented_object, get_target_person
from src.utils.histogram_utils import calculate_histogram
import os
import cv2
import torch
from PIL import Image
import numpy as np
from src.core.processed_image import ProcessedImage
from src.core.bucket import Bucket

# Todo: Either use it or lose it
# Function to get all images in a directory
def get_images(input_dir = "images"):
    images = {}
    for file in os.listdir(input_dir):
        # Construct file path and load image
        file_path = os.path.join(input_dir, file)
        file_extension = os.path.splitext(file)[1]
        if file_extension == ".jpg":
            image = Image.open(file_path)
            # image = cv2.imread(file_path) # (B G R) Image
            images[file.title()] = image
    return images

def sort_buckets(bucket_dict):

    while bucket_dict:
        # print(image_group_list)
        # Take the first image group from the list
        current_bucket = bucket_dict.popitem()[1]
        print(f"Popped off bucket #{current_bucket.bucket_id}")

        # Initialise variables to track the most similar match
        best_match = None
        best_similarity = float("inf")

        # Compare the current bucket with all remaining buckets
        for bucket_id, bucket in bucket_dict.items():

            print(f"Checking {bucket_id} against {current_bucket.bucket_id}")
            similarity = bucket.compare_with_bucket(current_bucket)

            if similarity < best_similarity:
                best_similarity = similarity
                best_match = bucket

        # Merge the bucket with the best match
        if best_similarity < 0.36:
            print(f"Best match for {current_bucket.bucket_id} - {best_match.bucket_id} with similarity {best_similarity}")
            best_match.merge_bucket(current_bucket)
            print("Current bucket images:")
            for image in current_bucket.images:
                print(image.image_name)
            print("Best match bucket images:")
            for image in best_match.images:
                print(image.image_name)
        else:
            print(f"No more matches for: {current_bucket.bucket_id}")



def main(image_dir, model_path):
    # Load pretrained YOLO model
    model = YOLO(model_path)

    # Run prediction on directory, filtering to only person class
    results = model.predict(image_dir, classes=[0], device=torch.device("mps"), stream=True)

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

        for section_number, section in processed_image.image_sections.items():
            extracted_bgr = cv2.cvtColor(np.asarray(extracted_image), cv2.COLOR_RGB2BGR)
            hist = calculate_histogram(extracted_bgr)
            # visualise_histogram(hist, title=f"Image Original {image_name} Section {section_number}")

        # Put ALL images into buckets (1 image per bucket)
        # Compare bucket 1 against rest using histogram
        # If similarity score isn't above a threshold, move bucket to completed list?
        # Combine best match into bucket 1
        # Compute average histogram
        # Move bucket 1 to end of list
        # Repeat from step 2

    buckets = {}
    for i, processed_image in enumerate(processed_images):
        bucket = Bucket(i)
        bucket.add_image(processed_image)
        buckets[i] = bucket

    sort_buckets(buckets)


if __name__ == '__main__':
    # image_dir = "../Market-1501-v15.09.15/bounding_box_test"
    image_dir = "../images"
    # model_path = "Training/yolo11x-seg.mlpackage"  # Use a segmentation-trained YOLO model
    model_path = "../Training/yolo11x-seg.pt"  # Use a segmentation-trained YOLO model
    main(image_dir, model_path)