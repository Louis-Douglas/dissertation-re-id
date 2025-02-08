import os
import requests
import threading
from pycocotools.coco import COCO
import numpy as np
import zipfile

# Base directory to store the subset for training
base_directory = 'COCO'

# Desired classes list
desired_classes = [
    'backpack',
    'handbag',
    'bicycle',
    'skateboard',
    'suitcase',
    'umbrella',
    'tennis racket',
    'cell phone'
]

# List of data splits to process
data_splits = ['train2017', 'val2017']


def clean_up(annotations_download_path, annotations_dir):
    """
    Cleans up unnecessary annotation files and the downloaded zip archive.

    This function performs the following operations:
    1. Deletes the downloaded zip file from the specified path.
    2. Removes all files in the annotations directory except for essential JSON files.

    Args:
        annotations_download_path (str): The base path of the downloaded zip archive.
        annotations_dir (str): The directory containing the extracted annotation files.

    Files preserved in the annotations directory:
        - "instances_train2017.json"
        - "instances_val2017.json"

    Files not in the keep list are deleted if they are regular files.

    """
    # Delete the downloaded zip file
    zip_path = f"{annotations_download_path}.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"Deleted {zip_path}")

    # List all files to keep in the annotations directory
    files_to_keep = {"instances_train2017.json", "instances_val2017.json"}

    for file in os.listdir(annotations_dir):
        file_path = os.path.join(annotations_dir, file)

        # Delete only if it's not in the keep list and is a file (not a directory)
        if file not in files_to_keep and os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")


def download_annotations(ann_file):
    file_name = "annotations_trainval2017"
    annotations_download_path = os.path.join(base_directory, file_name)
    annotations_dir = os.path.join(base_directory, "annotations")
    # Skip if extracted annotation file already exists
    if os.path.exists(f"{ann_file}"):
        print(f"Already exists: {file_name}")
        return

    url = f"http://images.cocodataset.org/annotations/{file_name}.zip"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"{annotations_download_path}.zip", 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {file_name}")

            with zipfile.ZipFile(f"{annotations_download_path}.zip", 'r') as zip_ref:
                zip_ref.extractall(base_directory)
                print(f"Extracted {file_name}.zip to {annotations_dir}")
            clean_up(annotations_download_path, annotations_dir)

        else:
            print(f"Failed to download {file_name} from {url} (status code: {response.status_code})")
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")

def download_images(images, data_type, images_dir):
    """
    Downloads a list of images sequentially.
    Checks if an image already exists before downloading.
    """
    for image in images:
        file_name = image['file_name']
        image_path = os.path.join(images_dir, file_name)
        # Skip if image already exists
        if os.path.exists(image_path):
            print(f"Already exists: {file_name}")
            continue

        url = f"http://images.cocodataset.org/{data_type}/{file_name}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {file_name}")
            else:
                print(f"Failed to download {file_name} from {url} (status code: {response.status_code})")
        except Exception as e:
            print(f"Error downloading {file_name}: {e}")


def main():
    for data_type in data_splits:
        # Create output directories for the current split: images and annotations.
        images_dir = os.path.join(base_directory, f'images_{data_type}')
        annotations_dir = os.path.join(base_directory, 'annotations')
        os.makedirs(images_dir, exist_ok=True)

        # Define paths for the annotation file
        annFile = os.path.join(annotations_dir, f'instances_{data_type}.json')

        download_annotations(annFile)

        # Initialize the COCO API for instance annotations
        coco = COCO(annFile)

        # Get category IDs for the desired classes
        catIds = coco.getCatIds(catNms=desired_classes)
        print("Categories: ", catIds)

        # Collect image IDs for any of the desired categories (union over the categories)
        imgId_union = set()
        for catId in catIds:
            imgIds_current = coco.getImgIds(catIds=[catId])
            imgId_union.update(imgIds_current)

        # Convert the set back to a list if needed
        imgIds_union = list(imgId_union)
        print("Number of images with any desired category:", len(imgIds_union))

        # Load image metadata
        images = coco.loadImgs(imgIds_union)

        # Filter out images that already exist in the output folder
        images_to_download = []
        for image in images:
            image_path = os.path.join(images_dir, image['file_name'])
            if not os.path.exists(image_path):
                images_to_download.append(image)

        print(f"Downloading {len(images_to_download)} images for {data_type}...")

        # Set the maximum number of threads to use
        max_threads = 32  # Adjust based on your system resources

        # Split the images list evenly among the threads
        split_images = np.array_split(images_to_download, max_threads)

        # Create and start threads for each chunk of images
        threads = []
        for image_chunk in split_images:
            t = threading.Thread(target=download_images, args=(image_chunk, data_type, images_dir))
            threads.append(t)
            t.start()

        # Wait for all threads to finish
        for t in threads:
            t.join()

if __name__ == "__main__":
    main()