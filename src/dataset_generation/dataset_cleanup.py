import os
import shutil
import random
from glob import glob
from PIL import Image
from ultralytics import YOLO
import torch

from src.utils.segmentation_utils import get_target_person


def rename_dirs(dataset_path):
    # Get and sort directories in order
    dirs = sorted(glob(os.path.join(dataset_path, '*/')))

    temp_mapping = {}
    for dir_index, dir_path in enumerate(dirs, start=1):
        dir_path = os.path.normpath(dir_path)  # Normalise path (remove trailing `/`)
        dir_name = os.path.basename(dir_path)

        # Generate temporary name
        temp_dir_name = f"tmp_{str(dir_index).zfill(4)}"
        temp_dir_path = os.path.join(dataset_path, temp_dir_name)

        # Rename directory to temp name
        os.rename(dir_path, temp_dir_path)
        temp_mapping[temp_dir_name] = dir_name  # Store original name mapping

    for dir_index, temp_dir_name in enumerate(sorted(temp_mapping.keys()), start=1):
        final_dir_name = f"{str(dir_index).zfill(4)}"  # Format as `0001`, `0002`, ...
        final_dir_path = os.path.join(dataset_path, final_dir_name)

        temp_dir_path = os.path.join(dataset_path, temp_dir_name)

        # Rename temp directory to final name
        os.rename(temp_dir_path, final_dir_path)
        print(f"Renamed {temp_mapping[temp_dir_name]} -> {final_dir_name}")


def rename_files_in_structure(parent_dir):
    person_dirs = []  # List to store valid directories

    for d in os.listdir(parent_dir):
        dir_path = os.path.join(parent_dir, d)

        # Check if it's a directory and has a numeric name
        if os.path.isdir(dir_path) and d.isdigit():
            person_dirs.append(d)

    # Sort the directories numerically
    person_dirs.sort()

    for person in person_dirs:
        dir_path = str(os.path.join(parent_dir, person))
        # Get all image files (.png, .jpg)
        files = sorted(glob(os.path.join(dir_path, "*.png")) + glob(os.path.join(dir_path, "*.jpg")))
        # Rename files to temporary names
        temp_mapping = {}
        for i, file_path in enumerate(files, start=1):
            temp_file_name = f"tmp_{i}.png"
            temp_file_path = os.path.join(dir_path, temp_file_name)
            os.rename(file_path, temp_file_path)
            temp_mapping[temp_file_name] = file_path

        # Rename temp files to final names, e.g., "0001_1.png", "0001_2.png", ...
        for i, temp_file_name in enumerate(sorted(temp_mapping.keys()), start=1):
            final_file_name = f"{person}_{i}.png"
            final_file_path = os.path.join(dir_path, final_file_name)
            temp_file_path = os.path.join(dir_path, temp_file_name)
            os.rename(temp_file_path, final_file_path)
            print(f"Renamed file in {dir_path}: {os.path.basename(temp_mapping[temp_file_name])} -> {final_file_name}")

        # Rename text files appropriately
        txt_files = sorted(glob(os.path.join(dir_path, "*.txt")))
        for txt_file_path in txt_files:
            final_txt_file_name = f"{person}.txt"
            final_txt_file_path = os.path.join(dir_path, final_txt_file_name)
            os.rename(txt_file_path, final_txt_file_path)
            print(f"Renamed text file in {dir_path}: {txt_file_path} -> {final_txt_file_name}")


def restructure_dataset(dataset_path):
    """
    Restructures the dataset at dataset_path into three directories:
    - gallery
    - train
    - query

    Replicate the original person directories (0001, 0002, ...) into both gallery and train directories.
    For each person in gallery, one random image is selected and moved into a corresponding
    person folder inside the query directory (with the naming convention 0001, 0002, ...).
    """
    gallery_dir = os.path.join(dataset_path, "gallery")
    train_dir = os.path.join(dataset_path, "train")
    query_dir = os.path.join(dataset_path, "query")

    os.makedirs(gallery_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)

    # Get list of person directories (exclude gallery, train, query)
    excluded_dirs = {"gallery", "train", "query"}
    person_dirs = []  # List to store valid directories

    # Iterate through items in dataset_path
    for d in os.listdir(dataset_path):
        dir_path = os.path.join(dataset_path, d)

        # Check if it's a directory and not in the excluded list
        if os.path.isdir(dir_path) and d not in excluded_dirs:
            person_dirs.append(d)

    # Copy each person folder into gallery and train
    for person in person_dirs:
        src_path = os.path.join(dataset_path, person)
        dest_gallery = os.path.join(gallery_dir, person)
        dest_train = os.path.join(train_dir, person)
        shutil.copytree(src_path, dest_gallery)
        shutil.copytree(src_path, dest_train)
        print(f"Copied {src_path} to gallery and train.")

    # For each person folder in gallery, select one random image and move it to query
    for person in person_dirs:
        gallery_person_dir = os.path.join(gallery_dir, person)
        query_person_dir = os.path.join(query_dir, person)
        os.makedirs(query_person_dir, exist_ok=True)
        # List all image files in the person's gallery directory
        images = glob(os.path.join(gallery_person_dir, "*.png")) + glob(os.path.join(gallery_person_dir, "*.jpg"))
        if images:
            random_image = random.choice(images)
            # In query, name the file as <person>.png
            query_image_path = os.path.join(query_person_dir, f"{person}.png")
            shutil.move(random_image, query_image_path)
            print(f"Moved random image from {gallery_person_dir} to {query_image_path}")
        else:
            print(f"No images found in {gallery_person_dir} for query creation.")

    # Delete the original person folders in the dataset_path
    for person in person_dirs:
        original_dir = os.path.join(dataset_path, person)
        shutil.rmtree(original_dir)
        print(f"Deleted original person directory: {original_dir}")


def resize_images(dataset_path, target_size=(640, 640)):
    """
    Resizes all images in the dataset directories to a fixed size.
    This function looks into each subdirectory of dataset_path.
    """
    images = glob(os.path.join(dataset_path, "*", "*.png")) + glob(os.path.join(dataset_path, "*", "*.jpg"))
    for image_path in images:
        try:
            with Image.open(image_path) as img:
                if img.size != target_size:
                    resized_img = img.resize(target_size)
                    resized_img.save(image_path)
                    print(f"Resized: {image_path} -> {target_size}")
                else:
                    print(f"Image already correct size: {image_path}")
        except Exception as e:
            print(f"Error resizing {image_path}: {e}")


def create_resnet_directory(source_dir, target_dir):
    """
    Clones the directory and crops all cloned images to detected person bounding box.
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    print(f"Cloned directory: {source_dir} -> {target_dir}")

    image_paths = glob(os.path.join(target_dir, "**", "*.png"), recursive=True) + \
                  glob(os.path.join(target_dir, "**", "*.jpg"), recursive=True)

    coco_model_path = "../../Training/yolo11x-seg.mlpackage"
    coco_model = YOLO(coco_model_path)

    for img_path in image_paths:
        cropped_img = crop_to_person(img_path, coco_model)
        if cropped_img:
            cropped_img.save(img_path)
            print(f"Cropped: {img_path}")
        else:
            print(f"No person detected in {img_path}. Skipping cropping.")


def crop_to_person(image_path, coco_model):
    """
    Detects the largest target person, crops to their bounding box, and resizes.
    """
    coco_results = coco_model.predict(image_path, classes=[0], device=torch.device("mps"), stream=True, retina_masks=True)


    for result in coco_results:
        target_person = get_target_person(result)
        if target_person and target_person.get('bounding_box'):
            x1, y1, x2, y2 = target_person['bounding_box']
            pil_img = Image.open(result.path).convert("RGB")
            cropped_img = pil_img.crop((x1, y1, x2, y2))
            return cropped_img
    return None


def dataset_cleanup():
    """
    Performs dataset cleanup in the following order:
      1. Rename directories (to standardised IDs).
      2. Resize images to a fixed size.
      3. Restructure the dataset into gallery, train, and query splits.
      4. Rename files within the new person directories (in gallery and train).
      5. Create the ResNet directory by cloning the structure and modifying images.
    """
    source_dataset_path = "../../datasets/Ethical-filtered"
    target_dataset_path = "../../datasets/Ethical-filtered-cropped"

    # Rename directories first
    rename_dirs(source_dataset_path)

    # Resize images early
    resize_images(source_dataset_path, (640, 640))

    # Restructure dataset into gallery, train, and query splits
    restructure_dataset(source_dataset_path)

    # Rename files in the new gallery and train directories
    gallery_path = os.path.join(source_dataset_path, "gallery")
    train_path = os.path.join(source_dataset_path, "train")
    rename_files_in_structure(gallery_path)
    rename_files_in_structure(train_path)

    # Create the ResNet directory last (cloning the current structure and cropping to target person bounding boxes)
    create_resnet_directory(source_dataset_path, target_dataset_path)


if __name__ == "__main__":
    dataset_cleanup()
