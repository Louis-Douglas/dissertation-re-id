import os
from PIL import Image
from glob import glob

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

def rename_files(dataset_path):
    # Process previously renamed directories
    dirs = sorted(glob(os.path.join(dataset_path, '*/')))
    for dir_path in dirs:  # Re-fetch sorted dirs
        dir_path = os.path.normpath(dir_path)  # Remove trailing /
        dir_name = os.path.basename(dir_path)
        # Extract the new directory number
        dir_num = int(dir_name)  # Convert "0001" to 1

        # Get all image files (.png, .jpg)
        files = sorted(glob(f"{dir_path}/*.png") + glob(f"{dir_path}/*.jpg"))

        # Rename files to temporary names (`tmp_xxx`)
        temp_mapping = {}
        for i, file_path in enumerate(files, start=1):
            temp_file_name = f"tmp_{i}.png"
            temp_file_path = os.path.join(dir_path, temp_file_name)

            # Rename to temporary name
            os.rename(file_path, temp_file_path)
            temp_mapping[temp_file_name] = file_path  # Store original names

        # Rename temp files to final names (`1_1.png`, `1_2.png`, ...)
        for i, temp_file_name in enumerate(sorted(temp_mapping.keys()), start=1):
            final_file_name = f"{dir_num}_{i}.png"
            final_file_path = os.path.join(dir_path, final_file_name)

            temp_file_path = os.path.join(dir_path, temp_file_name)

            # Rename temp file to final name
            os.rename(temp_file_path, final_file_path)
            print(f"Renamed {os.path.basename(temp_mapping[temp_file_name])} -> {final_file_name}")



def resize_images(dataset_path, target_size=(640, 640)):
    """
    Resizes all images in the dataset directories to a fixed size.

    Args:
        dataset_path (str): Path to the dataset containing directories.
        target_size (tuple[int, int]): Target size (width, height) for resizing images.
    """
    # Get all images in dataset path
    images = glob(f"{dataset_path}/*/*.png") + glob(f"{dataset_path}/*/*.jpg")

    for image_path in images:
        try:
            # Open and resize image properly
            with Image.open(image_path) as image:
                resized_image = image.resize(target_size)

                # Save resized image back to the same file
                resized_image.save(image_path)
                print(f"Resized: {image_path} -> {target_size}")

        except Exception as e:
            print(f"Error resizing {image_path}: {e}")


# Rename images & directories. Resize images to 640x640 and convert to png
def dataset_cleanup(dataset_path="../../datasets/Ethical-original"):
    rename_dirs(dataset_path)
    rename_files(dataset_path)
    resize_images(dataset_path, (640, 640))

if __name__ == "__main__":
    dataset_cleanup()