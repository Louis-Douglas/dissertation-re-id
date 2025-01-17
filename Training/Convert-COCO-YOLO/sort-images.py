import os
import shutil

# Define paths
images_folder = "datasets/coco/images"  # Folder containing the images
labels_folder = "YOLO-11-Modanet/train/labels"  # Folder containing the .txt files
output_folder = "YOLO-11-Modanet/train/images"  # Folder to move matching images

# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get the set of text file names (without extensions)
label_filenames = {os.path.splitext(txt_file)[0] for txt_file in os.listdir(labels_folder) if txt_file.endswith(".txt")}

# Iterate through the image files
for image_file in os.listdir(images_folder):
    # Get the image name without the extension by using splitext and then throwing away the extension
    image_name, _ = os.path.splitext(image_file)

    # Check if there's a matching label file within the labels_folder
    if image_name in label_filenames:
        # Move any matching images to the output folder
        shutil.move(os.path.join(images_folder, image_file), os.path.join(output_folder, image_file))

print("Images with matching text files have been moved.")
