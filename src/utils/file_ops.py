import os
from PIL import Image, ImageDraw, ImageFont
import csv
import shutil
from collections import defaultdict
import glob
import re
import cv2

def load_image(image_path):
    """Loads an image and converts it to RGB format for Matplotlib."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return img


def get_images(input_dir):
    """
    Loads all images from a specified directory and returns them in a list.

    Args:
        input_dir (str): The directory containing images.

    Returns:
        dict: A dictionary where keys are image filenames and values are PIL Image objects.
    """
    images = {}
    for file in os.listdir(input_dir):
        # Construct file path and load image
        file_path = os.path.join(input_dir, file)
        if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
            image = Image.open(file_path)
            images[file.title()] = image
    return images

def ensure_directory_exists(directory):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_dominant_colors_to_csv(dominant_colors, csv_filename="dominant_colors.csv"):
    """
    Saves the dominant colours dictionary to a CSV file.

    Args:
        dominant_colors (dict): Dictionary mapping class names to a list of hex colour codes.
        csv_filename (str): Name of the output CSV file.

    Based on guidance from:
    https://realpython.com/python-csv/
    https://docs.python.org/3/library/csv.html
    """
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "Hex Colors"])  # Header

        for cls, hex_codes in dominant_colors.items():
            writer.writerow([cls, ",".join(hex_codes)])  # Store colours as comma-separated values

    print(f"Saved dominant colors to {csv_filename}")

def load_dominant_colors_from_csv(csv_filename="dominant_colors.csv"):
    """
    Loads the dominant colours dictionary from a CSV file.

    Args:
        csv_filename (str): Name of the input CSV file.

    Returns:
        dict: Dictionary mapping class names to a list of hex colour codes.
    Based on guidance from:
    https://docs.python.org/3/library/csv.html
    """
    dominant_colors = {}

    with open(csv_filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            class_name = row[0]
            if row[1]:
                hex_codes = row[1].split(",")
            else:
                hex_codes = []  # Convert back to list
            dominant_colors[class_name] = hex_codes

    print(f"Loaded dominant colors from {csv_filename}")
    return dominant_colors

def clear_directory(directory_path):
    """
    Deletes all files and subdirectories inside the given directory.
    The directory itself remains.

    Based on guidance from:
    https://pynative.com/python-delete-files-and-directories/
    """
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory and its contents
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")


def save_comparison_image(image1, image2, class_name, output_path, similarity_score, label1="Image 1",
                          label2="Image 2"):
    """
    Merges two images side by side, adds labels, and displays the similarity score.

    Args:
        image1 (PIL.Image.Image): First image to compare.
        image2 (PIL.Image.Image): Second image to compare.
        class_name (str): The class of the object (e.g., "Top", "Footwear").
        output_path (str): Path to save the merged image.
        similarity_score (float): Similarity score between the two images.
        label1 (str): Label for the first image (default "Image 1").
        label2 (str): Label for the second image (default "Image 2").

    Based on guidance from:
    https://pillow.readthedocs.io/en/stable/handbook/tutorial.html
    https://cloudinary.com/guides/image-effects/a-guide-to-adding-text-to-images-with-python
    https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    """
    # Ensure both images are the same height (resize the shorter one)
    max_height = max(image1.height, image2.height)
    image1 = image1.resize((image1.width, max_height))
    image2 = image2.resize((image2.width, max_height))

    # Define minimum space for text
    text_padding = max(50, max_height // 10)  # Ensure enough space for text (at least 50px or 10% of height)

    # Create a blank canvas with extra space for text
    merged_width = max(image1.width + image2.width, 100)
    merged_height = max_height + text_padding  # Ensure dynamic padding
    merged_image = Image.new("RGBA", (merged_width, merged_height), (255, 255, 255, 255))

    # Paste images side by side
    merged_image.paste(image1, (0, text_padding))
    merged_image.paste(image2, (image1.width, text_padding))

    # Create a drawing context
    draw = ImageDraw.Draw(merged_image)

    # Define font (adjust dynamically based on image width)
    try:
        font_size = max(12, merged_width // 25)  # Dynamically scale font size based on image width
        font = ImageFont.truetype("arial.ttf", font_size)  # Windows
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)  # Linux/Mac
        except IOError:
            font = ImageFont.load_default()

    # Calculate text positions dynamically
    middle_x = (merged_width // 2) - font_size
    left_x = max(5, image1.width // 10)  # Adjust for narrow images

    # Add labels and similarity score
    draw.text((middle_x, 5), f"{label1} vs {label2}", fill="black", font=font)
    draw.text((left_x, text_padding // 3), f"Class: {class_name}", fill="black", font=font)
    draw.text((left_x, text_padding // 1.5), f"Similarity: {similarity_score}", fill="black", font=font)

    # Save the merged image
    merged_image.save(output_path)

def image_id_sort_key(s):
    """
    Sort helper that extracts the image ID (the number after '_') from a filename.
    """
    match = re.split('([0-9]+)', s)[5] # Regex match splitting numbers and text, get 6th element which is photo id
    if match:
        return int(match)
    return float('inf')  # Return infinity if no match is found


def split_gallery_evenly(gallery_dir, increments):
    """
    Splits a galleries subdirectories into subsets of increasing size.
    Each subset is returned as a list of image paths.

    Args:
        gallery_dir (str): Path to the gallery where each subdirectory corresponds to one person.
        increments (list of int): A list indicating the number of images per person for each increment.

    Returns:
        list of list: Each element is a list of image paths representing a gallery subset.
    """
    # Build a mapping: person_id -> list of image paths
    gallery = defaultdict(list)
    for person_id in os.listdir(gallery_dir):
        person_path = os.path.join(gallery_dir, person_id)
        if os.path.isdir(person_path):
            image_paths = sorted(glob.glob(os.path.join(person_path, "*.png")), key=image_id_sort_key)
            gallery[person_id] = image_paths

    gallery_splits = []
    for n in increments:
        split_list = []
        for person_id, image_paths in gallery.items():
            # For each person, select the first n images, or all images if fewer than n exist
            split_list.extend(image_paths[:n])
        gallery_splits.append(split_list)
    return gallery_splits