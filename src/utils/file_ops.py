import os
import PIL.Image as Image
import csv

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
            # image = cv2.imread(file_path) # (B G R) Image
            images[file.title()] = image
    return images

def ensure_directory_exists(directory):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_dominant_colors_to_csv(dominant_colors, csv_filename="dominant_colors.csv"):
    """
    Saves the dominant colors dictionary to a CSV file.

    Args:
        dominant_colors (dict): Dictionary mapping class names to a list of hex colour codes.
        csv_filename (str): Name of the output CSV file.
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
