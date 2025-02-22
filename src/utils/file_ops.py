import os
from PIL import Image, ImageDraw, ImageFont
import csv
import shutil

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

def clear_directory(directory_path):
    """
    Deletes all files and subdirectories inside the given directory.
    The directory itself remains.
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
