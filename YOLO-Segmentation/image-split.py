from PIL import Image
import cv2
from Histogram.utils import image_split

# Load the image
image_path = "segmentation-crop-results/segmented_object_0.png"
original_image = Image.open(image_path).convert("RGBA")

split_images = image_split(original_image)

for number, section in split_images.items():
    # Save each section as a separate image
    section_output_path = f"split-results/image_section_{number}.png"
    section.save(section_output_path)
    print(f"Saved section to {section_output_path}")
