from PIL import Image

# Load the image
image_path = "split-image/segmented_object_0.png"
original_image = Image.open(image_path).convert("RGBA")

# Split the image into 4 equal vertical rectangles
width, height = original_image.size
section_height = height // 4

for i in range(4):
    top = i * section_height # Get the top y value of the current section
    bottom = (i + 1) * section_height # Get the bottom y value of the current section
    cropped_section = original_image.crop((0, top, width, bottom)) # Crop to those y values

    # Save each section as a separate image
    section_output_path = f"split-results/image_section_{i}.png"
    cropped_section.save(section_output_path)
    print(f"Saved section to {section_output_path}")
