import os

class ProcessedImage:
    def __init__(self, image_path, image_object):
        """
        Initialises an ImageData object.

        Args:
            image_path (str): Path to the image file.
            image_object (PIL.Image.image): Extracted mage file.
        """
        self.image_path = image_path
        self.image_name = os.path.basename(image_path)  # Extracts just the filename
        self.image_object = image_object  # Store the image
        self.image_sections = self._split_image_into_sections()  # Stores 4 sections

    def _split_image_into_sections(self):
        """
        Splits the image into 4 equal vertical sections.

        Returns:
            dict: A dictionary containing 4 sections of the image with keys 1, 2, 3, and 4.
        """
        width, height = self.image_object.size
        section_height = height // 4  # Divide the height into 4 equal parts
        sections = {}

        for i in range(4):
            top = i * section_height  # Get the top y value of the current section
            bottom = (i + 1) * section_height # Get the bottom y value of the current section
            cropped_section = self.image_object.crop((0, top, width, bottom))  # Crop to those y values
            sections[i] = cropped_section

        return sections


    def get_section(self, section_number):
        """
        Retrieves a specific section of the image.

        Args:
            section_number (int): Section number (0-3).

        Returns:
            PIL.Image: The cropped section of the image.
        """
        return self.image_sections.get(section_number, None)

