import os

class ProcessedImage:
    def __init__(self, image_path, extracted_person, image_segments):
        """
        Initialises an ImageData object.

        Args:
            image_path (str): Path to the image file.
            extracted_person (PIL.Image.image): Image of person extracted from the background.
            image_segments (Dictionary): All the segments extracted from the image.
        """
        self.image_path = image_path
        self.image_name = os.path.basename(image_path)  # Extracts just the filename
        self.extracted_person = extracted_person  # Store the image
        self.image_segments = image_segments