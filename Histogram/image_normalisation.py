# https://www.geeksforgeeks.org/normalize-an-image-in-opencv-python/
import os
import cv2

# Input and output directories
input_dir = "images"
output_dir = "equalised"

# List to store all equalised images
normalised_images = {}

for file in os.listdir(input_dir):

    # Construct file path and load image
    file_path = os.path.join(input_dir, file)
    image = cv2.imread(file_path)

    # Convert the image to HSV colour space to obtain brightness channel
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Normalise the brightness channel
    hsv_image[:, :, 2] = cv2.normalize(hsv_image[:, :, 2], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Convert back to BGR colour space
    normalised_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    normalised_images[file] = normalised_image

for image in normalised_images:
    # Save the results
    cv2.imwrite('equalised/' + image, normalised_images[image])