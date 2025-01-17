import os
import cv2

# Input and output directories
input_dir = "images"
output_dir = "equalised"

# List to store all equalised images
equalized_images = {}

for file in os.listdir(input_dir):

    # Construct the full file path
    file_path = os.path.join(input_dir, file)

    # Load the image
    image = cv2.imread(file_path)

    # Convert the image to HSV colour space to obtain brightness channel
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Equalise the histograms Y (brightness) channel
    hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])

    # Convert back to BGR colour space
    equalized_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    equalized_images[file] = equalized_image

for image in equalized_images:
    cv2.compareHist(input_dir + '/' + image, equalized_images[image], cv2.NORM_MINMAX)
    # Save the results
    cv2.imwrite('equalised/' + image, equalized_images[image])