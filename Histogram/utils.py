import cv2

# Calculate the histogram for each colour channel, return as a list
def calculate_histogram(image):
    histograms = []
    colours = ("b", "g", "r")
    for channel, colour in enumerate(colours):
        hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
        histograms.append(hist)
    return histograms