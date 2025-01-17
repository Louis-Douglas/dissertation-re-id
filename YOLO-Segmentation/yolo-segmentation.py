from ultralytics import YOLO

# Loan pretrained YOLO11x model (x is the most powerful version)
model = YOLO("yolo11x-seg")

# Return a list of object detections
results = model(["images/image5.jpg"])

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
    
    # Result should be segmentation and bounding boxes around a person and bag