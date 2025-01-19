from ultralytics import YOLO
import numpy as np
from PIL import Image
import torch
from torchvision.ops import box_iou, distance_box_iou, generalized_box_iou, complete_box_iou

# Load pretrained YOLO model
model = YOLO("yolo11x-seg", task="segmentation")

# Load and process image
image_path = "images/image4.jpg"
# results = model([image_path], classes=[0])

# TODO: Accept input for classes we want, maybe an enum?
results = model.predict(image_path, classes=[1, 0, 24, 26, 28])

# Open the original image
original_image = Image.open(image_path).convert("RGBA")

# TODO: Input accept masks/result

# TODO: Setup unit test for if objects are connected and test against these thresholds to help fine tune
def is_connected(box1, box2, iou_threshold=0.1, distance_threshold=0.1):

    # Get intersection over union of the two boxes
    iou = box_iou(box1, box2)

    # Calculate center distance between two boxes
    distance = distance_box_iou(box1, box2)

    # Check IoU and distance thresholds
    return iou > iou_threshold or distance < distance_threshold


# Process results
for result in results:
    masks = result.masks  # Get segmentation masks
    boxes = result.boxes  # Bounding boxes for detected objects
    class_names = result.names  # Get class names
    print(class_names)

    # TODO: Check for multiple or no people in photo, return if so
    person_box = None

    # Store the bounding box of the person
    for box in boxes:
        class_id = int(box.cls[0].item())  # Class ID of the detection
        if class_names[int(class_id)] == "person":
            person_box = box.xyxy[0].tolist()

    # Iterate over the bounding boxes found and output iou amount/distance
    for box in boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Format: [x1, y1, x2, y2]
        confidence = box.conf[0].item()  # Confidence score of the detection
        class_id = int(box.cls[0].item())  # Class ID of the detection

        # Create PyTorch tensors of bounding box coordinates
        tensor_person_box = torch.tensor([person_box], dtype=torch.float32)
        tensor_accessory_box = torch.tensor([box.xyxy[0].tolist()], dtype=torch.float32)

        # Calculate the intersection over union for each box against the person
        iou = box_iou(tensor_person_box, tensor_accessory_box)

        # Calculate the Distance IoU between the centers of the bounding boxes
        diou = distance_box_iou(tensor_person_box, tensor_accessory_box)

        # Calculate the generalised IoU between the centers of the bounding boxes
        giou = generalized_box_iou(tensor_person_box, tensor_accessory_box)

        # Calculate the complete IoU between the centers of the bounding boxes
        ciou = complete_box_iou(tensor_person_box, tensor_accessory_box)


        print(f"Bounding Box: ({x1}, {y1}, {x2}, {y2}),"
              f" Confidence: {confidence},"
              f" Class: {class_names[int(class_id)]},"
              f" IoU: {iou.item()},"
              f" GIoU: {giou.item()},"
              f" CIoU: {ciou.item()},"
              f" Is connected: {is_connected(tensor_person_box, tensor_accessory_box).item()},"
              f" Distance IoU: {diou.item()}")


    # TODO: Determine correct amount of distance IoU to accept/reject objects

    if masks is not None:
        for i, mask in enumerate(masks.data):

            # Convert mask to a numpy array
            mask_array = mask.cpu().numpy().astype(np.uint8) * 255

            # Create a black and white PIL image from the mask array
            mask_image = Image.fromarray(mask_array, mode="L")

            # Create a blank RGBA image with transparent background
            cropped_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))

            # Ensure mask dimensions match the original image dimensions
            mask_image = mask_image.resize(original_image.size)

            # Paste the original image onto the blank image, using the mask as the alpha channel
            cropped_image.paste(original_image, mask=mask_image)

            # Crop the result to the bounding box of the mask
            bbox = mask_image.getbbox()
            if bbox:
                cropped_image = cropped_image.crop(bbox)

            # Save the resulting cropped objects in segmentation-crop-results directory
            # Todo: make this return the images with data attached
            output_path = f"segmentation-crop-results/segmented_object_{i}.png"
            cropped_image.save(output_path)
            print(f"Saved cropped image to {output_path}")
    result.save("segmentation-crop-results/bounding.png")
    print(f"Saved bounding box image to segmentation-crop-results/bounding.png")