class ProcessedSegment:
    """
    Represents a processed object segment detected by YOLO.

    Attributes:
        image (numpy.ndarray or PIL.Image): The cropped object image.
        class_id (int): Object class ID.
        class_name (str): Object class name.
        yolo_mask (torch.Tensor or numpy.ndarray): Segmentation mask.
        # yolo_confidence (float): Detection confidence score.
        # yolo_bbox (list): Bounding box [x1, y1, x2, y2].
        hist (numpy.ndarray or None): Colour histogram.
        colour_description (str): Description of object colours.
    """
    def __init__(self, image, class_id, class_name, box, mask):
        self.image = image
        self.class_id = class_id
        self.class_name = class_name
        self.yolo_mask = mask
        # self.yolo_confidence = float(box.conf[0].item())  # Confidence score
        # self.yolo_bbox = box.xyxy[0].tolist()
        self.hist = None
        self.colour_description = ""