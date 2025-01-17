from ultralytics.data.converter import convert_coco

# Convert COCO to YOLO
convert_coco(
    labels_dir="annotations/",
    save_dir="coco_converted_val/",
    use_segments=True,
    use_keypoints=False,
    cls91to80=False,
    lvis=False,
)
