# COMP303-2203712
This dissertation is submitted as a requirement for the degree of Bachelor of Science at Falmouth University. It presents work conducted exclusively by the author except where indicated in the text. The report may be freely copied and distributed provided the source is acknowledged.
## Person Re-Identification using You Only Look Once (YOLO) Segmentation and Colour Analysis:
This dissertation introduces a novel approach to person re-identification utilising both YOLOv11 object segmentation trained on the Modanet clothing dataset and performing colour comparison between those clothing objects.\
By giving query images for the system to match in a gallery, we can calculate the Rank1, Rank 5 and mean average precision (mAP) of this system compared to traditional systems such as Resnet50.
## Deployment Instructions

### Obtaining Modanet Model Weights:
#### Option 1: Download
Download the Modanet weights from [Release v1.0](https://github.falmouth.ac.uk/GA-Undergrad-Student-Work-24-25/COMP303-277982/releases/tag/v1.0)\
Choose either .pt for most systems or coreml for macOS

#### Option 2: Training from scratch
1. `pip install maskrcnn-modanet` (https://github.com/cad0p/maskrcnn-modanet (MIT License))
2. run `maskrcnn-modanet datasets download`
3. Download only the annotated version (52k images)
4. Modify and run the `upload_roboflow_dataset.py` script to upload Modanet to Roboflow.
5. Use Jypiter notebook in `src/processing/Modanet_Training_Roboflow.ipynb` on preferred platform for training (Recommended is Google Colab).
6. Run `yolo export model=<weights_file>.pt format=coreml` to convert the weights to coreml (if on macOS)

Alternatively use the generated publicly available dataset on roboflow [here](https://universe.roboflow.com/comp303dissertation/re-id-clothing-accessories-0w4kl/dataset/1) and skip to step 5.

Converting YOLO weights to coreml example:\
`yolo export model=yolo11x-seg.pt format=coreml`

The required MS COCO (Microsoft Common Objects in Context) weights will download automatically when running the system, although these will also need to be converted to coreml if running on macOS.

### Obtaining ResNet Weights:
Download from [here](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html)\
Place in weights directory

### Downloading Datasets
Download the filtered and filtered-cropped datasets from [Release v1.0](https://github.falmouth.ac.uk/GA-Undergrad-Student-Work-24-25/COMP303-277982/releases/tag/v1.0) and put them in the datasets directory in the repository.

Or generate your own using the imagen3_generate code in the dataset_generation directory.
