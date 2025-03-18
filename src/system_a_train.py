import torchreid
import torch
import glob
from src.utils.validation_utils import evaluate_rank_map, visualize_reid_results

from src.core.dataset_format import EthicalDataset


torchreid.data.register_image_dataset('ethical_reid_dataset', EthicalDataset)

# Data manager
datamanager = torchreid.data.ImageDataManager(
    root='../datasets/system_c', # The root directory of your dataset
    sources='ethical_reid_dataset',  # Indicate custom dataset
    targets='ethical_reid_dataset',
    height=640,
    width=640,
    batch_size_test=100,
)


# Model (using a pre-trained ResNet50)
model = torchreid.models.build_model(
    name='resnet50',
    num_classes=datamanager.num_train_pids,
    pretrained=True, # Load pre-trained weights
)

# extractor = torchreid.utils.FeatureExtractor(
#     model_name='resnet50',
#     model_path='../Training/resnet50.pth',
#     device="cpu" # Change if you have compatible GPUs
# )

model.train()

# query_images = sorted(glob.glob("../datasets/system_c/query/*/*.png"))
# gallery_images = sorted(glob.glob("../datasets/system_c/gallery/*/*.png"))
#
# # Extract features
# query_features = extractor(query_images)
# gallery_features = extractor(gallery_images)
#
# print(f"Extracted {query_features.shape[0]} query features")
# print(f"Extracted {gallery_features.shape[0]} gallery features")
#
# # Normalise features
# query_features = torch.nn.functional.normalize(query_features, dim=1)
# gallery_features = torch.nn.functional.normalize(gallery_features, dim=1)
#
# # Compute similarity
# similarity_matrix = torch.mm(query_features, gallery_features.t())  # Cosine similarity
#
# # Convert to numpy
# similarity_matrix = similarity_matrix.cpu().numpy()
#
# print(similarity_matrix)
#
# # Evaluate Re-ID Performance
# evaluate_rank_map(similarity_matrix, query_images, gallery_images)

# Visualise the results
# visualize_reid_results(query_images, gallery_images, similarity_matrix, top_k=5)
