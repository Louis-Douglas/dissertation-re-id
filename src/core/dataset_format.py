import os
import glob
import torchreid

# Modified from: https://kaiyangzhou.github.io/deep-person-reid/user_guide.html#use-your-own-dataset
class EthicalDataset(torchreid.data.ImageDataset):
    # Get the absolute path of the directory where THIS script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Define the project root
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

    # Now define paths relative to the project root
    dataset_dir = os.path.join(PROJECT_ROOT, "datasets", "Torch-Dataset")

    def __init__(self, root='', **kwargs):
        self.dataset_dir = root if root else self.dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, 'train')    # Training dataset
        self.query_dir = os.path.join(self.dataset_dir, 'query')    # Query dataset
        self.gallery_dir = os.path.join(self.dataset_dir, 'gallery') # Gallery dataset

        # Process training, query, and gallery directories
        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(EthicalDataset, self).__init__(train, query, gallery, **kwargs)

    # Ensure all person ids are sequential and unique
    def remap_ids(self, img_paths):
        # Create set of all person id's by folder name
        pid_set = set()
        for img_path in img_paths:
            pid = int(os.path.basename(os.path.dirname(img_path)))  # Extract person ID from folder name
            pid_set.add(pid)

        # Sort unique person IDs
        sorted_pid_list = sorted(pid_set)

        # Create a dictionary mapping old PIDs to new sequential labels
        pid2label = {}
        for label, pid in enumerate(sorted_pid_list):
            pid2label[pid] = label  # Assign new sequential label

        return pid2label

    # Create the list of tuples containing image paths, person ids and cam ids
    def process_dir(self, dir_path, relabel=False):
        print(f"Checking directory: {dir_path}")  # Debugging line

        # Fetch all image paths
        img_paths = (glob.glob(os.path.join(dir_path, '*/*.png')) +
                    glob.glob(os.path.join(dir_path, '*/*.jpg')) +
                    glob.glob(os.path.join(dir_path, '*/*.jpeg')))

        # Catch if no images in dir_path
        if len(img_paths) == 0:
            print(f"Warning: No images found in {dir_path}")
            return []

        # Get remapped person id to sequential label dictionary
        pid2label = self.remap_ids(img_paths)

        data = []
        for img_path in img_paths:
            pid = int(os.path.basename(os.path.dirname(img_path)))
            camid = 0  # Assign a dummy camera ID (defaulting to 0 due to time limitations)
            if relabel:
                pid = pid2label[pid]  # Convert PID to label for training
            data.append((img_path, pid, camid))

        print(f"Loaded {len(data)} images from {dir_path}")
        return data