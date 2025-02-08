from dotenv import load_dotenv
from roboflow import Roboflow
import os


def upload_dataset_with_annotations(workspace, dataset_dir, batch_name):
    """
    Uploads an entire dataset of images with an annotation file using Roboflow's `upload_dataset()` function.

    Args:
        workspace (Workspace): The roboflow workspace object.
        dataset_dir (str): Directory containing all the dataset files.
        batch_name (str): Name of the batch for tracking uploads.
    """

    print(f"Uploading dataset from {dataset_dir} as batch '{batch_name}'")

    try:
        workspace.upload_dataset(
            dataset_dir,  # The dataset path
            "re-id-clothing-accessories",  # This will either create or get a dataset with the given ID
            num_workers=25, # 25 is the maximum recommended number of workers
            project_license="MIT",
            project_type="instance-segmentation",
            batch_name=batch_name,
            num_retries=0
        )
        print(f"Finished uploading dataset from directory {dataset_dir}")
    except Exception as e:
        print(f"Error uploading dataset from {dataset_dir}: {e}")


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Access the API key
    api_key = os.getenv("ROBOFLOW_API_KEY")

    # Initialise Roboflow with the API key
    rf = Roboflow(api_key=api_key)

    # Get the workspace
    workspace = rf.workspace("comp303dissertation")

    upload_dataset_with_annotations(workspace, "../../Training/MODANET", "modanet")


if __name__ == "__main__":
    main()
