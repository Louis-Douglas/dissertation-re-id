from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os

from google.genai.types import PersonGeneration

from src.dataset_generation.prompt_generator import generate_prompts
safety = True

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# Base output directory
output_base_dir = "output"

# Ensure the base directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Get the next available batch number
existing_folders = [int(folder) for folder in os.listdir(output_base_dir) if folder.isdigit()]
next_batch = max(existing_folders, default=0) + 1
batch_folder = os.path.join(output_base_dir, str(next_batch))

# Create new batch folder
os.makedirs(batch_folder)

prompts = generate_prompts()

all_generated_images = []

for prompt in prompts:
    if safety:
        break
    # Generate images
    response = client.models.generate_images(
        model='imagen-3.0-generate-002',
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="1:1",
            person_generation=PersonGeneration.ALLOW_ADULT,
            guidance_scale=0.5
        )
    )

    all_generated_images.extend(response.generated_images)

# Save images in the batch folder with incremented file names
for i, generated_image in enumerate(all_generated_images):
    image = Image.open(BytesIO(generated_image.image.image_bytes))
    image_path = os.path.join(batch_folder, f"image-{i+1}.png")
    image.save(image_path)
    # image.show()

print(f"Images saved in: {batch_folder}")