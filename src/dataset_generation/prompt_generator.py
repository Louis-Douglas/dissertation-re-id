import random
import os
from src.utils.file_ops import load_dominant_colors_from_csv

# Define predetermined attribute options
attributes_options = {
    "Age": ["Younger adult", "Middle aged", "Elderly"], # (Cannot add child/teenager due to Imagen3's content policies)
    "Hair": ["Fine straight", "Thick straight", "Fine curly", "Thick curly", "Fine wavy", "Thick wavy"],
    "Skin": ["Fair", "Tan", "Dark", "Olive"],
    "Facial Expression": [
        "Neutral",
        "Smiling",
        "Serious",
        "Thoughtful"
    ],
    "Weight": ["Slender weight", "Average weight", "Heavyset", "Athletic build"]
}

# Clothing options separated by gender. If an item is not applicable, the only option is "None".
clothing_options = {
    "Headwear": {
        "Male": ["None", "Baseball cap", "Beanie"],
        "Female": ["None", "Sun hat", "Beanie"]
    },
    "Eyewear": {
        "Male": ["None", "Sunglasses", "Reading glasses"],
        "Female": ["None", "Sunglasses", "Reading glasses"]
    },
    "Scarf": {
        "Male": ["None", "Casual scarf"],
        "Female": ["None", "Silk scarf", "Winter scarf"]
    },
    "Tie": {
        "Male": ["None", "Striped tie", "Solid tie"],
        "Female": ["None"]  # Typically not used for females
    },
    "Top": {
        "Male": ["T-shirt", "Polo shirt", "Shirt", "Collared Shirt"],
        "Female": ["Blouse", "Top", "Crop top"]
    },
    "Overwear": {
        "Male": ["None", "Light jacket", "Blazer", "Winter coat"],
        "Female": ["None", "Light jacket", "Cardigan", "Winter coat"]
    },
    "Dress": {
        "Male": ["None"],
        "Female": ["Floral dress", "Evening gown", "Casual dress"]
    },
    "Skirt": {
        "Male": ["None"],
        "Female": ["Mini skirt", "Long skirt", "Pleated skirt"]
    },
    "Pants": {
        "Male": ["Jeans", "Chinos", "Formal trousers"],
        "Female": ["Jeans", "Chinos", "Formal trousers"]
    },
    "Shorts": {
        "Male": ["Cargo shorts", "Sport shorts"],
        "Female": ["Denim shorts", "Sport shorts"]
    },
    "Footwear": {
        "Male": ["sneakers", "walking boots", "loafers"],
        "Female": ["sneakers", "high heels", "ballet flats"]
    }
}

# Accessory options (maximum of one will be chosen)
accessory_options = {
    "Backpack": {
        "Male": ["Simple backpack", "Sport backpack"],
        "Female": ["Stylish backpack"]
    },
    "Handbag": {
        "Male": ["None"],
        "Female": ["Elegant handbag", "Casual tote"]
    },
    "Bicycle": {
        "Male": ["Vintage bicycle", "Sport bicycle"],
        "Female": ["Vintage bicycle", "Sport bicycle"]
    },
    "Skateboard": {
        "Male": ["bottomed skateboard"],
        "Female": ["bottomed skateboard"]
    },
    "Suitcase": {
        "Male": ["Travel suitcase"],
        "Female": ["Travel suitcase"]
    },
    "Umbrella": {
        "Male": ["umbrella"],
        "Female": ["umbrella"]
    },
    "Tennis racket": {
        "Male": ["Professional tennis racket"],
        "Female": ["Professional tennis racket"]
    },
    "Cell phone": {
        "Male": ["Latest smartphone"],
        "Female": ["Latest smartphone"]
    }
}

# Profile options for changing the side of the person
profile_options = [
    "A left-side profile angle",
    "A right-side profile angle",
    "A back profile angle",
    "A front facing profile angle"
]

# Location options to vary background
location_options = [
    "Shopping Centre Hallway - A spacious, modern shopping centre with glossy tiled floors reflecting bright, evenly distributed ceiling lights. Lined with glass storefronts showcasing mannequins and promotional banners. Potted plants and sleek benches enhance the atmosphere. Lighting is soft and even.",
    "Urban Street - A bustling city street with neon signs, busy traffic, and people hustling by. The environment is vibrant with graffiti walls and modern architecture. The time is midday. Lighting is soft and even.",
    "Park - A serene park with lush greenery, a small pond, and benches scattered along winding paths. The weather is overcast.  Lighting is soft and even.",
]

# Define camera information and shot composition
camera_details = ("Capture the photo in the style of a professional Canon EOS R5, F2 aperture, ISO 30, "
                  "35mm focal length, with zoom lens. The image is taken from a distance and from a slightly elevated vantage point, but cropped "
                  "slightly wider than the target person’s full height and width, ensuring their entire body is visible.")

# Load clothing colours from csv
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_INPUT_PATH = os.path.join(SCRIPT_DIR, "frequency_tables", "dominant_colors.csv")
clothing_colors = load_dominant_colors_from_csv(CSV_INPUT_PATH)

accessory_colors = ["Red", "Blue", "Green", "Black", "White", "Yellow", "Purple", "Orange", "Navy", "Beige", "Gray",
                    "Brown", "Olive", "Silver", "Gold"]

hair_colors = ["Brown", "Blonde", "Black", "Dirty blonde", "White", "Gray", "Dark red", "Ginger"]

def generate_prompt():
    # Randomly select a gender first as it influences clothing and accessory options.
    gender = random.choice(["Male", "Female"])

    # Set Attributes
    attributes = {}
    attributes["Age"] = random.choice(attributes_options['Age'])
    attributes["Hair"] = f"{random.choice(hair_colors)} {random.choice(attributes_options['Hair'])}"
    attributes["Skin"] = random.choice(attributes_options['Skin'])
    attributes["Facial Expression"] = random.choice(attributes_options['Facial Expression'])
    attributes["Weight"] = random.choice(attributes_options['Weight'])

    # Set Clothing
    clothing = {}
    clothing["Top"] = f"{random.choice(clothing_colors['top'])} style: {random.choice(clothing_options['Top'][gender])}"
    clothing["Overwear"] = f"Color hex: {random.choice(clothing_colors['outer'])} style: {random.choice(clothing_options['Overwear'][gender])}"

    # Handle mutually exclusive items
    clothing["Dress"] = "None"
    clothing["Skirt"] = "None"
    clothing["Pants"] = "None"
    clothing["Shorts"] = "None"
    available_bottoms_men = ["Pants", "Shorts"]
    available_bottoms_women = ["Dress", "Skirt", "Pants", "Shorts"]

    # Randomly pick one clothing bottoms choice based on gender
    if gender == "Male":
        chosen_bottom = random.choice(available_bottoms_men)
    else:
        chosen_bottom = random.choice(available_bottoms_women)

    bottoms_colors_combined = clothing_colors['skirt'] + clothing_colors['dress'] + clothing_colors['shorts'] + clothing_colors['pants']
    clothing[chosen_bottom] = f"Color hex: {random.choice(bottoms_colors_combined)} style: {random.choice(clothing_options[chosen_bottom][gender])}"


    # Select other independent clothing options
    clothing["Headwear"] = f"{random.choice(clothing_colors['headwear'])} {random.choice(clothing_options['Headwear'][gender])}"
    clothing["Eyewear"] = f"{random.choice(clothing_colors['sunglasses'])} {random.choice(clothing_options['Eyewear'][gender])}"

    clothing["Scarf"] = "None"
    clothing["Tie"] = "None"
    if clothing["Top"] == "Collared Shirt":
        clothing["Tie"] = random.choice(clothing_options["Tie"][gender])
    else:
        # Scarf and tie colours are mixed and limited, accessory colours would give more options
        clothing["Scarf"] = f"{random.choice(accessory_colors)} {random.choice(clothing_options['Scarf'][gender])}"

    clothing["Footwear"] = f"Color hex: {random.choice(clothing_colors['footwear'])} style: {random.choice(clothing_options['Footwear'][gender])}"

    accessory_keys = list(accessory_options.keys())
    chosen_accessory = random.choice(accessory_keys)
    accessory = f"{random.choice(accessory_colors)} {random.choice(accessory_options[chosen_accessory][gender])}"

    # Randomly select a profile and location.
    profile = random.choice(profile_options)
    location = random.choice(location_options)

    # Build the prompt text.
    prompt = "Generate a highly realistic, detailed photograph of a target person with the following attributes:\n"
    prompt += f"\n- Age: {attributes['Age']}"
    prompt += f"\n- Gender: {gender}"
    prompt += f"\n- Hair: {attributes['Hair']}"
    prompt += f"\n- Skin: {attributes['Skin']}"
    prompt += f"\n- Facial Expression: {attributes['Facial Expression']}"
    prompt += f"\n- Weight: {attributes['Weight']}\n"

    clothing_text = "\nThey are wearing the following items of clothing (Use the color hex code for the color of the clothing item):"
    for key, value in clothing.items():
        if "None" not in value:
            clothing_text += f"\n- {key}: {value}"
    prompt += clothing_text

    prompt += "\n\nThey are with or are carrying this accessory or object:"
    prompt += f"\n{accessory} "

    prompt += "\n\nTheir posture is natural and casual. There is only one target person. All attributes of the target person stay consistent, clothing stays consistent, any objects associated with or being carried by the target person stay consistent when generating multiple images."
    prompt += f"\n\nThe profile of the target person in the image is:\n{profile}"
    prompt += f"\n\nLocation:\n{location}"
    prompt += f"\n\n{camera_details}"
    return prompt

if __name__ == "__main__":

    prompt_text = generate_prompt()
    print(prompt_text)
