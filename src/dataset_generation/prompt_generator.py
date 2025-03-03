import random

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
    "Suitcase": {
        "Male": ["Travel suitcase"],
        "Female": ["Travel suitcase"]
    },
    "Cell phone": {
        "Male": ["Latest smartphone"],
        "Female": ["Latest smartphone"]
    }
}

# "Bicycle": {
#     "Male": ["Vintage bicycle", "Sport bicycle"],
#     "Female": ["Vintage bicycle", "Sport bicycle"]
# },
# "Skateboard": {
#     "Male": ["bottomed skateboard"],
#     "Female": ["bottomed skateboard"]
# },
# "Umbrella": {
#     "Male": ["umbrella"],
#     "Female": ["umbrella"]
# },

# Profile options for changing the side of the person
profile_options = [
    "A left-side profile angle with the person in motion walking",
    "A right-side profile angle with the person in motion walking",
    "A back profile angle with the person in motion walking",
    "A front facing profile angle with the person in motion walking"
]

# Location options to vary background
# location_options = [
#     "Shopping Centre Hallway - A spacious, modern shopping centre with glossy tiled floors reflecting bright, evenly distributed ceiling lights. Lined with glass storefronts showcasing mannequins and promotional banners. Potted plants and sleek benches enhance the atmosphere. Lighting is soft and even.",
#     "Urban Street - A bustling city street with neon signs, busy traffic, and people hustling by. The environment is vibrant with graffiti walls and modern architecture. The time is midday. Lighting is soft and even.",
#     "Park - A serene park with lush greenery, a small pond, and benches scattered along winding paths. The weather is overcast.  Lighting is soft and even.",
# ]

general_location_options = [
    "Shopping Centre Hallways: A spacious modern mall with polished tiled floors that reflect bright, evenly distributed ceiling lights. Large glass storefronts display mannequins and promotional banners. Potted plants and sleek benches line the walkways, while shoppers stroll through, some carrying branded bags or window-shopping. The lighting is soft and even, ensuring clear visibility throughout the space.",
    "Airport Terminals: A high-traffic international airport corridor with large glass windows allowing in diffused natural daylight. The floor is polished stone, subtly reflecting overhead recessed lighting. Digital flight boards, check-in counters, and waiting passengers with rolling suitcases create a dynamic atmosphere. The lighting remains soft and even, blending natural and artificial illumination seamlessly.",
    "Stadium Hallways: A wide, enclosed stadium concourse with bright LED overhead lights providing soft, even illumination. The floor is non-reflective commercial tile, and the walls are adorned with team banners and event posters. Fans in jerseys walk through, some carrying drinks and snacks, while security personnel stand at various points. The lighting is consistent and evenly distributed, preventing harsh shadows.",
    "Convention Centres: A sleek, upscale convention centre hallway with neutral-toned carpet and floor-to-ceiling glass walls letting in soft daylight. Overhead LED panel lighting ensures consistent, shadow-free illumination. Digital kiosks display event schedules while professionals in business attire walk between conference halls and networking areas. The entire space is bathed in soft, even lighting for a modern and professional ambiance.",
    "University Campuses: A spacious university hallway with smooth, matte-finish floors and walls lined with bulletin boards and digital screens displaying schedules. Large windows diffuse soft natural daylight, blending with recessed ceiling lights for a balanced illumination. Students walk through carrying backpacks, engaged in quiet discussions or checking their phones. The lighting is uniformly soft, creating a comfortable academic atmosphere.",
    "Metro & Train Stations: A bustling underground metro station corridor with bright, even fluorescent lighting. The floors are lightly textured, non-slip tiles with directional signage leading to platforms. Digital billboards display train arrival times as commuters walk briskly, some carrying backpacks or holding transit cards. The lighting is soft and evenly spread, ensuring clear visibility across the station.",
    "Government Buildings: A large, high-security government lobby with polished stone floors and floor-to-ceiling glass panels. The ceiling features recessed LED lights ensuring soft, even illumination. Turnstiles, metal detectors, and security checkpoints are visible, with people lining up for entry. The lighting is bright yet diffused, preventing any harsh highlights or deep shadows.",
    "Corporate Offices: A modern office hallway with frosted glass partitions, matte-finish flooring, and a neutral color scheme. Recessed LED lighting provides even brightness, preventing shadows. Employees in business attire walk through, some engaged in conversation while others check their devices. The lighting is perfectly soft and evenly distributed, enhancing the clean and professional aesthetic.",
    "Hospitals: A clean hospital corridor with glossy but non-glare flooring and soft LED lighting fixtures ensuring uniform brightness. Patient rooms, nurse stations, and emergency exit signs line the hallway. Doctors, nurses, and visitors move through the space, carrying medical files or pushing carts. The lighting is soft and evenly spread, creating a sterile yet welcoming environment.",
    "Parking Garages: A modern indoor parking structure with bright, evenly spaced LED lights eliminating dark corners. The floors are smooth concrete, with painted directional arrows and designated pedestrian walkways. Security cameras are mounted at key entry and exit points, and occasional parked cars are visible. The lighting is evenly distributed, preventing dark spots and enhancing security visibility."
]

location_options = [
    "Check-in Area: A spacious airport check-in area with sleek tiled flooring and bright, evenly distributed lighting. Digital flight information screens hang from the ceiling, displaying departure times. Airline check-in counters with self-service kiosks are arranged in neat rows. Travelers with rolling suitcases wait in line, some scanning passports while others interact with airline staff. The lighting is soft and diffused, ensuring a bright but shadow-free environment.",
    "Security Checkpoint: A modern airport security checkpoint with glass dividers and polished floors reflecting soft overhead LED lighting. Passengers remove shoes, belts, and laptops at stainless steel conveyor belts leading into X-ray scanners. Security personnel in uniform monitor screens while others guide travelers through body scanners. The space is brightly lit with soft, evenly distributed lighting to ensure visibility without harsh glare.",
    "Boarding Gate: A quiet airport boarding gate with rows of cushioned seats facing large floor-to-ceiling windows. The tarmac is visible outside, with airplanes parked at jet bridges. Overhead recessed lighting provides soft, even illumination, complementing the diffused midday natural light entering through the glass walls. A few passengers wait with carry-on bags, some checking phones while others converse quietly.",
    "Duty-Free Shopping Area: A high-end duty-free shopping zone with glossy tiled floors and luxurious storefronts selling cosmetics, perfumes, electronics, and designer brands. Recessed ceiling lighting evenly brightens the space without creating harsh reflections. Shoppers browse displays, some carrying branded shopping bags. Large promotional posters line the walls, and the ambient lighting is warm yet balanced.",
    "Baggage Claim: A spacious baggage claim area with long conveyor belts transporting luggage under soft LED panel lighting. Passengers stand near the carousel, waiting for their suitcases. Large digital displays show flight numbers and arrival details. The space has a modern, clean design with polished flooring, glass barriers, and evenly spread lighting to minimize shadows.",
    "Airport Lounge: A sleek, premium airport lounge with plush seating, wood-paneled walls, and soft, warm LED lighting. Floor-to-ceiling windows let in diffused daylight while ambient ceiling fixtures provide consistent illumination. Passengers relax in lounge chairs, sipping coffee or working on laptops. The atmosphere is quiet, with a mix of modern decor and a luxurious ambiance.",
    "Food Court: A bustling airport food court with a variety of fast food and café-style restaurants. Bright, evenly spread lighting ensures no harsh shadows over the tables and seating areas. Travelers carry food trays, some sitting at modern high-top tables while others queue at counters to order meals. Digital menu boards glow softly above each vendor, displaying meal options.",
    "International Arrivals Hall: A large international arrivals hall with high ceilings, glass walls, and a polished stone floor reflecting the soft midday lighting. Family members and chauffeurs wait near metal barriers, holding signs with passenger names. A large digital screen displays real-time flight arrivals. The even, diffused lighting ensures clarity and visibility for both travelers and greeters.",
    "Outdoor Drop-Off Zone: The busy airport drop-off zone outside the terminal, featuring wide driveways, covered sidewalks, and automatic sliding glass doors leading into the terminal. Taxis, private cars, and shuttle buses arrive in a constant flow, with passengers unloading luggage onto the sidewalk. Bright midday sunlight is diffused by an overhead canopy, creating a soft, even lighting effect.",
    "Jet Bridge: A narrow, enclosed jet bridge with ribbed metallic flooring and bright but evenly spread ceiling lights. The walls are made of reinforced glass and metal, offering a partial view of the tarmac. Passengers walk single-file, carrying personal items and rolling suitcases toward the aircraft door. The lighting is balanced to ensure a clean and professional atmosphere.",
    "Airport Parking Garage: A multi-level indoor airport parking structure with clearly marked spaces, pedestrian walkways, and soft LED ceiling lights providing even illumination. Signs direct passengers to shuttle pickup zones and terminal entrances. Parked cars line the space, while a few travelers roll their luggage toward elevators or exit ramps. The lighting eliminates dark spots, ensuring security and visibility.",
    "VIP Private Terminal: A high-end private airport terminal with a sleek, minimalist design, exclusive seating areas, and concierge service desks. Soft LED lighting highlights the clean, modern decor, while large tinted windows let in diffused daylight. Luxury vehicles wait outside for private jet passengers, and staff members assist VIPs with their luggage in a calm, elegant atmosphere.",
    "Runway Observation Deck: A designated public viewing area overlooking the runway, featuring large floor-to-ceiling glass panels and padded benches. Passengers and aviation enthusiasts watch planes take off and land. The midday sun casts soft, natural light through the windows, merging with evenly spaced ceiling fixtures for a bright but glare-free setting.",
    "Lost & Found Office: A small, well-organized lost and found office within the airport, featuring a modern desk, labeled storage shelves, and an information screen listing found items. A staff member in uniform assists passengers reporting missing luggage. Overhead LED lighting provides soft and even illumination, ensuring all details are clearly visible without creating shadows.",
    "Airport Transit Train Station: A sleek, automated airport transit train station with a futuristic design, clean glass walls, and an open boarding platform. The train, designed to connect terminals, waits with its doors open. The platform is brightly but evenly lit with recessed ceiling lights, ensuring full visibility without harsh reflections. A few travelers wait, some holding rolling suitcases or standing near digital route maps."
]


# Define camera information and shot composition
camera_details = ("Capture the photo in the style of a professional Canon EOS R5, F2 aperture, ISO 30, "
                  "35mm focal length, with zoom lens. The image is taken from a distance and from a slightly elevated vantage point, but cropped "
                  "slightly wider than the target person’s full height and width, ensuring their entire body is visible.")

# # Load clothing colours from csv
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# CSV_INPUT_PATH = os.path.join(SCRIPT_DIR, "frequency_tables", "dominant_colors.csv")
clothing_colors = {
    "Headwear": ["Black", "White", "Blue", "Gray", "Red", "Beige", "Brown"],
    "Eyewear": ["Black", "Brown", "Gold", "Silver", "Blue", "Tortoiseshell"],
    "Scarf": ["Red", "Blue", "Gray", "Black", "White", "Pink", "Beige"],
    "Tie": ["Black", "Red", "Blue", "Gray", "Striped", "Patterned"],
    "Top": ["Black", "White", "Gray", "Blue", "Red", "Green", "Yellow", "Purple"],
    "Overwear": ["Black", "Gray", "Brown", "Beige", "Blue", "Green", "Navy"],
    "Dress": ["Red", "Black", "White", "Blue", "Pink", "Floral", "Pastel"],
    "Skirt": ["Black", "Blue", "Pink", "Red", "White", "Gray", "Green"],
    "Pants": ["Black", "Blue", "Gray", "Brown", "Beige", "Navy", "Olive"],
    "Shorts": ["Black", "Blue", "Gray", "Beige", "White", "Denim"],
    "Footwear": ["Black", "White", "Brown", "Gray", "Beige", "Navy"]
}


accessory_colors = ["Red", "Blue", "Green", "Black", "White", "Yellow", "Purple", "Orange", "Navy", "Beige", "Gray",
                    "Brown", "Olive", "Silver", "Gold"]

hair_colors = ["Brown", "Blonde", "Black", "Dirty blonde", "White", "Gray", "Dark red", "Ginger"]

def generate_prompts():
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
    clothing["Top"] = f"{random.choice(clothing_colors['Top'])} {random.choice(clothing_options['Top'][gender])}"
    clothing["Overwear"] = f"{random.choice(clothing_colors['Overwear'])} {random.choice(clothing_options['Overwear'][gender])}"

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

    bottoms_colors_combined = clothing_colors['Skirt'] + clothing_colors['Dress'] + clothing_colors['Shorts'] + clothing_colors['Pants']
    clothing[chosen_bottom] = f"{random.choice(bottoms_colors_combined)} {random.choice(clothing_options[chosen_bottom][gender])}"


    # Select other independent clothing options
    clothing["Headwear"] = f"{random.choice(clothing_colors['Headwear'])} {random.choice(clothing_options['Headwear'][gender])}"
    clothing["Eyewear"] = f"{random.choice(clothing_colors['Eyewear'])} {random.choice(clothing_options['Eyewear'][gender])}"

    clothing["Scarf"] = "None"
    clothing["Tie"] = "None"
    if clothing["Top"] == "Collared Shirt":
        clothing["Tie"] = random.choice(clothing_options["Tie"][gender])
    else:
        # Scarf and tie colours are mixed and limited, accessory colours would give more options
        clothing["Scarf"] = f"{random.choice(accessory_colors)} {random.choice(clothing_options['Scarf'][gender])}"

    clothing["Footwear"] = f"{random.choice(clothing_colors['Footwear'])} {random.choice(clothing_options['Footwear'][gender])}"

    accessory_keys = list(accessory_options.keys())
    chosen_accessory = random.choice(accessory_keys)
    accessory = f"{random.choice(accessory_colors)} {random.choice(accessory_options[chosen_accessory][gender])}"


    prompts = []
    # Build the prompt text.
    for i in range(4):

        # Randomise location and profile for each prompt generation
        location = random.choice(location_options)
        profile = profile_options[i]

        prompt = "Generate a highly realistic, detailed photograph of a target person with the following attributes:\n"
        prompt += f"\n- Age: {attributes['Age']}"
        prompt += f"\n- Gender: {gender}"
        prompt += f"\n- Hair: {attributes['Hair']}"
        prompt += f"\n- Skin: {attributes['Skin']}"
        prompt += f"\n- Facial Expression: {attributes['Facial Expression']}"
        prompt += f"\n- Weight: {attributes['Weight']}\n"

        clothing_text = "\nThey are wearing the following items of clothing:"
        for key, value in clothing.items():
            if "None" not in value:
                clothing_text += f"\n- {key}: {value}"
        prompt += clothing_text

        prompt += "\n\nThey are with or are carrying this accessory or object:"
        prompt += f"\n{accessory} "

        prompt += "\n\nTheir posture is natural and casual. There is only one target person. All attributes of the target person stay consistent, clothing stays consistent, any objects associated with or being carried by the target person stay consistent when generating multiple images."
        prompt += f"\n\nThe profile of the target person in the image is:\n{profile}"
        prompt += f"\n\nLocation:\n{location}"
        prompt += f"\n\n{camera_details}\n\n"
        prompts.append(prompt)

    return prompts

if __name__ == "__main__":

    prompts_list = generate_prompts()
    for prompt in prompts_list:
        print(prompt)
        print("-" * 30)
