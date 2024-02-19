import requests 
# Define the FishBase API endpoint to retrieve a list of all fish species
api_url = "https://fishbase.ropensci.org/species/"

# Send a GET request to the API to retrieve the list of all species
response = requests.get(api_url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    species_list = response.json()

    # Loop through the list of species to retrieve image URLs and species names
    for species_data in species_list:
        species_name = species_data["Species"]
        image_url = species_data["image"]
        
        # Download the image or save the image URL as needed
        # You can use the 'requests' library to download the image if necessary

        print(f"Species Name: {species_name}")
        print(f"Image URL: {image_url}")
else:
    print("Failed to retrieve data from FishBase API.")
