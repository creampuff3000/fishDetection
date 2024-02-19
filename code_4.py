import os
import requests

# Set up the URL and the output file with an absolute path
url = "https://fishbase.us/summary/SpeciesList.php?ID=0&GenusName=&SpeciesName=&fc=365"
output_directory = "/Users/justinluo/projects/fishDetection/"
output_file = "images.txt"
output_path = os.path.join(output_directory, output_file)

# Number of images to scrape
num_images_to_scrape = 200

# Check if the specified directory is writable
if not os.access(output_directory, os.W_OK):
    print(f"Error: The directory '{output_directory}' is not writable.")
    exit()

# Make a request to the API endpoint
response = requests.get(url)

# Check if the response is successful (status code 200)
if response.status_code == 200:
    try:
        # Try to parse the response as JSON
        data = response.json()
        
        # Print the entire API response for debugging
        print(data)

        # Extract image URLs and species names
        images = data.get('aaData', [])[:num_images_to_scrape]

        # Open the output file for writing
        with open(output_path, "w") as f:
            f.write("Image URL,Species\n")

            # Loop through each entry and get the URL and species name
            for entry in images:
                img_url = entry.get('species_pic', 'N/A')
                species = entry.get('Species', 'N/A')

                # Write the URL and species name to the output file
                f.write(f"{img_url},{species}\n")
                print(f"Written: {img_url},{species}")

        print("Processing complete. Check the 'images.txt' file.")
    except requests.exceptions.JSONDecodeError:
        # If parsing as JSON fails, print the response content for analysis
        print(f"Failed to parse JSON. Response content:\n{response.content.decode('utf-8')}")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")
