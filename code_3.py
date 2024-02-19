import requests
from bs4 import BeautifulSoup

# set up the FishBase URL
base_url = "https://www.fishbase.org"
species_url = base_url + "/species/search.do"

# make a request to the species search page
response = requests.get(species_url)

# parse the HTML using BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")

# find the table of species names and images
species_table = soup.find("table", {"class": "species-list"})

# loop through each species in the table
for species_row in species_table.find_all("tr"):
    # extract the species name and image link
    species_name = species_row.find("td", {"class": "species-name"}).text
    image_url = species_row.find("td", {"class": "species-image"}).find("a")["href"]

    # make a request to the image link
    image_response = requests.get(image_url, headers={"Accept": "image/*"})

    # save the image to a file
    with open(f"{species_name.strip('.')}.jpg", "wb") as f:
        f.write(image_response.content)

# output all the species names and images
for species_name in species_table.find_all("td", {"class": "species-name"}):
    print(species_name.text)
